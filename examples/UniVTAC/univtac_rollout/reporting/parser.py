"""Crash-tolerant readers for UniVTAC rollout artifacts.

``summary.json`` is authoritative. CSV and atomic worker results are deliberately
fallbacks only, which prevents partially-written artifacts from double-counting
episodes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable

from ..core import CANONICAL_TASKS, ERROR_TERMINATIONS, TASK_DISPLAY_NAMES


MAX_EVENT_BYTES = 1_000_000
MAX_TELEMETRY_POINTS = 2_000
MAX_IMPORTANT_EVENTS = 10_000


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _boolean(value: Any) -> bool | None:
    if value is True or value in ("True", "true", "1", 1):
        return True
    if value is False or value in ("False", "false", "0", 0):
        return False
    return None


def _safe_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


@dataclass
class Episode:
    task: str
    episode_index: int
    seed: int | None
    outcome: str
    termination_reason: str
    num_steps: int | None
    sim_steps: int | None
    execution_horizon: int | None
    elapsed_seconds: float | None
    video_path: str = ""
    video_url: str = ""
    video_available: bool = False
    error: str = ""
    instruction: str = ""
    checkpoint: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
    executed: bool | None = None
    log_url: str = ""

    @property
    def key(self) -> tuple[str, int, int | None]:
        return self.task, self.episode_index, self.seed


@dataclass
class RunData:
    root: Path
    label: str
    checkpoint: str
    checkpoint_identity: str | None
    policy_type: str | None
    sidecar_type: str | None
    execution_horizon: int | None
    episodes_per_task: int | None
    task_order: list[str]
    metadata: dict[str, Any]
    episodes: list[Episode]
    latency: list[dict[str, Any]]
    telemetry: list[dict[str, Any]]
    warnings: list[str]
    unexecuted_episodes: int | None
    wall_clock_seconds: float | None
    protocol: dict[str, Any]
    malformed_telemetry_records: int = 0
    source: str = "none"


def _outcome(row: dict[str, Any]) -> str:
    success = _boolean(row.get("success"))
    reason = str(row.get("termination_reason") or "unknown")
    if success is True:
        return "success"
    if success is None or reason in ERROR_TERMINATIONS:
        return "error"
    return "failure"


def _episode(row: dict[str, Any]) -> Episode | None:
    task = str(row.get("task") or "")
    index = _integer(row.get("episode_index"))
    if not task or index is None:
        return None
    return Episode(
        task=task,
        episode_index=index,
        seed=_integer(row.get("seed")),
        outcome=_outcome(row),
        termination_reason=str(row.get("termination_reason") or "unknown"),
        num_steps=_integer(row.get("num_steps")),
        sim_steps=_integer(row.get("sim_steps")),
        execution_horizon=_integer(row.get("execution_horizon")),
        elapsed_seconds=_number(row.get("elapsed_seconds")),
        video_path=str(row.get("video_path") or ""),
        error=str(row.get("error") or ""),
        instruction=str(row.get("instruction") or ""),
        checkpoint=str(row.get("checkpoint") or ""),
    )


def _episodes_from_csv(path: Path) -> list[Episode]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return [episode for row in csv.DictReader(stream) if (episode := _episode(row))]
    except (OSError, UnicodeError, csv.Error):
        return []


def _episodes_from_workers(root: Path) -> list[Episode]:
    rows: list[Episode] = []
    for path in sorted((root / "logs" / "worker_results").glob("*.json")):
        payload = _safe_json(path)
        if payload is None or not isinstance(payload.get("episodes"), list):
            continue
        rows.extend(
            episode
            for row in payload["episodes"]
            if isinstance(row, dict) and (episode := _episode(row))
        )
    return rows


def _deduplicate(rows: Iterable[Episode]) -> list[Episode]:
    unique: dict[tuple[str, int, int | None], Episode] = {}
    for row in rows:
        unique.setdefault(row.key, row)
    return sorted(
        unique.values(), key=lambda item: (item.task, item.episode_index, item.seed or -1)
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _resolve_artifact(root: Path, raw: str) -> Path | None:
    if not raw or "\x00" in raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate = candidate.resolve()
    except OSError:
        return None
    return candidate if _inside(candidate, root) else None


def _relative_url(path: Path, output_dir: Path) -> str:
    return Path(os.path.relpath(path, output_dir)).as_posix()


def _read_events(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Stream and deterministically downsample telemetry without unbounded memory."""
    ordinary: list[dict[str, Any]] = []
    important_events: list[dict[str, Any]] = []
    ordinary_seen = 0
    important_seen = 0
    malformed = 0
    important = {
        "rollout_started",
        "episode_started",
        "episode_completed",
        "task_failed",
        "rollout_completed",
        "rollout_failed",
    }
    try:
        with path.open("rb") as stream:
            for raw in stream:
                if len(raw) > MAX_EVENT_BYTES:
                    malformed += 1
                    continue
                try:
                    event = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    malformed += 1
                    continue
                if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                    malformed += 1
                    continue
                if event["event"] in important or event.get("error_category"):
                    important_seen += 1
                    if len(important_events) < MAX_IMPORTANT_EVENTS:
                        important_events.append(event)
                    else:
                        important_events[important_seen % MAX_IMPORTANT_EVENTS] = event
                    continue
                if event["event"] in {
                    "episode_progress",
                    "inference_completed",
                    "system_diagnostics",
                }:
                    ordinary_seen += 1
                    if len(ordinary) < MAX_TELEMETRY_POINTS:
                        ordinary.append(event)
                    else:
                        ordinary[ordinary_seen % MAX_TELEMETRY_POINTS] = event
    except OSError:
        return [], 0
    selected = important_events + ordinary
    selected.sort(key=lambda row: _number(row.get("monotonic_seconds")) or 0)
    return selected, malformed


def _flatten_latency(payload: dict[str, Any] | None, source: str) -> list[dict[str, Any]]:
    if not payload:
        return []
    rows: list[dict[str, Any]] = []

    def add(stages: Any, prefix: str = "") -> None:
        if not isinstance(stages, dict):
            return
        for name, values in stages.items():
            if not isinstance(values, dict):
                continue
            if "count" in values:
                rows.append({"stage": f"{prefix}{name}", "source": source, **values})
            else:
                add(values, f"{prefix}{name} / ")

    add(payload.get("stages"))
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        add(metadata.get("worker_profiles"), "worker / ")
    return rows


_JOINT_LIMIT = re.compile(
    r"dof\s+(?P<dof>\d+):\s*value=(?P<value>[-+0-9.eE]+),\s*"
    r"range=\[(?P<lower>[-+0-9.eE]+),\s*(?P<upper>[-+0-9.eE]+)\]",
    re.IGNORECASE,
)


def extract_error_diagnostics(error: str) -> dict[str, Any]:
    """Extract only measurements explicitly recorded in an error string."""
    matches = []
    for match in _JOINT_LIMIT.finditer(error):
        value = _number(match.group("value"))
        lower = _number(match.group("lower"))
        upper = _number(match.group("upper"))
        violation = None
        if value is not None and lower is not None and upper is not None:
            violation = max(lower - value, value - upper, 0.0)
        matches.append(
            {
                "dof": int(match.group("dof")),
                "predicted_qpos": value,
                "lower": lower,
                "upper": upper,
                "violation_magnitude": violation,
            }
        )
    return {"joint_limit_violations": matches} if matches else {}


def _event_diagnostics(episodes: list[Episode], events: list[dict[str, Any]]) -> int | None:
    by_index = {(row.task, row.episode_index): row for row in episodes}
    unexecuted: int | None = None
    for event in events:
        kind = event.get("event")
        key = (str(event.get("task") or ""), _integer(event.get("episode_index")))
        episode = by_index.get(key)
        if episode is not None:
            diagnostics = event.get("diagnostics")
            if isinstance(diagnostics, dict):
                episode.diagnostics.update(diagnostics)
            if "executed" in event:
                episode.executed = bool(event["executed"])
        if kind == "task_failed":
            count = _integer(event.get("unexecuted_episodes"))
            if count is not None:
                unexecuted = (unexecuted or 0) + count
    for episode in episodes:
        episode.diagnostics.update(extract_error_diagnostics(episode.error))
    return unexecuted


def _wall_clock(events: list[dict[str, Any]]) -> float | None:
    started = next((row for row in events if row.get("event") == "rollout_started"), None)
    completed = next(
        (
            row
            for row in reversed(events)
            if row.get("event") in {"rollout_completed", "rollout_failed"}
        ),
        None,
    )
    if started and completed:
        first = _number(started.get("monotonic_seconds"))
        last = _number(completed.get("monotonic_seconds"))
        if first is not None and last is not None and last >= first:
            return last - first
    # Summed episode durations are not a wall-clock measurement: setup, task
    # transitions, and process startup would be omitted. Keep this unknown.
    return None


def _first_event(events: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next((row for row in events if row.get("event") == name), {})


def load_run(
    run_dir: str | Path, *, label: str | None = None, output_dir: str | Path | None = None
) -> RunData:
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Rollout directory does not exist: {root}")
    summary_path = root / "summary.json"
    payload = _safe_json(summary_path)
    warnings: list[str] = []
    source = "summary.json"
    if payload is not None and isinstance(payload.get("episodes"), list):
        episodes = [
            episode
            for row in payload["episodes"]
            if isinstance(row, dict) and (episode := _episode(row))
        ]
    elif (root / "episodes.csv").is_file():
        source = "episodes.csv fallback"
        episodes = _episodes_from_csv(root / "episodes.csv")
        warnings.append(
            "summary.json was missing or malformed; episodes.csv was used as a fallback."
        )
        payload = payload or {}
    else:
        source = "worker result fallback"
        episodes = _episodes_from_workers(root)
        warnings.append(
            "Final summary artifacts were unavailable; atomic worker results were used."
        )
        payload = payload or {}
    episodes = _deduplicate(episodes)
    event_path = root / "logs" / "rollout_events.jsonl"
    telemetry, malformed = _read_events(event_path)
    if malformed:
        warnings.append(f"Ignored {malformed} malformed telemetry record(s).")
    unexecuted = _event_diagnostics(episodes, telemetry)
    output_path = Path(output_dir).resolve() if output_dir else root
    for episode in episodes:
        artifact = _resolve_artifact(root, episode.video_path)
        if artifact is not None and artifact.is_file() and artifact.suffix.lower() == ".mp4":
            episode.video_available = True
            episode.video_url = _relative_url(artifact, output_path)
        log = root / "logs" / f"{episode.task}.log"
        if log.is_file() and _inside(log, root):
            episode.log_url = _relative_url(log, output_path)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    runtime = _safe_json(root / "runtime_metadata.json") or {}
    if runtime:
        metadata = {**runtime, **metadata}
    started = _first_event(telemetry, "rollout_started")
    model_info = metadata.get("model_info") if isinstance(metadata.get("model_info"), dict) else {}
    checkpoint = str(payload.get("checkpoint") or started.get("checkpoint") or "")
    task_order = [str(value) for value in payload.get("task_order", []) if isinstance(value, str)]
    if not task_order:
        task_order = [task for task in CANONICAL_TASKS if any(row.task == task for row in episodes)]
        task_order.extend(sorted({row.task for row in episodes} - set(task_order)))
    episodes_per_task = _integer(payload.get("episodes_per_task"))
    planned = episodes_per_task * len(task_order) if episodes_per_task is not None else None
    if unexecuted is None and planned is not None and len(episodes) < planned:
        unexecuted = planned - len(episodes)
    latency = []
    latency.extend(_flatten_latency(_safe_json(root / "logs" / "latency_metrics.json"), "runtime"))
    latency.extend(
        _flatten_latency(_safe_json(root / "logs" / "monitor_latency_metrics.json"), "monitoring")
    )
    runtime_by_task = metadata.get("runtime_by_task")
    task_configs = (
        sorted(
            {
                str(value.get("task_config"))
                for value in runtime_by_task.values()
                if isinstance(value, dict) and value.get("task_config")
            }
        )
        if isinstance(runtime_by_task, dict)
        else []
    )
    protocol = {
        "tasks": task_order,
        "seed_start": metadata.get("seed_start"),
        "episodes_per_task": episodes_per_task,
        "execution_horizon": _integer(payload.get("execution_horizon")),
        "step_limit": metadata.get("step_limit"),
        "task_config": metadata.get("task_config") or task_configs,
        "instruction_type": metadata.get("instruction_type"),
        "policy_type": started.get("policy_type") or model_info.get("policy_type"),
        "dataset": metadata.get("dataset") or metadata.get("dataset_path"),
        "environment": metadata.get("environment") or metadata.get("univtac_commit"),
    }
    return RunData(
        root=root,
        label=label or root.name or "UniVTAC Rollout",
        checkpoint=checkpoint,
        checkpoint_identity=str(
            started.get("checkpoint_identity") or model_info.get("checkpoint_identity") or ""
        )
        or None,
        policy_type=str(started.get("policy_type") or model_info.get("policy_type") or "") or None,
        sidecar_type=str(started.get("sidecar_type") or model_info.get("sidecar_type") or "")
        or None,
        execution_horizon=_integer(payload.get("execution_horizon")),
        episodes_per_task=episodes_per_task,
        task_order=task_order,
        metadata=metadata,
        episodes=episodes,
        latency=latency,
        telemetry=telemetry,
        warnings=warnings,
        unexecuted_episodes=unexecuted,
        wall_clock_seconds=_wall_clock(telemetry),
        protocol=protocol,
        malformed_telemetry_records=malformed,
        source=source,
    )


def display_task(task: str) -> str:
    return TASK_DISPLAY_NAMES.get(task, task.replace("_", " ").title())


def generated_at() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
