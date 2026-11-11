"""Metric derivation for WAH Rollout Analyzer."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from statistics import mean
from typing import Any, Iterable

from .parser import Episode, RunData, display_task


def _mean(values: Iterable[int | float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return mean(finite) if finite else None


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2))
    margin /= denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def episode_summary(rows: list[Episode]) -> dict[str, Any]:
    counts = Counter(row.outcome for row in rows)
    successes = counts["success"]
    failures = counts["failure"]
    errors = counts["error"]
    evaluated = successes + failures
    completed = len(rows)
    return {
        "successes": successes,
        "failures": failures,
        "errors": errors,
        "evaluated": evaluated,
        "completed": completed,
        "success_rate": successes / evaluated if evaluated else None,
        "error_rate": errors / completed if completed else None,
        "overall_success_fraction": successes / completed if completed else None,
        "mean_steps": _mean(row.num_steps for row in rows if row.outcome != "error"),
        "mean_duration": _mean(row.elapsed_seconds for row in rows),
        "wilson_95": wilson(successes, evaluated),
    }


def task_summaries(run: RunData) -> list[dict[str, Any]]:
    summaries = []
    for task in run.task_order:
        rows = [row for row in run.episodes if row.task == task]
        summary = episode_summary(rows)
        summaries.append(
            {
                "task": task,
                "display_name": display_task(task),
                **summary,
                "mean_successful_steps": _mean(
                    row.num_steps for row in rows if row.outcome == "success"
                ),
                "mean_failed_steps": _mean(
                    row.num_steps for row in rows if row.outcome == "failure"
                ),
                "termination_reasons": dict(Counter(row.termination_reason for row in rows)),
            }
        )
    return summaries


_ERROR_PATTERNS = (
    ("joint_limit_violation", re.compile(r"articulation limits|joint limit", re.I)),
    ("nonfinite_action", re.compile(r"nan|inf|non.?finite", re.I)),
    ("action_dimension_mismatch", re.compile(r"action.*(?:dimension|shape)|dimensions; got", re.I)),
    ("gripper_boundary", re.compile(r"gripper.*(?:limit|overshoot|boundary)", re.I)),
    (
        "missing_modality",
        re.compile(r"missing|must contain|unavailable|required.*(?:sensor|modality)", re.I),
    ),
    ("inference_timeout", re.compile(r"timed? out|timeout", re.I)),
    ("server_failure", re.compile(r"server|zmq|connection|socket", re.I)),
    ("worker_crash", re.compile(r"worker|signal|return.?code|crash", re.I)),
)


def error_signature(row: Episode) -> str:
    text = row.error.strip()
    for name, pattern in _ERROR_PATTERNS:
        if pattern.search(text):
            return name
    return row.termination_reason if row.termination_reason else "unknown_error"


def error_domain(row: Episode) -> str:
    signature = error_signature(row)
    if signature in {
        "joint_limit_violation",
        "nonfinite_action",
        "action_dimension_mismatch",
        "gripper_boundary",
    }:
        return "invalid model output"
    if signature == "missing_modality":
        return "missing sensor / modality"
    if signature in {"worker_crash", "server_failure", "inference_timeout", "worker_failure"}:
        return "infrastructure failure"
    if row.termination_reason == "exception":
        return "rollout integration issue"
    return "unclassified error"


def error_analysis(run: RunData) -> dict[str, Any]:
    rows = [row for row in run.episodes if row.outcome == "error"]
    groups: dict[str, list[Episode]] = defaultdict(list)
    for row in rows:
        groups[error_signature(row)].append(row)
    return {
        "by_category": dict(Counter(error_signature(row) for row in rows)),
        "by_task": dict(Counter(row.task for row in rows)),
        "by_reason": dict(Counter(row.termination_reason for row in rows)),
        "by_episode_index": dict(Counter(str(row.episode_index) for row in rows)),
        "by_domain": dict(Counter(error_domain(row) for row in rows)),
        "groups": [
            {
                "signature": signature,
                "count": len(values),
                "example": next(
                    (row.error for row in values if row.error), "No error text recorded"
                ),
            }
            for signature, values in sorted(
                groups.items(), key=lambda item: (-len(item[1]), item[0])
            )
        ],
    }


def run_summary(run: RunData) -> dict[str, Any]:
    tasks = task_summaries(run)
    overall = episode_summary(run.episodes)
    rates = [task["success_rate"] for task in tasks if task["success_rate"] is not None]
    # This exactly follows core.aggregate_results: macro is unavailable if any selected task
    # has no valid evaluated episodes.
    macro = mean(rates) if rates and len(rates) == len(tasks) else None
    planned = run.episodes_per_task * len(run.task_order) if run.episodes_per_task else None
    return {
        **overall,
        "macro_success_rate": macro,
        "planned": planned,
        "coverage": overall["evaluated"] / planned if planned else None,
        "unexecuted": run.unexecuted_episodes,
        "wall_clock_seconds": run.wall_clock_seconds,
        "tasks": tasks,
    }


def protocol_warnings(runs: list[RunData]) -> list[str]:
    if len(runs) < 2:
        return []
    warnings = []
    keys = {
        "tasks": "task selections",
        "seed_start": "seed ranges",
        "episodes_per_task": "episode counts",
        "execution_horizon": "execution horizons",
        "step_limit": "step limits",
        "task_config": "task configurations",
        "instruction_type": "instruction types",
        "policy_type": "policy types",
        "dataset": "dataset metadata",
        "environment": "environment metadata",
    }
    for key, label in keys.items():
        known = [
            run.protocol.get(key) for run in runs if run.protocol.get(key) not in (None, "", [])
        ]
        normalized = {repr(value) for value in known}
        if len(normalized) > 1:
            warnings.append(f"Runs differ in {label}.")
    task_sets = [set(run.task_order) for run in runs]
    if any(tasks != task_sets[0] for tasks in task_sets[1:]):
        warnings.append("Missing tasks are shown as N/A and are never treated as zero success.")
    return warnings


def comparison(runs: list[RunData]) -> dict[str, Any]:
    summaries = [run_summary(run) for run in runs]
    task_order = []
    for run in runs:
        for task in run.task_order:
            if task not in task_order:
                task_order.append(task)
    by_run = []
    for run, summary in zip(runs, summaries, strict=True):
        task_map = {row["task"]: row for row in summary["tasks"]}
        inference = [
            row.get("mean_ms")
            for row in run.latency
            if row.get("stage") == "inference.total"
            and isinstance(row.get("mean_ms"), (int, float))
        ]
        by_run.append(
            {
                "label": run.label,
                "summary": summary,
                "tasks": {task: task_map.get(task) for task in task_order},
                "inference_latency_ms": mean(inference) if inference else None,
            }
        )
    episode_maps = [
        {(row.task, row.seed): row for row in run.episodes if row.seed is not None} for run in runs
    ]
    common = set.intersection(*(set(values) for values in episode_maps)) if episode_maps else set()
    matched = []
    for task, seed in sorted(common):
        outcomes = [values[(task, seed)] for values in episode_maps]
        changed = len({row.outcome for row in outcomes}) > 1
        transition = "unchanged"
        if changed and len(outcomes) >= 2:
            transition = f"{outcomes[0].outcome} → {outcomes[1].outcome}"
        matched.append(
            {
                "task": task,
                "seed": seed,
                "changed": changed,
                "transition": transition,
                "episodes": outcomes,
            }
        )
    return {
        "runs": by_run,
        "tasks": task_order,
        "protocol_warnings": protocol_warnings(runs),
        "matched": matched,
        "paired": bool(matched),
    }


def public_episode(row: Episode, run_index: int) -> dict[str, Any]:
    violations = row.diagnostics.get("joint_limit_violations")
    dofs = [str(value.get("dof")) for value in violations or [] if value.get("dof") is not None]
    return {
        "id": f"r{run_index}-{row.task}-{row.episode_index}",
        "run": run_index,
        "task": row.task,
        "task_display": display_task(row.task),
        "episode_index": row.episode_index,
        "seed": row.seed,
        "outcome": row.outcome,
        "termination_reason": row.termination_reason,
        "num_steps": row.num_steps,
        "sim_steps": row.sim_steps,
        "execution_horizon": row.execution_horizon,
        "elapsed_seconds": row.elapsed_seconds,
        "video_available": row.video_available,
        "video_url": row.video_url,
        "error": row.error,
        "error_summary": row.error.strip().splitlines()[-1][:240] if row.error else "",
        "instruction": row.instruction,
        "checkpoint": row.checkpoint,
        "diagnostics": row.diagnostics,
        "error_signature": error_signature(row) if row.outcome == "error" else "",
        "error_domain": error_domain(row) if row.outcome == "error" else "valid task result",
        "dofs": dofs,
        "log_url": row.log_url,
    }
