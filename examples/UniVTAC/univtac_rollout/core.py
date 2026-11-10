"""Pure-Python contracts and result handling for closed-loop UniVTAC evaluation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np


CANONICAL_TASKS = (
    "lift_bottle",
    "lift_can",
    "insert_HDMI",
    "insert_hole",
    "insert_tube",
    "pull_out_key",
    "put_bottle_in_shelf",
    "grasp_classify",
)

TASK_DISPLAY_NAMES = {
    "lift_bottle": "Lift Bottle",
    "lift_can": "Lift Can",
    "insert_HDMI": "Insert HDMI",
    "insert_hole": "Insert Hole",
    "insert_tube": "Insert Tube",
    "pull_out_key": "Pull Out Key",
    "put_bottle_in_shelf": "Put Bottle in Shelf",
    "grasp_classify": "Grasp & Classify",
}

ERROR_TERMINATIONS = frozenset({"exception", "invalid_action", "worker_failure"})
VALID_TERMINATIONS = frozenset({"success", "step_limit", "early_stop"}) | ERROR_TERMINATIONS
PANDA_GRIPPER_DOF = 7
DEFAULT_GRIPPER_LIMIT_TOLERANCE_M = 1e-4


class InvalidActionError(ValueError):
    """Raised when a predicted qpos action is unsafe or structurally invalid."""


@dataclass(frozen=True)
class EpisodeResult:
    task: str
    seed: int
    episode_index: int
    success: bool | None
    num_steps: int
    execution_horizon: int
    checkpoint: str
    instruction: str
    elapsed_seconds: float
    termination_reason: str
    video_path: str
    sim_steps: int | None = None
    error: str = ""

    def __post_init__(self) -> None:
        if self.task not in CANONICAL_TASKS:
            raise ValueError(f"Unknown UniVTAC task: {self.task!r}")
        if self.termination_reason not in VALID_TERMINATIONS:
            raise ValueError(f"Unknown termination reason: {self.termination_reason!r}")
        if self.success is True and self.termination_reason != "success":
            raise ValueError("A successful episode must use termination_reason='success'.")
        if self.success is False and self.termination_reason == "success":
            raise ValueError("A failed episode cannot use termination_reason='success'.")
        if self.success is None and self.termination_reason not in ERROR_TERMINATIONS:
            raise ValueError("Only exception/invalid_action episodes may have success=None.")


def _to_numpy(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive error context
        raise TypeError(f"{name} cannot be converted to a NumPy array: {type(value)}") from exc


def format_univtac_observation(
    observation: dict[str, Any], instruction: str
) -> dict[str, dict[str, Any]]:
    """Convert one UniVTAC observation to the strict RGB GR00T checkpoint schema."""
    try:
        head = _to_numpy(observation["observation"]["head"]["rgb"], name="head RGB")
        wrist = _to_numpy(observation["observation"]["wrist"]["rgb"], name="wrist RGB")
        joint = _to_numpy(observation["embodiment"]["joint"], name="joint state")
    except KeyError as exc:
        raise KeyError(
            "UniVTAC observation must contain observation.head.rgb, "
            "observation.wrist.rgb, and embodiment.joint. "
            f"Missing component: {exc}"
        ) from exc

    for name, image in (("head", head), ("wrist", wrist)):
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{name} RGB must have shape [H, W, 3]; got {image.shape}.")
        if image.dtype != np.uint8:
            raise TypeError(f"{name} RGB must have dtype uint8; got {image.dtype}.")
        if image.shape[0] < 1 or image.shape[1] < 1:
            raise ValueError(f"{name} RGB has an empty spatial dimension: {image.shape}.")

    joint = np.asarray(joint).reshape(-1)
    if joint.size < 8:
        raise ValueError(
            "UniVTAC embodiment.joint must contain at least 8 coordinates "
            f"(7 Franka joints + gripper); got shape {joint.shape}."
        )
    joint = joint[:8].astype(np.float32, copy=False)
    if not np.isfinite(joint).all():
        raise ValueError("UniVTAC joint observation contains NaN or Inf.")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("task.instruction must be a non-empty string.")

    return {
        "video": {
            "head": np.ascontiguousarray(head[None, None, ...]),
            "wrist": np.ascontiguousarray(wrist[None, None, ...]),
        },
        "state": {"joint": np.ascontiguousarray(joint[None, None, ...])},
        "language": {"annotation.human.task_description": [[instruction]]},
    }


def extract_action_chunk(response: Any, *, expected_dim: int = 8) -> np.ndarray:
    """Extract and validate the decoded ``joint`` chunk returned by PolicyServer."""
    if not isinstance(response, (list, tuple)) or len(response) != 2:
        raise InvalidActionError(
            "GR00T get_action response must be a two-item (action, info) sequence; "
            f"got {type(response).__name__}."
        )
    action = response[0]
    if not isinstance(action, dict) or "joint" not in action:
        keys = sorted(action) if isinstance(action, dict) else []
        raise InvalidActionError(f"GR00T action must contain key 'joint'; got keys {keys}.")

    chunk = np.asarray(action["joint"])
    if chunk.ndim != 3:
        raise InvalidActionError(
            f"GR00T action.joint must have shape [batch, horizon, dim]; got {chunk.shape}."
        )
    if chunk.shape[0] != 1:
        raise InvalidActionError(f"Rollout requires action batch size 1; got {chunk.shape[0]}.")
    if chunk.shape[1] < 1:
        raise InvalidActionError("GR00T returned an empty action horizon.")
    if chunk.shape[2] != expected_dim:
        raise InvalidActionError(
            f"GR00T qpos action must have {expected_dim} dimensions; got {chunk.shape[2]}."
        )
    if not np.issubdtype(chunk.dtype, np.floating):
        raise InvalidActionError(f"GR00T action must be floating point; got {chunk.dtype}.")
    if not np.isfinite(chunk).all():
        bad = np.argwhere(~np.isfinite(chunk))[:8].tolist()
        raise InvalidActionError(f"GR00T action contains NaN/Inf at indices {bad}.")
    return np.ascontiguousarray(chunk[0].astype(np.float32, copy=False))


def validate_action_step(
    action: Any,
    *,
    expected_dim: int = 8,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    gripper_limit_tolerance_m: float = DEFAULT_GRIPPER_LIMIT_TOLERANCE_M,
) -> np.ndarray:
    """Validate one qpos command and correct tiny Panda-gripper limit overshoots.

    Only qpos DOF 7 is corrected, and only when its overshoot is no larger than
    ``gripper_limit_tolerance_m``. All other limit violations remain errors.
    """
    step = np.asarray(action)
    if step.shape != (expected_dim,):
        raise InvalidActionError(
            f"UniVTAC qpos action must have shape ({expected_dim},); got {step.shape}."
        )
    if not np.issubdtype(step.dtype, np.floating):
        raise InvalidActionError(f"UniVTAC qpos action must be floating point; got {step.dtype}.")
    if not np.isfinite(step).all():
        raise InvalidActionError("UniVTAC qpos action contains NaN or Inf.")
    step = step.astype(np.float32, copy=False)

    try:
        gripper_limit_tolerance_m = float(gripper_limit_tolerance_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("Gripper limit tolerance must be a finite nonnegative float.") from exc
    if not math.isfinite(gripper_limit_tolerance_m) or gripper_limit_tolerance_m < 0:
        raise ValueError("Gripper limit tolerance must be a finite nonnegative float.")

    if (lower is None) != (upper is None):
        raise ValueError("Both lower and upper joint limits must be supplied together.")
    if lower is not None and upper is not None:
        lower = np.asarray(lower, dtype=np.float32)
        upper = np.asarray(upper, dtype=np.float32)
        if lower.shape != step.shape or upper.shape != step.shape:
            raise ValueError(
                f"Joint-limit shape mismatch: action={step.shape}, lower={lower.shape}, "
                f"upper={upper.shape}."
            )
        below = np.isfinite(lower) & (step < lower)
        above = np.isfinite(upper) & (step > upper)

        # UniVTAC's 8-D qpos contract is seven Panda arm joints followed by the
        # scalar gripper position. Correct only numerical-scale gripper drift;
        # arm commands and material gripper violations are never clipped.
        if expected_dim > PANDA_GRIPPER_DOF:
            gripper = PANDA_GRIPPER_DOF
            if below[gripper]:
                overshoot = float(lower[gripper]) - float(step[gripper])
                if overshoot <= gripper_limit_tolerance_m:
                    step = step.copy()
                    step[gripper] = lower[gripper]
                    below[gripper] = False
            elif above[gripper]:
                overshoot = float(step[gripper]) - float(upper[gripper])
                if overshoot <= gripper_limit_tolerance_m:
                    step = step.copy()
                    step[gripper] = upper[gripper]
                    above[gripper] = False

        if np.any(below | above):
            indices = np.flatnonzero(below | above).tolist()
            details = [
                f"dof {index}: value={step[index]:.6g}, "
                f"range=[{lower[index]:.6g}, {upper[index]:.6g}]"
                for index in indices
            ]
            raise InvalidActionError(
                "GR00T qpos action exceeds UniVTAC articulation limits: " + "; ".join(details)
            )
    return np.ascontiguousarray(step)


def resolve_tasks(values: Sequence[str]) -> list[str]:
    if not values:
        raise ValueError("At least one task or 'all' is required.")
    if "all" in values:
        if len(values) != 1:
            raise ValueError("Use --tasks all by itself, or list canonical task names.")
        return list(CANONICAL_TASKS)
    unknown = [value for value in values if value not in CANONICAL_TASKS]
    if unknown:
        raise ValueError(
            f"Unknown UniVTAC task(s): {unknown}. Valid tasks: {list(CANONICAL_TASKS)}"
        )
    if len(set(values)) != len(values):
        raise ValueError("--tasks contains duplicate task names.")
    return list(values)


def classify_result(success: bool | None, termination_reason: str) -> str:
    if termination_reason not in VALID_TERMINATIONS:
        raise ValueError(f"Unknown termination reason: {termination_reason!r}")
    if success is True:
        return "success"
    if success is None or termination_reason in ERROR_TERMINATIONS:
        return "error"
    return "failure"


def video_output_path(
    output_dir: str | Path,
    *,
    task: str,
    seed: int,
    success: bool | None,
    termination_reason: str,
) -> Path:
    if task not in CANONICAL_TASKS:
        raise ValueError(f"Unknown UniVTAC task: {task!r}")
    classification = classify_result(success, termination_reason)
    return (
        Path(output_dir)
        / "videos"
        / classification
        / task
        / f"{task}_seed_{seed}_{classification}.mp4"
    )


def _safe_mean(values: Iterable[int | float]) -> float | None:
    values = list(values)
    return float(mean(values)) if values else None


def aggregate_results(
    episodes: Sequence[EpisodeResult], task_order: Sequence[str]
) -> dict[str, Any]:
    """Aggregate task rates; macro is deliberately unweighted across tasks."""
    summaries: list[dict[str, Any]] = []
    for task in task_order:
        task_rows = [row for row in episodes if row.task == task]
        successes = [row for row in task_rows if row.success is True]
        failures = [row for row in task_rows if row.success is False]
        errors = [row for row in task_rows if row.success is None]
        evaluated = len(successes) + len(failures)
        summaries.append(
            {
                "task": task,
                "display_name": TASK_DISPLAY_NAMES[task],
                "successes": len(successes),
                "failures": len(failures),
                "errors": len(errors),
                "evaluated_episodes": evaluated,
                "total_rollouts": len(task_rows),
                "total_completed_episodes": len(task_rows),
                "success_rate": len(successes) / evaluated if evaluated else None,
                "success_rate_percent": 100.0 * len(successes) / evaluated if evaluated else None,
                "error_rate": len(errors) / len(task_rows) if task_rows else None,
                "error_rate_percent": 100.0 * len(errors) / len(task_rows) if task_rows else None,
                "mean_episode_steps": _safe_mean(row.num_steps for row in successes + failures),
                "mean_successful_episode_steps": _safe_mean(row.num_steps for row in successes),
                "mean_failed_episode_steps": _safe_mean(row.num_steps for row in failures),
            }
        )

    rates = [row["success_rate"] for row in summaries if row["success_rate"] is not None]
    macro_average = float(mean(rates)) if len(rates) == len(summaries) and rates else None
    total_successes = sum(row["successes"] for row in summaries)
    total_evaluated = sum(row["evaluated_episodes"] for row in summaries)
    return {
        "tasks": summaries,
        "macro_average": macro_average,
        "macro_average_percent": 100.0 * macro_average if macro_average is not None else None,
        "micro_average": total_successes / total_evaluated if total_evaluated else None,
        "micro_average_percent": 100.0 * total_successes / total_evaluated
        if total_evaluated
        else None,
        "total_successes": total_successes,
        "total_failures": sum(row["failures"] for row in summaries),
        "total_errors": sum(row["errors"] for row in summaries),
        "total_evaluated_episodes": total_evaluated,
        "total_rollouts": len(episodes),
        "total_completed_episodes": len(episodes),
        "error_rate": sum(row["errors"] for row in summaries) / len(episodes)
        if episodes
        else None,
        "error_rate_percent": 100.0 * sum(row["errors"] for row in summaries) / len(episodes)
        if episodes
        else None,
        "episode_counts_differ": len({row["evaluated_episodes"] for row in summaries}) > 1,
    }


def _percent(value: float | None, *, average: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    if not average and math.isclose(value, round(value), abs_tol=1e-9):
        return f"{round(value):.0f}%"
    return f"{value:.2f}%"


def format_summary_text(
    aggregate: dict[str, Any],
    *,
    checkpoint: str,
    execution_horizon: int,
    episodes_per_task: int,
) -> str:
    lines = [
        "UniVTAC Closed-loop Evaluation",
        f"Checkpoint: {checkpoint}",
        f"Execution horizon: {execution_horizon}",
        f"Episodes per task: {episodes_per_task}",
        "",
        "Task                    Success Rate",
        "-------------------------------------",
    ]
    for row in aggregate["tasks"]:
        lines.append(f"{row['display_name']:<24}{_percent(row['success_rate_percent']):>13}")
    lines.append("-------------------------------------")
    lines.append(f"{'Average':<24}{_percent(aggregate['macro_average_percent'], average=True):>13}")
    if aggregate["episode_counts_differ"]:
        lines.append(
            f"{'Overall / Micro Average':<24}"
            f"{_percent(aggregate['micro_average_percent'], average=True):>13}"
        )
    if aggregate["total_errors"]:
        lines.extend(
            [
                "",
                f"Errors excluded from Success Rate: {aggregate['total_errors']}",
                "See episodes.csv and logs/ for exception/invalid_action details.",
            ]
        )
    return "\n".join(lines) + "\n"


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_csv_atomic(
    path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_results(
    output_dir: str | Path,
    episodes: Sequence[EpisodeResult],
    task_order: Sequence[str],
    *,
    checkpoint: str,
    execution_horizon: int,
    episodes_per_task: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_results(episodes, task_order)

    episode_rows = [asdict(row) for row in episodes]
    _write_csv_atomic(
        output_dir / "episodes.csv",
        episode_rows,
        fieldnames=list(EpisodeResult.__dataclass_fields__),
    )

    summary_rows = [dict(row) for row in aggregate["tasks"]]
    summary_rows.append(
        {
            "task": "average",
            "display_name": "Average",
            "successes": aggregate["total_successes"],
            "failures": aggregate["total_failures"],
            "errors": aggregate["total_errors"],
            "evaluated_episodes": aggregate["total_evaluated_episodes"],
            "total_rollouts": aggregate["total_rollouts"],
            "total_completed_episodes": aggregate["total_completed_episodes"],
            "success_rate": aggregate["macro_average"],
            "success_rate_percent": aggregate["macro_average_percent"],
            "error_rate": aggregate["error_rate"],
            "error_rate_percent": aggregate["error_rate_percent"],
            "mean_episode_steps": "",
            "mean_successful_episode_steps": "",
            "mean_failed_episode_steps": "",
        }
    )
    if aggregate["episode_counts_differ"]:
        summary_rows.append(
            {
                **summary_rows[-1],
                "task": "micro_average",
                "display_name": "Overall / Micro Average",
                "success_rate": aggregate["micro_average"],
                "success_rate_percent": aggregate["micro_average_percent"],
            }
        )
    _write_csv_atomic(
        output_dir / "summary.csv",
        summary_rows,
        fieldnames=list(summary_rows[0]),
    )

    summary_text = format_summary_text(
        aggregate,
        checkpoint=checkpoint,
        execution_horizon=execution_horizon,
        episodes_per_task=episodes_per_task,
    )
    _write_text_atomic(output_dir / "summary.txt", summary_text)

    markdown = [
        "# UniVTAC Closed-loop Evaluation",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Execution horizon: {execution_horizon}",
        f"- Episodes per task: {episodes_per_task}",
        "",
        "| Task | Successes | Failures | Errors | Success Rate | Mean Steps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["tasks"]:
        mean_steps = row["mean_episode_steps"]
        markdown.append(
            f"| {row['display_name']} | {row['successes']} | {row['failures']} | "
            f"{row['errors']} | {_percent(row['success_rate_percent'], average=True)} | "
            f"{mean_steps:.2f} |"
            if mean_steps is not None
            else f"| {row['display_name']} | {row['successes']} | {row['failures']} | "
            f"{row['errors']} | N/A | N/A |"
        )
    markdown.extend(
        [
            f"| **Average (macro)** |  |  |  | **{_percent(aggregate['macro_average_percent'], average=True)}** |  |",
            "",
            "The macro average weights each selected task equally. Exception and invalid-action "
            "episodes are reported separately and excluded from the Success Rate denominator.",
        ]
    )
    if aggregate["episode_counts_differ"]:
        markdown.append(
            f"Overall / micro average: **{_percent(aggregate['micro_average_percent'], average=True)}**."
        )
    _write_text_atomic(output_dir / "summary.md", "\n".join(markdown) + "\n")

    payload = {
        "checkpoint": checkpoint,
        "execution_horizon": execution_horizon,
        "episodes_per_task": episodes_per_task,
        "task_order": list(task_order),
        "summary": aggregate,
        "episodes": episode_rows,
        "metadata": metadata or {},
    }
    _write_text_atomic(output_dir / "summary.json", json.dumps(payload, indent=2) + "\n")
    return aggregate
