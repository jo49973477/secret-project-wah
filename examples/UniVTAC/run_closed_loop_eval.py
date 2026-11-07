#!/usr/bin/env python3
"""Orchestrate closed-loop GR00T rollouts across UniVTAC tasks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any

from univtac_rollout.core import (
    DEFAULT_GRIPPER_LIMIT_TOLERANCE_M,
    EpisodeResult,
    resolve_tasks,
    write_results,
)
from univtac_rollout.policy import Gr00tUniVTACPolicy

# Allow the rollout driver to emit progress events without depending on the
# model environment or on any optional terminal UI package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from univtac_progress import emit_event  # noqa: E402


MODE_EPISODES = {"smoke": 5, "quick": 20, "full": 100}


@dataclass(frozen=True)
class ProcessOutcome:
    return_code: int
    signal_name: str | None
    elapsed_seconds: float


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--univtac-root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-identity",
        help="Server SHA-256 identity when checkpoint weights are remote-only.",
    )
    parser.add_argument("--policy", choices=("auto", "rgb", "wah"), default="auto")
    parser.add_argument("--sensor-contract", type=Path)
    parser.add_argument("--scene-calibration", type=Path)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--mode", choices=tuple(MODE_EPISODES), default="smoke")
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--task-config", default="clean")
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default="seen")
    parser.add_argument("--step-limit", type=int)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--gripper-limit-tolerance-m",
        type=float,
        default=DEFAULT_GRIPPER_LIMIT_TOLERANCE_M,
        help="Maximum Panda gripper DOF 7 limit overshoot corrected at execution time.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/univtac_rollout_eval"))
    parser.add_argument(
        "--python-command",
        default=sys.executable,
        help="Isaac Python command, e.g. '/path/to/IsaacLab/isaaclab.sh -p'.",
    )
    parser.add_argument(
        "--expected-python",
        type=Path,
        help="Exact Python executable the driver and worker must use.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--debug-first-rollout", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def _validate(args: argparse.Namespace) -> tuple[list[str], int]:
    tasks = resolve_tasks(args.tasks)
    episodes = (
        args.episodes_per_task if args.episodes_per_task is not None else MODE_EPISODES[args.mode]
    )
    if episodes < 1:
        raise ValueError("--episodes-per-task must be positive.")
    if args.execution_horizon < 1:
        raise ValueError("--execution-horizon must be positive.")
    if args.seed_start < 0:
        raise ValueError("--seed-start must be nonnegative.")
    if args.request_timeout_seconds <= 0:
        raise ValueError("--request-timeout-seconds must be positive.")
    if not math.isfinite(args.gripper_limit_tolerance_m) or args.gripper_limit_tolerance_m < 0:
        raise ValueError("--gripper-limit-tolerance-m must be finite and nonnegative.")
    if args.expected_python is not None:
        expected = args.expected_python.expanduser().absolute()
        actual = Path(sys.executable).absolute()
        if not expected.is_file():
            raise FileNotFoundError(f"--expected-python does not exist: {expected}")
        if actual != expected:
            raise RuntimeError(
                "Wrong Python environment selected for the rollout driver: "
                f"expected {expected}, running {actual}. Deactivate the unrelated Conda/venv "
                "or invoke Isaac Lab with those environment variables unset."
            )
    root = args.univtac_root.resolve()
    required = [root / "envs" / "_base_task.py", root / "task_config", root / "instructions"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"--univtac-root is not a UniVTAC checkout; missing: {missing}")
    worker = Path(__file__).with_name("run_closed_loop_worker.py")
    if not worker.is_file():
        raise FileNotFoundError(f"Rollout worker is missing: {worker}")
    return tasks, episodes


def _load_worker_result(
    path: Path,
) -> tuple[list[EpisodeResult], dict[str, Any], bool, str, str]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    return (
        [EpisodeResult(**row) for row in payload["episodes"]],
        payload.get("runtime_metadata", {}),
        payload.get("completed") is True,
        str(payload.get("phase", "unknown")),
        str(payload.get("fatal_error", "")),
    )


def _stream_process(command: list[str], *, cwd: Path, log_path: Path) -> ProcessOutcome:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Command: " + shlex.join(command) + "\n\n")
        log.flush()
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            # This stdout is redirected into rollout.log by local/split mode.
            # Flush each worker event so the parent can show live progress.
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
        elapsed = time.perf_counter() - started
        signal_name = None
        if return_code < 0:
            try:
                signal_name = signal.Signals(-return_code).name
            except ValueError:
                signal_name = f"signal {-return_code}"
        footer = (
            "\n[orchestrator] worker process finished: "
            f"pid={process.pid} return_code={return_code} "
            f"signal={signal_name or 'none'} elapsed_seconds={elapsed:.3f}\n"
        )
        print(footer, end="", flush=True)
        log.write(footer)
        log.flush()
        return ProcessOutcome(return_code, signal_name, elapsed)


def _worker_failure_result(
    args: argparse.Namespace,
    *,
    task: str,
    episode_index: int,
    error: str,
) -> EpisodeResult:
    return EpisodeResult(
        task=task,
        seed=args.seed_start + episode_index,
        episode_index=episode_index,
        success=None,
        num_steps=0,
        sim_steps=0,
        execution_horizon=args.execution_horizon,
        checkpoint=args.checkpoint,
        instruction="",
        elapsed_seconds=0.0,
        termination_reason="worker_failure",
        video_path="",
        error=error,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    tasks, episodes_per_task = _validate(args)
    output_dir = args.output_dir.resolve()
    result_dir = output_dir / "logs" / "worker_results"
    existing = [
        path for path in (output_dir / "episodes.csv", output_dir / "summary.json") if path.exists()
    ]
    existing.extend(path for path in result_dir.glob("*.json") if path.exists())
    if existing:
        raise FileExistsError(
            "Refusing to mix a new evaluation with existing results. Choose a new "
            f"--output-dir. Existing paths: {[str(path) for path in existing]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Checking GR00T server {args.server_host}:{args.server_port} and checkpoint contract...",
        flush=True,
    )
    with Gr00tUniVTACPolicy(
        args.server_host,
        args.server_port,
        execution_horizon=args.execution_horizon,
        timeout_ms=round(args.request_timeout_seconds * 1000),
        checkpoint=args.checkpoint,
        expected_checkpoint_identity=getattr(args, "checkpoint_identity", None),
        policy=getattr(args, "policy", "auto"),
        sensor_contract=getattr(args, "sensor_contract", None),
        scene_calibration=getattr(args, "scene_calibration", None),
    ) as connected_policy:
        server_info = getattr(connected_policy, "model_info", None)

    python_command = shlex.split(args.python_command)
    if not python_command:
        raise ValueError("--python-command cannot be empty.")
    worker = Path(__file__).with_name("run_closed_loop_worker.py").resolve()
    all_episodes: list[EpisodeResult] = []
    runtime_metadata: dict[str, Any] = {}

    emit_event("start", total=len(tasks) * episodes_per_task)

    for task_index, task in enumerate(tasks):
        emit_event("task", task=task, index=task_index + 1, task_count=len(tasks))
        result_json = result_dir / f"{task}.json"
        command = [
            *python_command,
            str(worker),
            "--univtac-root",
            str(args.univtac_root.resolve()),
            "--task",
            task,
            "--task-config",
            args.task_config,
            "--instruction-type",
            args.instruction_type,
            "--episodes",
            str(episodes_per_task),
            "--seed-start",
            str(args.seed_start),
            "--execution-horizon",
            str(args.execution_horizon),
            "--checkpoint",
            args.checkpoint,
            "--policy",
            getattr(args, "policy", "auto"),
            "--server-host",
            args.server_host,
            "--server-port",
            str(args.server_port),
            "--request-timeout-seconds",
            str(args.request_timeout_seconds),
            "--gripper-limit-tolerance-m",
            str(args.gripper_limit_tolerance_m),
            "--output-dir",
            str(output_dir),
            "--result-json",
            str(result_json),
            "--device",
            args.device,
        ]
        if getattr(args, "checkpoint_identity", None):
            command.extend(("--checkpoint-identity", args.checkpoint_identity))
        if getattr(args, "sensor_contract", None) is not None:
            command.extend(("--sensor-contract", str(args.sensor_contract.resolve())))
        if getattr(args, "scene_calibration", None) is not None:
            command.extend(("--scene-calibration", str(args.scene_calibration.resolve())))
        if args.headless:
            command.append("--headless")
        if args.expected_python is not None:
            command.extend(("--expected-python", str(args.expected_python.expanduser().absolute())))
        command.append(
            "--debug-first-rollout"
            if args.debug_first_rollout and task_index == 0
            else "--no-debug-first-rollout"
        )
        if args.step_limit is not None:
            command.extend(("--step-limit", str(args.step_limit)))

        print(
            f"\nStarting UniVTAC task {task} ({task_index + 1}/{len(tasks)})",
            flush=True,
        )
        outcome = _stream_process(
            command,
            cwd=args.univtac_root.resolve(),
            log_path=output_dir / "logs" / f"{task}.log",
        )
        task_episodes: list[EpisodeResult] = []
        task_metadata: dict[str, Any] = {}
        completed = False
        phase = "no_worker_result"
        fatal_error = ""
        result_error = ""
        if result_json.is_file():
            try:
                task_episodes, task_metadata, completed, phase, fatal_error = _load_worker_result(
                    result_json
                )
            except Exception as exc:
                result_error = f"Could not read worker result {result_json}: {exc}"
        else:
            result_error = f"Worker did not create its result file: {result_json}"

        process_metadata = asdict(outcome)
        runtime_metadata[task] = {
            **task_metadata,
            "worker_phase": phase,
            "worker_completed": completed,
            "worker_fatal_error": fatal_error,
            "process": process_metadata,
        }
        expected_indices = set(range(episodes_per_task))
        observed_indices = {row.episode_index for row in task_episodes if row.task == task}
        invalid_rows = [
            row
            for row in task_episodes
            if row.task != task
            or row.episode_index not in expected_indices
            or row.seed != args.seed_start + row.episode_index
            or row.execution_horizon != args.execution_horizon
        ]
        duplicate_indices = len(observed_indices) != len(task_episodes)
        missing_videos = []
        for row in task_episodes:
            if row.success is None:
                continue
            video_path = Path(row.video_path)
            if not video_path.is_absolute():
                video_path = output_dir / video_path
            if not row.video_path or not video_path.is_file() or video_path.stat().st_size <= 0:
                missing_videos.append(row.episode_index)
        issues = []
        if outcome.return_code != 0:
            issues.append(
                f"worker return_code={outcome.return_code}"
                + (f" ({outcome.signal_name})" if outcome.signal_name else "")
            )
        if not completed:
            issues.append(f"worker did not complete (last phase: {phase})")
        if fatal_error:
            issues.append(f"worker fatal error:\n{fatal_error}")
        if result_error:
            issues.append(result_error)
        if invalid_rows or duplicate_indices:
            issues.append(
                "worker result has duplicate, unexpected-task, out-of-range, wrong-seed, "
                "or wrong-horizon episodes"
            )
        if missing_videos:
            issues.append(
                "completed success/failure episodes have missing or empty MP4 files at indices "
                f"{missing_videos}"
            )
        missing_indices = sorted(expected_indices - observed_indices)
        if missing_indices:
            issues.append(f"worker omitted episode indices {missing_indices}")

        failure_detail = "\n".join(issues) or "worker did not produce a valid episode result"
        all_episodes.extend(task_episodes)
        all_episodes.extend(
            _worker_failure_result(
                args,
                task=task,
                episode_index=episode_index,
                error=failure_detail,
            )
            for episode_index in missing_indices
        )

        run_incomplete = bool(issues)
        write_results(
            output_dir,
            all_episodes,
            tasks,
            checkpoint=args.checkpoint,
            execution_horizon=args.execution_horizon,
            episodes_per_task=episodes_per_task,
            metadata={
                "mode": args.mode,
                "seed_start": args.seed_start,
                "server": f"{args.server_host}:{args.server_port}",
                "gripper_limit_tolerance_m": args.gripper_limit_tolerance_m,
                "evaluation_status": "incomplete" if run_incomplete else "running",
                "runtime_by_task": runtime_metadata,
            },
        )
        if run_incomplete:
            raise RuntimeError(
                f"UniVTAC worker for {task} did not complete a valid rollout: "
                f"{failure_detail}. See {output_dir / 'logs' / f'{task}.log'} and "
                f"{output_dir / 'summary.json'}"
            )

    aggregate = write_results(
        output_dir,
        all_episodes,
        tasks,
        checkpoint=args.checkpoint,
        execution_horizon=args.execution_horizon,
        episodes_per_task=episodes_per_task,
        metadata={
            "mode": args.mode,
            "seed_start": args.seed_start,
            "server": f"{args.server_host}:{args.server_port}",
            "gripper_limit_tolerance_m": args.gripper_limit_tolerance_m,
            "evaluation_status": "completed_with_errors"
            if any(row.success is None for row in all_episodes)
            else "completed",
            "runtime_by_task": runtime_metadata,
            "model_info": server_info,
        },
    )
    print("\n" + (output_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"Results: {output_dir}")
    return aggregate


def evaluation_exit_code(aggregate: dict[str, Any]) -> int:
    """Errors are infrastructure/evaluation failures; ordinary task failures are valid results."""
    return 1 if aggregate["total_errors"] > 0 else 0


def main() -> None:
    aggregate = run(_parser().parse_args())
    raise SystemExit(evaluation_exit_code(aggregate))


if __name__ == "__main__":
    main()
