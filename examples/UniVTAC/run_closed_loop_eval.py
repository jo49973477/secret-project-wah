#!/usr/bin/env python3
"""Orchestrate closed-loop GR00T rollouts across UniVTAC tasks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from univtac_rollout.core import (
    DEFAULT_GRIPPER_LIMIT_TOLERANCE_M,
    EpisodeResult,
    classify_result,
    resolve_tasks,
    write_results,
)
from univtac_rollout.dashboard import RolloutDashboard
from univtac_rollout.policy import Gr00tUniVTACPolicy
from univtac_rollout.profiling import GLOBAL_PROFILER
from univtac_rollout.telemetry import (
    DIAGNOSTIC_LEVELS,
    LOG_LEVELS,
    EventReader,
    EventWriter,
    RolloutAggregator,
)


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
        "--event-log",
        type=Path,
        help="Structured JSONL telemetry path (default: OUTPUT/logs/rollout_events.jsonl).",
    )
    parser.add_argument("--rollout-diagnostics", choices=DIAGNOSTIC_LEVELS, default="standard")
    parser.add_argument("--diagnostic-sample-every", type=int, default=10)
    parser.add_argument("--dashboard", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--dashboard-refresh", type=float, default=1.0)
    parser.add_argument("--episode-telemetry-interval", type=float, default=1.0)
    parser.add_argument("--stall-warning-seconds", type=float, default=30.0)
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="normal")
    parser.add_argument("--profile-runtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-sync-profile", action=argparse.BooleanOptionalAction, default=False)
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
    if getattr(args, "diagnostic_sample_every", 10) < 1:
        raise ValueError("--diagnostic-sample-every must be positive.")
    dashboard_refresh = getattr(args, "dashboard_refresh", 1.0)
    if not math.isfinite(dashboard_refresh) or dashboard_refresh <= 0:
        raise ValueError("--dashboard-refresh must be finite and positive.")
    for name in ("episode_telemetry_interval", "stall_warning_seconds"):
        value = getattr(args, name, 1.0)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")
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


def _stream_process(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    on_output: Any = None,
    echo: bool = True,
    poll_seconds: float = 0.2,
) -> ProcessOutcome:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", buffering=64 * 1024) as log:
        log.write("Command: " + shlex.join(command) + "\n\n")
        log.flush()
        environment = os.environ.copy()
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
        output_queue: queue.Queue[str | None] = queue.Queue(maxsize=1024)

        def read_output() -> None:
            try:
                for child_line in process.stdout:
                    output_queue.put(child_line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=read_output, name="worker-stdout", daemon=True)
        reader.start()
        output_finished = False
        last_log_flush = time.monotonic()
        while not output_finished or process.poll() is None:
            try:
                line = output_queue.get(timeout=poll_seconds)
            except queue.Empty:
                line = ""
            if line is None:
                output_finished = True
            elif line:
                if echo:
                    print(line, end="")
                write_started_ns = time.perf_counter_ns()
                log.write(line)
                GLOBAL_PROFILER.record_ns(
                    "monitoring.log_writing", time.perf_counter_ns() - write_started_ns
                )
            now = time.monotonic()
            if now - last_log_flush >= 1.0:
                flush_started_ns = time.perf_counter_ns()
                log.flush()
                GLOBAL_PROFILER.record_ns(
                    "monitoring.log_flush", time.perf_counter_ns() - flush_started_ns
                )
                last_log_flush = now
            if on_output is not None:
                on_output()
        reader.join(timeout=1.0)
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
        if echo:
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
    event_log = (
        getattr(args, "event_log", None) or output_dir / "logs" / "rollout_events.jsonl"
    ).resolve()
    GLOBAL_PROFILER.enabled = getattr(args, "profile_runtime", True)
    events = EventWriter(event_log, source="orchestrator", profiler=GLOBAL_PROFILER)
    event_reader = EventReader(event_log, profiler=GLOBAL_PROFILER)
    dashboard_state = RolloutAggregator(
        tasks,
        episodes_per_task,
        stall_warning_seconds=getattr(args, "stall_warning_seconds", 30.0),
    )
    dashboard = RolloutDashboard(
        dashboard_state,
        enabled={"auto": None, "on": True, "off": False}.get(
            getattr(args, "dashboard", "auto"), None
        ),
        refresh_seconds=getattr(args, "dashboard_refresh", 1.0),
        screen=False,
    )
    dashboard.start()

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
        profiler=GLOBAL_PROFILER,
        log_level=getattr(args, "log_level", "normal"),
    ) as connected_policy:
        server_info = getattr(connected_policy, "model_info", None)
        policy_type = getattr(connected_policy, "policy_type", "unknown")

    start_event = events.emit(
        "rollout_started",
        event_id="rollout-started",
        flush=True,
        experiment_name=output_dir.name,
        checkpoint=args.checkpoint,
        checkpoint_identity=(server_info or {}).get("checkpoint_identity"),
        policy_type=policy_type,
        sidecar_type=(server_info or {}).get("sidecar_type"),
        action_horizon=(server_info or {}).get("action_horizon"),
        execution_horizon=args.execution_horizon,
        tasks=tasks,
        episodes_per_task=episodes_per_task,
        total_episodes=len(tasks) * episodes_per_task,
        phase="Preparing tasks",
    )
    dashboard.update([start_event], force=True)

    python_command = shlex.split(args.python_command)
    if not python_command:
        raise ValueError("--python-command cannot be empty.")
    worker = Path(__file__).with_name("run_closed_loop_worker.py").resolve()
    all_episodes: list[EpisodeResult] = []
    runtime_metadata: dict[str, Any] = {}

    for task_index, task in enumerate(tasks):
        task_started = time.monotonic()
        events.emit(
            "task_started",
            event_id=f"task-started:{task}",
            task=task,
            task_index=task_index,
            total_tasks=len(tasks),
            planned_episodes=episodes_per_task,
            phase="Task initialization",
        )
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
            "--event-log",
            str(event_log),
            "--rollout-diagnostics",
            getattr(args, "rollout_diagnostics", "standard"),
            "--diagnostic-sample-every",
            str(getattr(args, "diagnostic_sample_every", 10)),
            "--episode-telemetry-interval",
            str(getattr(args, "episode_telemetry_interval", 1.0)),
            "--log-level",
            getattr(args, "log_level", "normal"),
            "--profile-runtime"
            if getattr(args, "profile_runtime", True)
            else "--no-profile-runtime",
            "--gpu-sync-profile"
            if getattr(args, "gpu_sync_profile", False)
            else "--no-gpu-sync-profile",
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
            on_output=lambda: dashboard.update(event_reader.read()),
            echo=not dashboard.interactive,
        )
        dashboard.update(event_reader.read(), force=True)
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

        # Replay every atomic worker result into telemetry. Stable event IDs make
        # this a no-op for normal runs and recover the narrow crash window after
        # result persistence but before the worker's event append.
        for row in task_episodes:
            event_id = f"episode-completed:{row.task}:{row.episode_index}"
            if event_id in dashboard_state.seen_event_ids:
                continue
            events.emit(
                "episode_completed",
                event_id=event_id,
                task=row.task,
                episode_index=row.episode_index,
                seed=row.seed,
                success=row.success,
                status=classify_result(row.success, row.termination_reason),
                termination_reason=row.termination_reason,
                error_category=row.termination_reason if row.success is None else None,
                num_steps=row.num_steps,
                sim_steps=row.sim_steps,
                elapsed_seconds=row.elapsed_seconds,
                execution_horizon=row.execution_horizon,
                recovered_from_worker_result=True,
            )
        dashboard.update(event_reader.read(), force=True)

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
        missing_results = [
            _worker_failure_result(
                args,
                task=task,
                episode_index=episode_index,
                error=failure_detail,
            )
            for episode_index in missing_indices
        ]
        all_episodes.extend(missing_results)
        for row in missing_results:
            events.emit(
                "episode_completed",
                event_id=f"episode-completed:{row.task}:{row.episode_index}",
                task=row.task,
                episode_index=row.episode_index,
                seed=row.seed,
                success=None,
                status="error",
                termination_reason="worker_failure",
                error_category="worker_failure",
                num_steps=0,
                sim_steps=0,
                elapsed_seconds=0.0,
                execution_horizon=row.execution_horizon,
                executed=False,
            )
        dashboard.update(event_reader.read(), force=True)

        run_incomplete = bool(issues)
        running_aggregate = write_results(
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
            failed_event = events.emit(
                "task_failed",
                event_id=f"task-failed:{task}",
                flush=True,
                task=task,
                phase=phase,
                error_category="worker_failure",
                message=failure_detail,
                unexecuted_episodes=len(missing_indices),
                worker_return_code=outcome.return_code,
                worker_log=str(output_dir / "logs" / f"{task}.log"),
            )
            dashboard.update([failed_event], force=True)
            dashboard.close()
            raise RuntimeError(
                f"UniVTAC worker for {task} did not complete a valid rollout: "
                f"{failure_detail}. See {output_dir / 'logs' / f'{task}.log'} and "
                f"{output_dir / 'summary.json'}"
            )
        task_summary = next(row for row in running_aggregate["tasks"] if row["task"] == task)
        task_event = events.emit(
            "task_completed",
            event_id=f"task-completed:{task}",
            flush=True,
            task=task,
            completed_episodes=task_summary["total_rollouts"],
            successes=task_summary["successes"],
            failures=task_summary["failures"],
            errors=task_summary["errors"],
            success_rate=task_summary["success_rate"],
            elapsed_seconds=time.monotonic() - task_started,
            phase="Task completed",
        )
        dashboard.update([task_event], force=True)

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
    completed_event = events.emit(
        "rollout_completed",
        event_id="rollout-completed",
        flush=True,
        completed_episodes=aggregate["total_rollouts"],
        successes=aggregate["total_successes"],
        failures=aggregate["total_failures"],
        errors=aggregate["total_errors"],
        macro_success_rate=aggregate["macro_average"],
        micro_success_rate=aggregate["micro_average"],
        phase="Completed",
    )
    dashboard.update([completed_event], force=True)
    dashboard.close()
    GLOBAL_PROFILER.write_json(
        output_dir / "logs" / "latency_metrics.json",
        metadata={
            "worker_profiles": {
                task: values.get("performance_profile", {})
                for task, values in runtime_metadata.items()
            },
            "gpu_synchronized": bool(getattr(args, "gpu_sync_profile", False)),
        },
    )
    events.close()
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
