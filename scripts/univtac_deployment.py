#!/usr/bin/env python3
"""Process orchestration for split and local UniVTAC rollouts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "UniVTAC"))

from univtac_rollout.checkpoint import checkpoint_identity  # noqa: E402
from univtac_rollout.client import MinimalPolicyClient  # noqa: E402
from univtac_rollout.dashboard import RolloutDashboard  # noqa: E402
from univtac_rollout.diagnostics import dependency_preflight  # noqa: E402
from univtac_rollout.monitoring import AsyncSampler  # noqa: E402
from univtac_rollout.multimodal import load_sensor_contract, validate_contract  # noqa: E402
from univtac_rollout.profiling import GLOBAL_PROFILER  # noqa: E402
from univtac_rollout.telemetry import LOG_LEVELS, EventReader, EventWriter, RolloutAggregator  # noqa: E402


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument("--deployment", choices=("split", "local"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-identity", help="SHA-256 identity for remote-only split checkpoints"
    )
    parser.add_argument("--policy", choices=("auto", "rgb", "wah"), default="auto")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--model-python", type=Path, default=ROOT / ".venv" / "bin" / "python")
    parser.add_argument("--model-device", default="cuda:0")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--model-visible-devices")
    parser.add_argument("--sim-visible-devices")
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--retry-interval", type=float, default=2)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sensor-contract", type=Path)
    parser.add_argument("--scene-calibration", type=Path)
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--dashboard", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--dashboard-refresh", type=float, default=1.0)
    parser.add_argument("--episode-telemetry-interval", type=float, default=1.0)
    parser.add_argument("--gpu-telemetry-interval", type=float, default=5.0)
    parser.add_argument("--gpu-sampling-timeout", type=float, default=2.0)
    parser.add_argument("--stall-warning-seconds", type=float, default=30.0)
    parser.add_argument("--log-level", choices=LOG_LEVELS, default="normal")
    parser.add_argument("--profile-runtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-sync-profile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--rollout-diagnostics", choices=("off", "standard", "full"), default="standard"
    )
    parser.add_argument("--diagnostic-sample-every", type=int, default=10)
    parser.add_argument("--trainctl-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--trainctl-message-mode",
        choices=("standard", "koyori", "roboco", "neuro", "sopia", "sophia"),
        default="standard",
    )
    parser.add_argument("--rollout-message-style", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--notify-task-complete", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--notify-halfway", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--notify-errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--validate-ffmpeg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=argparse.SUPPRESS,
    )
    return parser.parse_known_args()


def check_port(host: str, port: int) -> None:
    with socket.socket() as sock:
        if sock.connect_ex((host, port)) == 0:
            raise RuntimeError(
                f"{host}:{port} is occupied. Local mode will not stop or reuse an existing service."
            )


def validate_python(
    python: Path, packages: tuple[str, ...], role: str, launcher: str | None = None
) -> None:
    if not python.is_file():
        raise FileNotFoundError(f"{role} Python is missing: {python}")
    script = (
        "import importlib.util,sys; "
        "missing=[p for p in sys.argv[1:] if importlib.util.find_spec(p) is None]; "
        "sys.exit('missing packages: '+','.join(missing) if missing else 0)"
    )
    command = shlex.split(launcher) if launcher else [str(python)]
    result = subprocess.run(
        [*command, "-c", script, *packages], cwd=ROOT, capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(f"{role} Python {python} failed validation: {result.stderr.strip()}")


def validate_simulator_ffmpeg(launcher: str) -> None:
    command = shlex.split(launcher)
    script = (
        "import shutil,sys; executable=shutil.which('ffmpeg'); "
        "sys.exit(0 if executable else 'FFmpeg executable unavailable in simulator environment')"
    )
    result = subprocess.run([*command, "-c", script], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            result.stderr.strip()
            or result.stdout.strip()
            or "FFmpeg executable unavailable in simulator environment"
        )


def probe(
    host: str,
    port: int,
    checkpoint: str,
    policy: str,
    horizon: int,
    expected_identity: str | None = None,
) -> dict:
    with MinimalPolicyClient(host, port, timeout_ms=2000) as client:
        if not client.ping():
            raise ConnectionError(f"Policy server at {host}:{port} did not return readiness.")
        info = client.get_model_info()
        configs = client.get_modality_config()
    expected = expected_identity or checkpoint_identity(checkpoint)
    if not expected.startswith("sha256:") or len(expected) != 71:
        raise ValueError("--checkpoint-identity must be sha256:<64 hex digits>.")
    if info["checkpoint_identity"] != expected:
        raise ValueError(
            f"Checkpoint identity mismatch: requested {expected}, server loaded "
            f"{info['checkpoint_identity']} ({info.get('checkpoint_path')})."
        )
    info["policy_type"] = validate_contract(configs, horizon, policy)
    return info


def wait_ready(args: argparse.Namespace, server: subprocess.Popen | None) -> dict:
    started = time.monotonic()
    deadline = started + args.startup_timeout
    last_report = started
    print(
        f"Waiting for GR00T server readiness (timeout {args.startup_timeout:g}s)...",
        flush=True,
    )
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if server is not None and server.poll() is not None:
            raise RuntimeError(
                f"Inference server exited before readiness (code {server.returncode})."
            )
        try:
            return probe(
                args.server_host,
                args.server_port,
                args.checkpoint,
                args.policy,
                args.execution_horizon,
                getattr(args, "checkpoint_identity", None),
            )
        except (ConnectionError, TimeoutError, OSError) as exc:
            last_error = exc
            now = time.monotonic()
            if now - last_report >= 10:
                print(
                    f"Still loading GR00T server... {int(now - started)}s elapsed "
                    f"(last probe: {type(exc).__name__})",
                    flush=True,
                )
                last_report = now
            time.sleep(min(args.retry_interval, max(0, deadline - time.monotonic())))
    raise TimeoutError(
        f"Policy server readiness timed out after {args.startup_timeout}s: {last_error}"
    )


def gpu_memory(timeout_seconds: float = 2.0) -> list[dict]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            try:
                rows.append(
                    dict(
                        index=int(parts[0]),
                        model=parts[1],
                        total_mib=int(parts[2]),
                        free_mib=int(parts[3]),
                        used_mib=int(parts[4]),
                    )
                )
            except ValueError:
                continue
    return rows


def stop_owned(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def failure_category(text: str) -> str:
    lowered = text.lower()
    patterns = (
        ("no space left on device", "no_space_left"),
        ("ffmpeg", "ffmpeg_unavailable_or_failed"),
        ("out of memory", "cuda_oom"),
        ("nan", "nonfinite_action"),
        ("checkpoint identity mismatch", "checkpoint_identity_mismatch"),
        ("readiness timed out", "server_startup_timeout"),
        ("sensor contract", "sensor_contract_mismatch"),
        ("invalid_action", "invalid_action"),
    )
    return next(
        (category for pattern, category in patterns if pattern in lowered), "process_failure"
    )


def last_relevant_error(path: Path) -> str | None:
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    relevant = [
        line.strip()
        for line in lines
        if any(
            token in line.lower()
            for token in ("error", "exception", "traceback", "failed", "out of memory", "no space")
        )
    ]
    if not relevant:
        return None
    # Keep credentials and URLs out of structured telemetry/notifications.
    value = relevant[-1]
    for secret in ("webhook", "token", "authorization"):
        if secret in value.lower():
            return "[redacted credential-bearing error; inspect local log]"
    return value[:500]


def print_rollout_summary(output: Path, logs: Path) -> dict | None:
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        print(f"Rollout summary unavailable; logs: {logs}", file=sys.stderr, flush=True)
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        counts = (
            int(summary["total_successes"]),
            int(summary["total_failures"]),
            int(summary["total_errors"]),
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(
            f"Could not read rollout summary {summary_path}: {exc}; logs: {logs}", file=sys.stderr
        )
        return None
    print(
        f"Rollout result: success={counts[0]} failure={counts[1]} error={counts[2]}",
        flush=True,
    )
    print(f"Summary: {summary_path}", flush=True)
    print(f"Logs: {logs}", flush=True)
    return summary


def build_rollout_command(
    args: argparse.Namespace, output: Path, passthrough: list[str]
) -> list[str]:
    """Both deployment topologies execute the same evaluation command."""
    cmd = [
        "bash",
        str(ROOT / "scripts" / "run_gr00t_univtac_rollout.sh"),
        "--checkpoint",
        args.checkpoint,
        "--policy",
        args.policy,
        "--server-host",
        args.server_host,
        "--server-port",
        str(args.server_port),
        "--execution-horizon",
        str(args.execution_horizon),
        "--device",
        args.sim_device,
        "--output-dir",
        str(output),
        "--event-log",
        str(output / "logs" / "rollout_events.jsonl"),
        "--rollout-diagnostics",
        getattr(args, "rollout_diagnostics", "standard"),
        "--diagnostic-sample-every",
        str(getattr(args, "diagnostic_sample_every", 10)),
        "--episode-telemetry-interval",
        str(getattr(args, "episode_telemetry_interval", 1.0)),
        "--stall-warning-seconds",
        str(getattr(args, "stall_warning_seconds", 30.0)),
        "--log-level",
        getattr(args, "log_level", "normal"),
        "--profile-runtime" if getattr(args, "profile_runtime", True) else "--no-profile-runtime",
        "--gpu-sync-profile"
        if getattr(args, "gpu_sync_profile", False)
        else "--no-gpu-sync-profile",
        # The deployment controller owns the user's terminal and renders events.
        "--dashboard",
        "off",
    ]
    if getattr(args, "checkpoint_identity", None) is not None:
        cmd += ["--checkpoint-identity", args.checkpoint_identity]
    if args.sensor_contract is not None:
        cmd += ["--sensor-contract", str(args.sensor_contract.resolve())]
    if args.scene_calibration is not None:
        cmd += ["--scene-calibration", str(args.scene_calibration.resolve())]
    return cmd + passthrough


def _planned_episodes(passthrough: list[str]) -> int:
    modes = {"smoke": 5, "quick": 20, "full": 100}
    episodes = None
    mode = "smoke"
    tasks = 8
    for index, value in enumerate(passthrough):
        if value == "--episodes-per-task" and index + 1 < len(passthrough):
            episodes = int(passthrough[index + 1])
        elif value == "--mode" and index + 1 < len(passthrough):
            mode = passthrough[index + 1]
        elif value == "--tasks":
            selected = []
            for item in passthrough[index + 1 :]:
                if item.startswith("--"):
                    break
                selected.append(item)
            tasks = 8 if selected == ["all"] or not selected else len(selected)
    return max(1, (episodes or modes.get(mode, 5)) * tasks)


def trainctl_rollout_command(
    args: argparse.Namespace, output: Path, command: list[str], passthrough: list[str]
) -> list[str]:
    control_python = args.model_python
    check = subprocess.run(
        [
            str(control_python),
            "-c",
            "from trainctl import tracker; "
            "assert all(hasattr(tracker,n) for n in "
            "('init','define_metrics','log','notify','finish','fail'))",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "trainctl was explicitly requested but is unavailable in the control Python. "
            f"Install WAH's locked dependencies into {control_python} with 'uv sync --locked'."
        )
    event_log = output / "logs" / "rollout_events.jsonl"
    name = f"rollout-{Path(args.checkpoint).name}-{output.name}"
    bridge = [
        str(control_python),
        str(ROOT / "scripts" / "univtac_rollout_trainctl_bridge.py"),
        "--event-log",
        str(event_log),
        "--name",
        name,
        "--message-mode",
        args.trainctl_message_mode,
        "--message-style",
        str(args.rollout_message_style),
        "--notify-task-complete" if args.notify_task_complete else "--no-notify-task-complete",
        "--notify-halfway" if args.notify_halfway else "--no-notify-halfway",
        "--notify-errors" if args.notify_errors else "--no-notify-errors",
        "--",
        *command,
    ]
    return [
        str(control_python),
        "-m",
        "gr00t.experiment.trainctl_cli",
        "run",
        "--name",
        name,
        "--checkpoint-dir",
        str(output),
        "--total-steps",
        str(_planned_episodes(passthrough)),
        # The SDK bridge supplies rollout-specific persona text. Keep the supervisor
        # itself factual so training dialogue is never attached to a rollout.
        "--message-mode",
        "standard",
        "--",
        *bridge,
    ]


def run(args: argparse.Namespace, passthrough: list[str]) -> int:
    defaults = {
        "dashboard": "auto",
        "dashboard_refresh": 1.0,
        "episode_telemetry_interval": 1.0,
        "gpu_telemetry_interval": 5.0,
        "gpu_sampling_timeout": 2.0,
        "stall_warning_seconds": 30.0,
        "log_level": "normal",
        "profile_runtime": True,
        "gpu_sync_profile": False,
        "rollout_diagnostics": "standard",
        "diagnostic_sample_every": 10,
        "trainctl_enabled": False,
        "trainctl_message_mode": "standard",
        "rollout_message_style": 1,
    }
    for name, default in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    if args.startup_timeout <= 0 or args.retry_interval <= 0:
        raise ValueError("Startup timeout and retry interval must be positive.")
    for option in (
        "dashboard_refresh",
        "episode_telemetry_interval",
        "gpu_telemetry_interval",
        "gpu_sampling_timeout",
        "stall_warning_seconds",
    ):
        value = getattr(args, option)
        if value <= 0 or not math.isfinite(value):
            raise ValueError(f"--{option.replace('_', '-')} must be finite and positive.")
    if args.diagnostic_sample_every < 1:
        raise ValueError("--diagnostic-sample-every must be positive.")
    args.checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    args.model_python = args.model_python.expanduser().absolute()
    if args.deployment == "local" or getattr(args, "checkpoint_identity", None) is None:
        local_identity = checkpoint_identity(args.checkpoint)
        if (
            getattr(args, "checkpoint_identity", None) is not None
            and args.checkpoint_identity != local_identity
        ):
            raise ValueError("--checkpoint-identity disagrees with the local checkpoint content.")
        args.checkpoint_identity = local_identity
    elif not args.checkpoint_identity.startswith("sha256:") or len(args.checkpoint_identity) != 71:
        raise ValueError("--checkpoint-identity must be sha256:<64 hex digits>.")
    sim_python = os.environ.get("UNIVTAC_EXPECTED_PYTHON")
    if not sim_python or not Path(sim_python).is_file():
        raise FileNotFoundError(
            "UNIVTAC_EXPECTED_PYTHON must point to the Isaac-compatible Python executable."
        )
    validate_python(
        Path(sim_python),
        ("isaaclab", "numpy", "zmq", "msgpack"),
        "Isaac",
        os.environ.get("UNIVTAC_PYTHON_COMMAND"),
    )
    if getattr(args, "validate_ffmpeg", False):
        validate_simulator_ffmpeg(os.environ.get("UNIVTAC_PYTHON_COMMAND", str(Path(sim_python))))
    if not os.environ.get("UNIVTAC_ROOT"):
        raise ValueError("UNIVTAC_ROOT must point to the UniVTAC checkout.")
    univtac_root = Path(os.environ["UNIVTAC_ROOT"]).expanduser().resolve()
    if not (univtac_root / "envs" / "_base_task.py").is_file():
        raise FileNotFoundError(f"UNIVTAC_ROOT is not a UniVTAC checkout: {univtac_root}")
    if args.deployment == "local":
        if not args.model_python.is_file():
            raise FileNotFoundError(f"Model Python is missing: {args.model_python}")
        validate_python(args.model_python, ("torch", "transformers", "gr00t"), "GR00T")
        if args.model_python.resolve() == Path(sim_python).resolve():
            raise ValueError("Local mode needs separate model and Isaac Python environments.")
        check_port(args.server_host, args.server_port)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_dir or ROOT / "outputs" / "rollouts" / Path(args.checkpoint).name / stamp
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    preflight = dependency_preflight(
        output_dir=output,
        checkpoint=args.checkpoint,
        server_host=args.server_host,
        server_port=args.server_port,
        # Local mode already ran check_port above; do not probe the same socket
        # twice (and keep this diagnostics helper independently testable).
        require_server_port=False,
        sensor_contract=args.sensor_contract,
        require_ffmpeg=False,
    )
    event_path = logs / "rollout_events.jsonl"
    GLOBAL_PROFILER.enabled = args.profile_runtime
    event_writer = EventWriter(event_path, source="deployment", profiler=GLOBAL_PROFILER)
    event_reader = EventReader(event_path, profiler=GLOBAL_PROFILER)
    rollout_state = RolloutAggregator(stall_warning_seconds=args.stall_warning_seconds)
    dashboard = RolloutDashboard(
        rollout_state,
        enabled={"auto": None, "on": True, "off": False}[args.dashboard],
        refresh_seconds=args.dashboard_refresh,
    )
    dashboard.start()
    preflight_event = event_writer.emit(
        "system_diagnostics",
        event_id="system-diagnostics:preflight",
        diagnostics={
            **preflight,
            "server_status": "not started" if args.deployment == "local" else "external",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        },
    )
    dashboard.update([preflight_event], force=True)
    memory: dict[str, object] = {
        "before_launch": gpu_memory(),
        "after_model_load": [],
        "after_sim_init": [],
        "peak_used_mib": {},
    }
    for row in memory["before_launch"]:
        print(
            f"GPU {row['index']} {row['model']}: {row['free_mib']}/{row['total_mib']} MiB free before launch",
            flush=True,
        )
    model_visibility = (
        args.model_visible_devices
        if args.model_visible_devices is not None
        else os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    sim_visibility = (
        args.sim_visible_devices
        if args.sim_visible_devices is not None
        else os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if args.deployment == "local" and model_visibility != sim_visibility:
        print(
            "Model and simulator CUDA visibility masks differ; cuda:0 may address different physical GPUs.",
            flush=True,
        )
    server: subprocess.Popen | None = None
    rollout: subprocess.Popen | None = None
    interrupted = False
    start = time.monotonic()

    def on_signal(signum: int, frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        stop_owned(rollout)
        stop_owned(server)

    previous_int = signal.signal(signal.SIGINT, on_signal)
    previous_term = signal.signal(signal.SIGTERM, on_signal)
    server_log = None
    rollout_log = None
    monitor_stop = threading.Event()
    monitor_thread: threading.Thread | None = None
    gpu_sampler: AsyncSampler | None = None
    try:
        server_starting = event_writer.emit(
            "server_starting",
            event_id="server-starting",
            server_status="starting" if args.deployment == "local" else "probing external",
            server_log=str(logs / "server.log") if args.deployment == "local" else None,
            phase="Server starting",
        )
        dashboard.update([server_starting], force=True)
        if args.deployment == "local":
            server_log = (logs / "server.log").open("w", encoding="utf-8")
            model_env = os.environ.copy()
            if args.model_visible_devices is not None:
                model_env["CUDA_VISIBLE_DEVICES"] = args.model_visible_devices
            server_cmd = [
                str(args.model_python),
                str(ROOT / "gr00t" / "eval" / "run_gr00t_server.py"),
                "--model-path",
                args.checkpoint,
                "--embodiment-tag",
                "NEW_EMBODIMENT",
                "--device",
                args.model_device,
                "--host",
                args.server_host,
                "--port",
                str(args.server_port),
            ]
            server = subprocess.Popen(
                server_cmd,
                cwd=ROOT,
                env=model_env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            event_writer.emit(
                "system_diagnostics",
                event_id="system-diagnostics:server-process",
                diagnostics={
                    "server_pid": getattr(server, "pid", None),
                    "server_status": "starting",
                },
            )
        try:
            info = wait_ready(args, server)
        except Exception:
            if server_log is not None:
                server_log.flush()
                print((logs / "server.log").read_text(encoding="utf-8"), file=sys.stderr)
            raise
        memory["after_model_load"] = gpu_memory()
        if info["policy_type"] == "wah":
            load_sensor_contract(args.sensor_contract)
        if args.deployment == "local" and memory["before_launch"] and memory["after_model_load"]:
            before = {row["index"]: row for row in memory["before_launch"]}
            for row in memory["after_model_load"]:
                old = before.get(row["index"])
                if old is not None:
                    model_cost = max(0, old["free_mib"] - row["free_mib"])
                    if model_cost and row["free_mib"] < model_cost:
                        print(
                            f"Warning: GPU {row['index']} has {row['free_mib']} MiB free after model load, "
                            f"less than the model's observed {model_cost} MiB allocation. "
                            "Isaac Sim may run out of memory; this is a diagnostic, not a compatibility limit.",
                            flush=True,
                        )
        print(
            f"Server ready: {info['checkpoint_identity']} {info['policy_type']} {info.get('sidecar_type')}",
            flush=True,
        )
        ready_event = event_writer.emit(
            "server_ready",
            event_id="server-ready",
            flush=True,
            server_status="ready",
            server_pid=getattr(server, "pid", None),
            server_initialization_seconds=time.monotonic() - start,
            checkpoint_identity=info.get("checkpoint_identity"),
            policy_type=info.get("policy_type"),
            sidecar_type=info.get("sidecar_type"),
            phase="Server ready",
        )
        dashboard.update([ready_event], force=True)
        cmd = build_rollout_command(args, output, passthrough)
        if args.trainctl_enabled:
            cmd = trainctl_rollout_command(args, output, cmd, passthrough)
        sim_env = os.environ.copy()
        if args.sim_visible_devices is not None:
            sim_env["CUDA_VISIBLE_DEVICES"] = args.sim_visible_devices
        rollout_log_path = logs / "rollout.log"
        rollout_log = rollout_log_path.open("w", encoding="utf-8")
        rollout = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=sim_env,
            stdout=rollout_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        log_offsets: dict[Path, int] = {}
        log_tails: dict[Path, str] = {}

        def sample_gpu() -> list[dict] | None:
            rows = gpu_memory(args.gpu_sampling_timeout)
            return rows or None

        gpu_sampler = AsyncSampler(
            sample_gpu,
            interval_seconds=args.gpu_telemetry_interval,
            stage="monitoring.gpu_telemetry_collection",
            profiler=GLOBAL_PROFILER,
        )
        gpu_sampler.start()

        def monitor() -> None:
            while not monitor_stop.wait(1.0):
                rows, gpu_error, gpu_sample_time = gpu_sampler.snapshot()
                rows = rows or []
                for row in rows:
                    key = str(row["index"])
                    memory["peak_used_mib"][key] = max(
                        memory["peak_used_mib"].get(key, 0), row["used_mib"]
                    )
                if not memory["after_sim_init"]:
                    for task_log in logs.glob("*.log"):
                        if task_log.name in {"server.log", "rollout.log"}:
                            continue
                        with task_log.open("rb") as stream:
                            stream.seek(log_offsets.get(task_log, 0))
                            tail = log_tails.get(task_log, "") + stream.read().decode(
                                "utf-8", errors="replace"
                            )
                            log_offsets[task_log] = stream.tell()
                            log_tails[task_log] = tail[-100:]
                        if "phase=task_construction: complete" in tail:
                            memory["after_sim_init"] = list(rows)
                            break
                selected = rows[0] if rows else {}
                event_writer.emit(
                    "system_diagnostics",
                    diagnostics={
                        "server_status": "ready"
                        if server is None or server.poll() is None
                        else "exited",
                        "server_pid": getattr(server, "pid", None),
                        "server_exit_code": server.poll() if server is not None else None,
                        "simulator_pid": getattr(rollout, "pid", None),
                        "simulator_status": "running"
                        if rollout is not None and rollout.poll() is None
                        else "exited",
                        "gpu_memory_used_mib": selected.get("used_mib"),
                        "gpu_memory_free_mib": selected.get("free_mib"),
                        "gpu_memory_total_mib": selected.get("total_mib"),
                        "gpu_memory_peak_mib": max(memory["peak_used_mib"].values(), default=0),
                        "gpu_index": selected.get("index"),
                        "gpu_model": selected.get("model"),
                        "gpu_telemetry_error": gpu_error,
                        "gpu_sample_monotonic": gpu_sample_time,
                        "cuda_visible_devices": sim_visibility or "all",
                    },
                )

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        while rollout.poll() is None:
            dashboard.update(event_reader.read())
            if server is not None and server.poll() is not None:
                print(
                    f"Inference server crashed during rollout (code {server.returncode}); "
                    f"see {logs / 'server.log'}.",
                    file=sys.stderr,
                )
                stop_owned(rollout)
                break
            time.sleep(min(args.dashboard_refresh, 0.2))
        code = rollout.wait()
        dashboard.update(event_reader.read(), force=True)
        if server is not None and server.returncode is not None and code == 0:
            code = 1
        monitor_stop.set()
        monitor_thread.join(timeout=3)
        gpu_sampler.close(timeout=args.gpu_sampling_timeout + 0.5)
        rollout_log.flush()
        summary = print_rollout_summary(output, logs)
        if summary is not None and summary["total_errors"] > 0 and code == 0:
            code = 1
        completed_with_errors = bool(
            summary is not None
            and summary["total_errors"] > 0
            and rollout_state.status == "Completed"
        )
        if code:
            if completed_with_errors:
                warning = event_writer.emit(
                    "system_warning",
                    event_id="system-warning:completed-with-errors",
                    flush=True,
                    message=(
                        "All scheduled episodes completed, but evaluation/infrastructure errors "
                        "require a nonzero exit. See summary.json and worker logs."
                    ),
                    error_category="completed_with_errors",
                )
                dashboard.update([warning], force=True)
                dashboard.close()
                print(
                    f"Rollout completed with {summary['total_errors']} error episode(s); "
                    f"returning code {code}. Logs preserved in {logs}",
                    file=sys.stderr,
                )
                return 130 if interrupted else code
            server_error = last_relevant_error(logs / "server.log")
            rollout_error = last_relevant_error(logs / "rollout.log")
            failure_event = event_writer.emit(
                "rollout_failed",
                event_id="rollout-failed",
                flush=True,
                error_category="inference_server_crash"
                if server is not None and server.returncode is not None
                else failure_category(rollout_error or "process failure"),
                message=f"Rollout process exited with code {code}",
                exit_code=code,
                server_exit_code=server.returncode if server is not None else None,
                server_log=str(logs / "server.log") if server_log is not None else None,
                rollout_log=str(logs / "rollout.log"),
                last_relevant_error=server_error or rollout_error,
                phase=rollout_state.phase,
            )
            dashboard.update([failure_event], force=True)
            dashboard.close()
            print((logs / "rollout.log").read_text(encoding="utf-8")[-5000:], file=sys.stderr)
            if "out of memory" in (logs / "rollout.log").read_text(encoding="utf-8").lower():
                print("CUDA OOM in rollout/Isaac process.", file=sys.stderr)
            if (
                server_log is not None
                and "out of memory" in (logs / "server.log").read_text(encoding="utf-8").lower()
            ):
                print("CUDA OOM in GR00T server process.", file=sys.stderr)
            print(
                f"Rollout process failed with code {code}; logs preserved in {logs}",
                file=sys.stderr,
            )
        return 130 if interrupted else code
    except BaseException as exc:
        if rollout_state.status not in {"Failed", "Completed"}:
            category = (
                "no_space_left"
                if isinstance(exc, OSError) and exc.errno == 28
                else failure_category(str(exc))
            )
            failure_event = event_writer.emit(
                "rollout_failed",
                event_id="rollout-failed",
                flush=True,
                error_category=category,
                message=str(exc)[:1000],
                server_exit_code=server.poll() if server is not None else None,
                server_log=str(logs / "server.log") if server_log is not None else None,
                rollout_log=str(logs / "rollout.log") if rollout_log is not None else None,
                phase=rollout_state.phase,
            )
            dashboard.update([failure_event], force=True)
            dashboard.close()
        raise
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=3)
        if gpu_sampler is not None:
            gpu_sampler.close(timeout=args.gpu_sampling_timeout + 0.5)
        stop_owned(rollout)
        stop_owned(server)
        if server_log is not None:
            server_log.close()
        if rollout_log is not None:
            rollout_log.close()
        dashboard.update(event_reader.read(), force=True)
        dashboard.close()
        metadata = {
            "deployment_mode": args.deployment,
            "checkpoint_identity": locals().get("info", {}).get("checkpoint_identity"),
            "model_info": locals().get("info"),
            "server_address": f"{args.server_host}:{args.server_port}",
            "model_device": args.model_device if args.deployment == "local" else None,
            "sim_device": args.sim_device,
            "model_visible_devices": model_visibility if args.deployment == "local" else None,
            "sim_visible_devices": sim_visibility,
            "sim_python": sim_python,
            "sim_python_command": os.environ.get("UNIVTAC_PYTHON_COMMAND", sim_python),
            "model_python": str(args.model_python) if args.deployment == "local" else None,
            "python_version": sys.version,
            "git_commit": git_commit(),
            "duration_seconds": time.monotonic() - start,
            "gpu_memory": memory,
            "rollout_events": str(event_path),
            "rollout_diagnostics": args.rollout_diagnostics,
            "dashboard": args.dashboard,
            "trainctl_enabled": args.trainctl_enabled,
            "trainctl_message_mode": args.trainctl_message_mode if args.trainctl_enabled else None,
            "rollout_message_style": args.rollout_message_style if args.trainctl_enabled else None,
            "utc_finished": datetime.now(timezone.utc).isoformat(),
        }
        summary_path = output / "summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                metadata["simulator_runtime_by_task"] = summary.get("metadata", {}).get(
                    "runtime_by_task", {}
                )
            except (OSError, ValueError):
                pass
        (output / "runtime_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if args.profile_runtime:
            GLOBAL_PROFILER.write_json(
                logs / "monitor_latency_metrics.json",
                metadata={
                    "gpu_telemetry_interval_seconds": args.gpu_telemetry_interval,
                    "dashboard_refresh_seconds": args.dashboard_refresh,
                },
            )
        event_writer.close()
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    args, passthrough = parse_args()
    try:
        raise SystemExit(run(args, passthrough))
    except Exception as exc:
        print(f"UniVTAC {args.deployment} deployment failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
