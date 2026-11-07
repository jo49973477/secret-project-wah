#!/usr/bin/env python3
"""Process orchestration for split and local UniVTAC rollouts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
from univtac_rollout.multimodal import load_sensor_contract, validate_contract  # noqa: E402
from univtac_progress import RolloutProgress  # noqa: E402


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


def gpu_memory() -> list[dict]:
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
            timeout=5,
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
    ]
    if getattr(args, "checkpoint_identity", None) is not None:
        cmd += ["--checkpoint-identity", args.checkpoint_identity]
    if args.sensor_contract is not None:
        cmd += ["--sensor-contract", str(args.sensor_contract.resolve())]
    if args.scene_calibration is not None:
        cmd += ["--scene-calibration", str(args.scene_calibration.resolve())]
    return cmd + passthrough


def run(args: argparse.Namespace, passthrough: list[str]) -> int:
    if args.startup_timeout <= 0 or args.retry_interval <= 0:
        raise ValueError("Startup timeout and retry interval must be positive.")
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
    try:
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
        print(f"Starting UniVTAC rollout (live progress; detailed logs: {logs})", flush=True)
        cmd = build_rollout_command(args, output, passthrough)
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
        progress = RolloutProgress()

        def monitor() -> None:
            rollout_offset = 0
            pending_line = ""
            last_gpu_sample = 0.0

            def drain_rollout_log() -> None:
                nonlocal rollout_offset, pending_line
                with rollout_log_path.open(encoding="utf-8", errors="replace") as stream:
                    stream.seek(rollout_offset)
                    pending_line += stream.read()
                    rollout_offset = stream.tell()
                while "\n" in pending_line:
                    line, pending_line = pending_line.split("\n", 1)
                    progress.consume(line)

            while True:
                drain_rollout_log()
                progress.render()

                now = time.monotonic()
                if now - last_gpu_sample >= 2:
                    last_gpu_sample = now
                    for row in gpu_memory():
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
                            memory["after_sim_init"] = gpu_memory()
                            break
                if monitor_stop.wait(0.5):
                    # The child may have flushed its final event after our
                    # previous read; drain again before printing the summary.
                    drain_rollout_log()
                    if pending_line:
                        progress.consume(pending_line)
                    return

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        while rollout.poll() is None:
            if server is not None and server.poll() is not None:
                print(
                    f"Inference server crashed during rollout (code {server.returncode}); "
                    f"see {logs / 'server.log'}.",
                    file=sys.stderr,
                )
                stop_owned(rollout)
                break
            time.sleep(1)
        code = rollout.wait()
        if server is not None and server.returncode is not None and code == 0:
            code = 1
        monitor_stop.set()
        monitor_thread.join(timeout=3)
        rollout_log.flush()
        progress.finish()
        summary = print_rollout_summary(output, logs)
        if summary is not None and summary["total_errors"] > 0 and code == 0:
            code = 1
        if code:
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
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=3)
        if "progress" in locals():
            progress.finish()
        stop_owned(rollout)
        stop_owned(server)
        if server_log is not None:
            server_log.close()
        if rollout_log is not None:
            rollout_log.close()
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
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    args, passthrough = parse_args()
    try:
        raise SystemExit(run(args, passthrough))
    except Exception as exc:
        print(f"UniVTAC {args.deployment} deployment failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
