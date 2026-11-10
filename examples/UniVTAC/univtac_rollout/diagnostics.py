"""Low-overhead control-side rollout preflight and system diagnostics."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from typing import Any, Iterable


MIN_TEMP_FREE_GIB = 1.0


def filesystem_diagnostics(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    existing = path
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    stats = os.statvfs(existing)
    block_size = stats.f_frsize or stats.f_bsize
    return {
        "path": str(path),
        "filesystem_path": str(existing),
        "device": os.stat(existing).st_dev,
        "free_bytes": stats.f_bavail * block_size,
        "free_gib": stats.f_bavail * block_size / 2**30,
        "total_gib": stats.f_blocks * block_size / 2**30,
        "free_inodes": stats.f_favail if stats.f_files else None,
        "total_inodes": stats.f_files or None,
    }


def collect_filesystems(output_dir: str | Path) -> dict[str, dict[str, Any]]:
    paths = {
        "output": Path(output_dir),
        "temp": Path(tempfile.gettempdir()),
        "root": Path("/"),
    }
    if Path("/dev/shm").is_dir():
        paths["dev_shm"] = Path("/dev/shm")
    return {name: filesystem_diagnostics(path) for name, path in paths.items()}


def check_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise RuntimeError(
                f"Inference server port {host}:{port} is unavailable: {exc}"
            ) from exc


def validate_checkpoint(path: str | Path, *, require_local: bool = True) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if require_local and not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if path.exists() and path.is_dir():
        candidates = tuple(path.glob("*.safetensors")) + tuple(path.glob("*.bin"))
        if not candidates:
            raise ValueError(
                f"Checkpoint directory has no .safetensors or .bin weight files: {path}"
            )
        empty = [str(candidate) for candidate in candidates if candidate.stat().st_size <= 0]
        if empty:
            raise ValueError(f"Checkpoint contains empty weight files: {empty}")
        return {
            "path": str(path),
            "weight_files": len(candidates),
            "weight_bytes": sum(candidate.stat().st_size for candidate in candidates),
        }
    if path.exists() and path.stat().st_size <= 0:
        raise ValueError(f"Checkpoint file is empty: {path}")
    return {
        "path": str(path),
        "weight_files": int(path.exists()),
        "weight_bytes": path.stat().st_size if path.exists() else None,
    }


def dependency_preflight(
    *,
    output_dir: str | Path,
    checkpoint: str | Path,
    server_host: str,
    server_port: int,
    require_server_port: bool,
    sensor_contract: str | Path | None,
    python_packages: Iterable[str] = ("numpy", "zmq", "msgpack"),
    require_ffmpeg: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    probe = output / ".rollout-write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(f"Rollout output directory is not writable: {output}: {exc}") from exc

    missing = [package for package in python_packages if importlib.util.find_spec(package) is None]
    if missing:
        raise RuntimeError(
            f"Control Python {sys.executable} is missing required packages: {missing}"
        )
    if require_ffmpeg and shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg executable unavailable in simulator environment (not found on control PATH)."
        )
    contract = None
    if sensor_contract is not None:
        contract = Path(sensor_contract).expanduser().resolve()
        if not contract.is_file():
            raise FileNotFoundError(f"Required sensor contract does not exist: {contract}")
    filesystems = collect_filesystems(output)
    temp = filesystems["temp"]
    if temp["free_gib"] < MIN_TEMP_FREE_GIB:
        raise OSError(
            28,
            f"Insufficient free space in temporary filesystem {temp['filesystem_path']}: "
            f"{temp['free_gib']:.2f} GiB free",
        )
    if require_server_port:
        check_port_available(server_host, server_port)
    return {
        "python_executable": sys.executable,
        "ffmpeg": shutil.which("ffmpeg"),
        "checkpoint": validate_checkpoint(checkpoint, require_local=require_server_port),
        "sensor_contract": str(contract) if contract else None,
        "filesystems": filesystems,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }
