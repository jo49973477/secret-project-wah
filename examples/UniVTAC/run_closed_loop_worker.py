#!/usr/bin/env python3
"""Run one task in one Isaac Sim process for GR00T closed-loop evaluation."""

from __future__ import annotations

import argparse
import faulthandler
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
from univtac_rollout.core import (
    DEFAULT_GRIPPER_LIMIT_TOLERANCE_M,
    EpisodeResult,
    InvalidActionError,
    validate_action_step,
    video_output_path,
)
from univtac_rollout.policy import Gr00tUniVTACPolicy
from univtac_rollout.sensors import UniVTACSensorAdapter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from univtac_progress import emit_event  # noqa: E402


class UnsupportedRuntimeError(RuntimeError):
    def __init__(self, message: str, metadata: dict[str, Any]):
        super().__init__(message)
        self.metadata = metadata


def _diagnostic(phase: str, message: str = "") -> None:
    suffix = f": {message}" if message else ""
    print(f"[rollout-worker] phase={phase}{suffix}", flush=True)


def _parse_args() -> argparse.Namespace:
    _diagnostic("import_app_launcher", "starting")
    from isaaclab.app import AppLauncher

    _diagnostic("import_app_launcher", "complete")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--univtac-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="clean")
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default="seen")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-identity")
    parser.add_argument("--policy", choices=("auto", "rgb", "wah"), default="auto")
    parser.add_argument("--sensor-contract", type=Path)
    parser.add_argument("--scene-calibration", type=Path)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--gripper-limit-tolerance-m",
        type=float,
        default=DEFAULT_GRIPPER_LIMIT_TOLERANCE_M,
        help="Maximum Panda gripper DOF 7 limit overshoot corrected at execution time.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--expected-python", type=Path)
    parser.add_argument("--step-limit", type=int)
    parser.add_argument(
        "--debug-first-rollout", action=argparse.BooleanOptionalAction, default=True
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return args


def _load_yaml(path_or_name: str, root: Path) -> tuple[dict[str, Any], Path]:
    import yaml

    path = Path(path_or_name)
    if path.suffix not in (".yml", ".yaml"):
        path = root / "task_config" / f"{path_or_name}.yml"
    elif not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"UniVTAC task config does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"UniVTAC task config must be a mapping: {path}")
    return config, path


def _load_instructions(root: Path, task: str, instruction_type: str) -> list[str]:
    path = root / "instructions" / f"{task}.json"
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    instructions = payload.get(instruction_type)
    if (
        not isinstance(instructions, list)
        or not instructions
        or not all(isinstance(item, str) and item.strip() for item in instructions)
    ):
        raise ValueError(f"{path} has no non-empty {instruction_type!r} instruction list.")
    return instructions


def _git_value(path: Path, *arguments: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={path}", "-C", str(path), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return None


def _config_profile(config: dict[str, Any]) -> tuple[str, str, str]:
    if "env_settings" in config or "observation_settings" in config:
        required = {"env_settings", "observation_settings"}
        missing = required.difference(config)
        if missing:
            raise ValueError(
                f"Incomplete UniVTAC isaac51 task config; missing keys {sorted(missing)}."
            )
        return "isaac51", "5.1", "2.3.0"
    if "observations" in config:
        return "main", "4.5", "2.1.1"
    raise ValueError(
        "Unrecognized UniVTAC task-config schema. Expected main keys such as "
        "'observations', or isaac51 keys 'env_settings' and 'observation_settings'."
    )


def _installed_isaac_sim_version() -> str | None:
    isaac_path = os.environ.get("ISAAC_PATH")
    if isaac_path:
        version_path = Path(isaac_path) / "VERSION"
        if version_path.is_file():
            return version_path.read_text(encoding="utf-8").splitlines()[0].strip()
    for distribution in ("isaacsim", "isaac-sim"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def _installed_isaac_lab_version(isaaclab: Any) -> str | None:
    module_version = getattr(isaaclab, "__version__", None)
    if module_version is not None:
        return str(module_version)
    try:
        return importlib.metadata.version("isaaclab")
    except importlib.metadata.PackageNotFoundError:
        return None


def _isaac_lab_release_matches(
    profile: str, package_version: str | None, git_describe: str | None
) -> bool:
    expected_tag = "v2.1.1" if profile == "main" else "v2.3.0"
    if git_describe in {expected_tag, f"{expected_tag}-dirty"}:
        return True
    # Isaac Lab v2.1.1 predates aligned distribution numbering and identifies
    # its core package as 0.41.3. The isaac51 installer pins isaaclab==2.3.0.
    expected_package_versions = {
        "main": {"0.41.3", "2.1.1"},
        "isaac51": {"2.3.0"},
    }
    return package_version in expected_package_versions[profile] and git_describe is None


def _preflight(
    args: argparse.Namespace, config: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    import isaaclab

    expected_python = args.expected_python.expanduser().absolute() if args.expected_python else None
    actual_python = Path(sys.executable).absolute()
    profile, required_sim, required_lab = _config_profile(config)
    sim_version = _installed_isaac_sim_version()
    lab_version = _installed_isaac_lab_version(isaaclab)
    isaaclab_root = next(
        (
            parent
            for parent in Path(isaaclab.__file__).resolve().parents
            if (parent / ".git").is_dir()
        ),
        None,
    )
    lab_describe = (
        _git_value(isaaclab_root, "describe", "--tags", "--always", "--dirty")
        if isaaclab_root
        else None
    )
    dependency_modules = ("tacex", "tacex_assets", "tacex_uipc", "uipc", "curobo")
    missing_dependencies = [
        name for name in dependency_modules if importlib.util.find_spec(name) is None
    ]

    metadata = {
        "python_executable": str(actual_python),
        "univtac_branch": _git_value(args.univtac_root.resolve(), "branch", "--show-current"),
        "univtac_profile": profile,
        "task_config": str(config_path),
        "required_isaac_sim": required_sim,
        "required_isaac_lab": required_lab,
        "detected_isaac_sim": sim_version,
        "detected_isaac_lab": lab_version,
        "detected_isaac_lab_describe": lab_describe,
        "missing_dependencies": missing_dependencies,
    }
    _diagnostic("preflight", json.dumps(metadata, sort_keys=True))

    errors = []
    if expected_python is not None and actual_python != expected_python:
        errors.append(f"expected Python {expected_python}, but running {actual_python}")
    if sim_version is None:
        errors.append("could not determine the Isaac Sim version")
    elif not sim_version.startswith(f"{required_sim}."):
        errors.append(
            f"UniVTAC {profile} requires Isaac Sim {required_sim}.x, detected {sim_version}"
        )
    if not _isaac_lab_release_matches(profile, lab_version, lab_describe):
        errors.append(
            f"UniVTAC {profile} requires Isaac Lab release {required_lab}, detected "
            f"package {lab_version or 'unknown'} ({lab_describe or 'no git description'})"
        )
    if missing_dependencies:
        errors.append(
            "the selected Python is missing UniVTAC runtime modules "
            f"{missing_dependencies}; install the vendored TacEx/UIPC and pinned cuRobo stack"
        )
    if errors:
        raise UnsupportedRuntimeError(
            "Unsupported UniVTAC runtime:\n- "
            + "\n- ".join(errors)
            + "\nUse an isolated environment matching the checked-out UniVTAC branch; "
            "the existing Isaac Sim installation was not modified.",
            metadata,
        )
    return metadata


def _git_commit(path: Path) -> str | None:
    return _git_value(path, "rev-parse", "HEAD")


def _runtime_metadata(univtac_root: Path) -> dict[str, Any]:
    import isaaclab

    versions: dict[str, Any] = {
        "python": sys.version,
        "univtac_commit": _git_commit(univtac_root),
        "isaaclab_version": _installed_isaac_lab_version(isaaclab),
        "isaaclab_module": str(Path(isaaclab.__file__).resolve()),
    }
    for distribution in ("isaacsim", "isaac-sim", "isaaclab"):
        try:
            versions[f"distribution_{distribution}"] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    isaaclab_root = next(
        (
            parent
            for parent in Path(isaaclab.__file__).resolve().parents
            if (parent / ".git").is_dir()
        ),
        None,
    )
    versions["isaaclab_commit"] = _git_commit(isaaclab_root) if isaaclab_root else None
    return versions


def _positive_int(mapping: dict[str, Any], name: str, *, allow_zero: bool = False) -> int:
    value = mapping.get(name)
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer; got {value!r}.")
    return value


def _apply_task_config(
    env_cfg: Any,
    task_config: dict[str, Any],
    *,
    task_name: str,
    output_dir: Path,
) -> tuple[str, dict[str, Any]]:
    profile, _, _ = _config_profile(task_config)
    env_cfg.save_dir = output_dir / "logs" / "univtac" / task_name
    if profile == "main":
        env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
        observations = task_config.get("observations", {})
        env_cfg.obs_data_type = observations
        env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
        env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
        env_cfg.random_texture = task_config.get("random_texture", False)
        env_cfg.tactile_sensor_type = task_config.get("sensor_type", env_cfg.tactile_sensor_type)
    else:
        settings = task_config["env_settings"]
        frequencies = settings["frequencies"]
        physical_hz = _positive_int(frequencies, "physical")
        eval_hz = _positive_int(frequencies, "eval")
        video_hz = _positive_int(frequencies, "video", allow_zero=True)
        render_hz = _positive_int(frequencies, "render", allow_zero=True)
        if physical_hz % eval_hz:
            raise ValueError(
                f"physical frequency {physical_hz} must be divisible by eval {eval_hz}."
            )
        decimation = physical_hz // eval_hz

        def control_period(name: str, frequency: int) -> int:
            if frequency == 0:
                return 0
            if physical_hz % frequency:
                raise ValueError(
                    f"physical frequency {physical_hz} must be divisible by {name} {frequency}."
                )
            physics_period = physical_hz // frequency
            if physics_period % decimation:
                raise ValueError(
                    f"{name} period {physics_period} is not divisible by decimation {decimation}."
                )
            return physics_period // decimation

        observations = task_config["observation_settings"]
        env_cfg.obs_data_type = observations
        env_cfg.tactile_sensor_type = settings["sensor_type"]
        if not hasattr(env_cfg, "tactile_optical_backend"):
            raise RuntimeError(
                "The isaac51 config requests an optical backend, but TaskCfg lacks "
                "tactile_optical_backend. The UniVTAC code/config branches do not match."
            )
        env_cfg.tactile_optical_backend = settings["optical_backend"]
        env_cfg.random_texture = settings["random_texture"]
        env_cfg.decimation = decimation
        env_cfg.save_frequency = 1
        env_cfg.video_frequency = control_period("video", video_hz)
        env_cfg.render_frequency = control_period("render", render_hz)
        env_cfg.sim.dt = 1.0 / physical_hz
        env_cfg.uipc_sim.dt = env_cfg.sim.dt
        env_cfg.sim.render_interval = decimation
        for camera_cfg in env_cfg.cameras:
            camera_cfg.update_period = env_cfg.sim.dt
        for option_name, option_value in task_config.get("task_overrides", {}).items():
            if not hasattr(env_cfg, option_name):
                raise ValueError(f"Unknown task override {option_name!r} for task {task_name!r}.")
            setattr(env_cfg, option_name, option_value)
    return profile, observations


def _qpos_limits(task: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Read Isaac Lab's articulation soft limits in UniVTAC's 7+1 qpos order."""
    manager = getattr(task, "_robot_manager", None)
    robot = getattr(manager, "robot", None)
    data = getattr(robot, "data", None)
    limits = getattr(data, "soft_joint_pos_limits", None)
    if limits is None:
        return None
    limits = limits.detach().cpu().numpy() if hasattr(limits, "detach") else np.asarray(limits)
    if limits.ndim == 3:
        limits = limits[0]
    if limits.ndim != 2 or limits.shape[1] != 2:
        raise ValueError(f"Unexpected articulation limit shape: {limits.shape}")

    arm_ids = np.asarray(manager._arm_ids.detach().cpu().numpy(), dtype=np.int64)
    gripper_ids = np.asarray(manager._gripper_ids.detach().cpu().numpy(), dtype=np.int64)
    if arm_ids.shape != (7,) or gripper_ids.size < 1:
        raise ValueError(
            f"Unexpected UniVTAC DOF mapping: arm_ids={arm_ids}, gripper_ids={gripper_ids}"
        )
    joint_names = list(robot.joint_names)
    arm_names = [joint_names[index] for index in arm_ids]
    gripper_names = [joint_names[index] for index in gripper_ids]
    expected_arm_names = list(getattr(manager, "_arm_joint_names", ()))
    expected_gripper_names = list(getattr(manager, "_gripper_joint_names", ()))
    if arm_names != expected_arm_names or gripper_names != expected_gripper_names:
        raise ValueError(
            "UniVTAC action-to-articulation DOF mapping is inconsistent: "
            f"arm={arm_names} expected={expected_arm_names}, "
            f"gripper={gripper_names} expected={expected_gripper_names}."
        )
    _diagnostic(
        "dof_mapping",
        f"all={joint_names} action_arm={arm_names} action_gripper={gripper_names}",
    )
    lower = np.concatenate((limits[arm_ids, 0], [np.max(limits[gripper_ids, 0])])).astype(
        np.float32
    )
    upper = np.concatenate((limits[arm_ids, 1], [np.min(limits[gripper_ids, 1])])).astype(
        np.float32
    )
    return lower, upper


def _inspect_articulation(task: Any) -> dict[str, Any]:
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is unavailable after UniVTAC task construction.")
    root_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if not root_paths:
        raise RuntimeError("No prim with ArticulationRootAPI exists on the UniVTAC stage.")
    manager = task._robot_manager
    robot = manager.robot
    view = getattr(robot, "root_physx_view", None)
    view_paths = [str(path) for path in getattr(view, "prim_paths", ())]
    if view_paths and not any(
        view_path == root_path or view_path.startswith(root_path + "/")
        for view_path in view_paths
        for root_path in root_paths
    ):
        raise RuntimeError(
            f"Robot PhysX view paths {view_paths} do not descend from articulation roots {root_paths}."
        )
    result = {
        "articulation_root_prim_paths": root_paths,
        "robot_physx_view_paths": view_paths,
        "robot_joint_names": list(robot.joint_names),
    }
    _diagnostic("articulation_inspection", json.dumps(result, sort_keys=True))
    return result


def _write_worker_results(
    path: Path,
    episodes: list[EpisodeResult],
    metadata: dict[str, Any],
    *,
    completed: bool,
    phase: str,
    fatal_error: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "episodes": [episode.__dict__ for episode in episodes],
                "runtime_metadata": metadata,
                "completed": completed,
                "phase": phase,
                "fatal_error": fatal_error,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_partial_worker_results(path: Path) -> tuple[list[EpisodeResult], dict[str, Any]]:
    if not path.is_file():
        return [], {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            [EpisodeResult(**row) for row in payload.get("episodes", [])],
            payload.get("runtime_metadata", {}),
        )
    except Exception:
        return [], {}


def _finish_video(
    task: Any,
    *,
    monitor: dict[str, Any],
    output_dir: Path,
    task_name: str,
    seed: int,
    success: bool | None,
    termination_reason: str,
) -> Path:
    tag = "success" if success is True else "failed" if success is False else "error"
    ffmpeg_process = getattr(task.video_handler, "ffmpeg", None)
    _diagnostic(
        "video_finalize",
        f"tag={tag} frames={monitor['frames']} ffmpeg_pid={getattr(ffmpeg_process, 'pid', None)}",
    )
    task.clean_cache(result=tag)
    if monitor["frames"] < 1:
        raise RuntimeError("UniVTAC video handler received no frames.")
    if ffmpeg_process is None:
        raise RuntimeError("UniVTAC video handler never initialized FFmpeg.")
    if ffmpeg_process.returncode != 0:
        raise RuntimeError(f"FFmpeg exited with code {ffmpeg_process.returncode}.")
    source = task.save_video_path.with_name(f"{task.save_video_path.stem}_{tag}.mp4")
    destination = video_output_path(
        output_dir,
        task=task_name,
        seed=seed,
        success=success,
        termination_reason=termination_reason,
    )
    if not source.is_file():
        raise FileNotFoundError(f"UniVTAC video handler did not create {source}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite rollout video: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), destination)
    _validate_mp4(destination)
    _diagnostic(
        "video_complete",
        f"path={destination} bytes={destination.stat().st_size} frames={monitor['frames']} "
        f"ffmpeg_return_code={ffmpeg_process.returncode}",
    )
    return destination


def _validate_mp4(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Finalized MP4 is missing or empty: {path}")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                f"ffprobe could not read {path} (code {probe.returncode}): {probe.stderr.strip()}"
            )
        payload = json.loads(probe.stdout)
        streams = payload.get("streams", [])
        if not streams or streams[0].get("codec_name") != "h264":
            raise RuntimeError(f"MP4 has no readable H.264 video stream: {path}")
        return

    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        readable, frame = capture.read()
        if not readable or frame is None or frame.size == 0:
            raise RuntimeError(f"OpenCV could not read a frame from {path}")
    finally:
        capture.release()


def _instrument_video_handler(task: Any) -> dict[str, Any]:
    monitor: dict[str, Any] = {"frames": 0, "resets": 0}
    handler = task.video_handler
    original_reset = handler.reset
    original_write = handler.write

    def monitored_reset(video_path: Any, video_size: Any) -> Any:
        monitor["resets"] += 1
        monitor["frames"] = 0
        _diagnostic("video_reset", f"path={video_path} size={video_size}")
        result = original_reset(video_path, video_size)
        process = getattr(handler, "ffmpeg", None)
        if process is None or process.poll() is not None:
            raise RuntimeError("UniVTAC VideoHandler failed to start FFmpeg.")
        _diagnostic("video_ffmpeg_started", f"pid={process.pid} command={process.args!r}")
        return result

    def monitored_write(frame: Any) -> Any:
        result = original_write(frame)
        monitor["frames"] += 1
        if monitor["frames"] == 1:
            _diagnostic("video_first_frame", f"shape={tuple(frame.shape)}")
        return result

    handler.reset = monitored_reset
    handler.write = monitored_write
    return monitor


def _run(
    args: argparse.Namespace,
    task_config: dict[str, Any],
    task_config_path: Path,
    metadata: dict[str, Any],
) -> None:
    if not np.isfinite(args.gripper_limit_tolerance_m) or args.gripper_limit_tolerance_m < 0:
        raise ValueError("--gripper-limit-tolerance-m must be finite and nonnegative.")
    _diagnostic("import_torch", "starting")
    import torch

    _diagnostic("import_torch", f"complete version={torch.__version__}")
    root = args.univtac_root.resolve()
    sys.path.insert(0, str(root))
    _diagnostic("load_instructions", "starting")
    instructions = _load_instructions(root, args.task, args.instruction_type)
    _diagnostic("load_instructions", f"complete count={len(instructions)}")
    _write_worker_results(
        args.result_json,
        [],
        metadata,
        completed=False,
        phase="import_task_module",
    )
    _diagnostic("import_task_module", f"envs.{args.task}")
    task_module = importlib.import_module(f"envs.{args.task}")
    _diagnostic("import_task_module", "complete")

    _diagnostic("configure_task", "starting")
    env_cfg = task_module.TaskCfg()
    profile, observations = _apply_task_config(
        env_cfg,
        task_config,
        task_name=args.task,
        output_dir=args.output_dir,
    )
    env_cfg.scene.num_envs = 1
    if getattr(args, "device", None) is not None:
        env_cfg.sim.device = args.device
    if args.step_limit is not None:
        if args.step_limit < 1:
            raise ValueError("--step-limit must be positive.")
        env_cfg.step_lim = args.step_limit
    if env_cfg.video_frequency <= 0:
        raise ValueError("Rollout evaluation requires a positive video frequency.")
    required_observations = {
        "camera": {"rgb"},
        "embodiment": {"joint"},
        "tactile": {"rgb_marker"},
    }
    for group, required in required_observations.items():
        actual = set(observations.get(group, []))
        if not required.issubset(actual):
            raise ValueError(
                f"Task config {task_config_path} must request {group} observations "
                f"{sorted(required)}; got {sorted(actual)}."
            )

    metadata.update(_runtime_metadata(root))
    metadata.update(
        {
            "task_config": str(task_config_path),
            "task_config_profile": profile,
            "task": args.task,
            "device": str(env_cfg.sim.device),
            "action_safety": {
                "policy_scope": "rgb_and_wah",
                "corrected_dof": 7,
                "gripper_limit_tolerance_m": args.gripper_limit_tolerance_m,
            },
        }
    )
    try:
        from isaacsim.core.version import get_version

        metadata["isaac_sim_runtime_version"] = get_version()[0]
    except Exception:
        metadata["isaac_sim_runtime_version"] = None
    _diagnostic("configure_task", json.dumps(metadata, sort_keys=True))

    task = None
    policy = None
    sensor_adapter = None
    episodes: list[EpisodeResult] = []
    try:
        _write_worker_results(
            args.result_json,
            episodes,
            metadata,
            completed=False,
            phase="policy_initialization",
        )
        _diagnostic("policy_initialization", "starting ping/config validation")
        policy = Gr00tUniVTACPolicy(
            args.server_host,
            args.server_port,
            execution_horizon=args.execution_horizon,
            timeout_ms=round(args.request_timeout_seconds * 1000),
            checkpoint=args.checkpoint,
            expected_checkpoint_identity=args.checkpoint_identity,
            policy=args.policy,
            sensor_contract=args.sensor_contract,
            scene_calibration=args.scene_calibration,
        )
        if policy.policy_type == "wah":
            camera_types = env_cfg.obs_data_type.setdefault("camera", [])
            if "depth" not in camera_types:
                camera_types.append("depth")
            for camera_cfg in getattr(env_cfg, "cameras", []):
                if hasattr(camera_cfg, "data_types"):
                    data_types = list(camera_cfg.data_types)
                    if "depth" not in data_types:
                        data_types.append("depth")
                    camera_cfg.data_types = data_types
            embodiment_types = env_cfg.obs_data_type.setdefault("embodiment", [])
            if "ee" not in embodiment_types:
                embodiment_types.append("ee")
            source = policy.history.sensor_contract["tactile"]["rgb"]["source_key"].split("/")[-1]
            tactile_types = env_cfg.obs_data_type.setdefault("tactile", [])
            if source not in tactile_types:
                tactile_types.append(source)
        _diagnostic("policy_initialization", "complete")

        _write_worker_results(
            args.result_json,
            episodes,
            metadata,
            completed=False,
            phase="task_construction",
        )
        _diagnostic("task_construction", "starting")
        task = task_module.Task(env_cfg, mode="eval")
        _diagnostic("task_construction", "complete")
        metadata.update(_inspect_articulation(task))
        if policy.policy_type == "wah":
            sensor_adapter = UniVTACSensorAdapter(task, policy.history.sensor_contract)
            metadata["wah_sensor_adapter"] = sensor_adapter.metadata
            _diagnostic("wah_sensor_adapter", json.dumps(sensor_adapter.metadata, sort_keys=True))
        video_monitor = _instrument_video_handler(task)
        limits = _qpos_limits(task)
        if limits is None:
            _diagnostic("articulation_validation", "soft joint limits unavailable")
        else:
            _diagnostic(
                "articulation_validation",
                f"lower={limits[0].tolist()} upper={limits[1].tolist()}",
            )

        for episode_index in range(args.episodes):
            seed = args.seed_start + episode_index
            start = time.perf_counter()
            success: bool | None = False
            termination_reason = "step_limit"
            error = ""
            video_path = ""
            instruction = ""
            reset_started = False
            try:
                _write_worker_results(
                    args.result_json,
                    episodes,
                    metadata,
                    completed=False,
                    phase=f"episode_{episode_index}_reset",
                )
                _diagnostic("episode_reset", f"index={episode_index} seed={seed} starting")
                task.mode = "eval"
                task.reset(seed=seed, instructions=instructions)
                reset_started = True
                instruction = str(task.instruction)
                task.mean_steps = task.cfg.step_lim
                policy.reset()
                _diagnostic("episode_reset", f"index={episode_index} complete")
                _diagnostic("observation_initial", "starting")
                observation = task._get_observations()
                if sensor_adapter is not None:
                    observation = sensor_adapter.attach(observation)
                    _diagnostic(
                        "wah_calibration",
                        json.dumps(sensor_adapter.metadata, sort_keys=True),
                    )
                _diagnostic("observation_initial", "complete")
                task.video_handler.write(task.get_frame_shot(observation))

                while task.take_action_cnt < task.cfg.step_lim:
                    _diagnostic("gr00t_inference", f"action_index={task.take_action_cnt} starting")
                    action_chunk, formatted = policy.predict_chunk(
                        observation, task.instruction, timestamp=task.take_action_cnt
                    )
                    _diagnostic(
                        "gr00t_inference",
                        f"complete shape={action_chunk.shape} dtype={action_chunk.dtype}",
                    )
                    first_action = episode_index == 0 and task.take_action_cnt == 0
                    if args.debug_first_rollout and first_action:
                        debug = {
                            "head": [
                                formatted["video"]["head"].shape,
                                str(formatted["video"]["head"].dtype),
                            ],
                            "wrist": [
                                formatted["video"]["wrist"].shape,
                                str(formatted["video"]["wrist"].dtype),
                            ],
                            "joint": [
                                formatted["state"]["joint"].shape,
                                str(formatted["state"]["joint"].dtype),
                            ],
                            "action_chunk": [action_chunk.shape, str(action_chunk.dtype)],
                        }
                        _diagnostic("first_action_contract", repr(debug))

                    for predicted in action_chunk:
                        lower, upper = limits if limits is not None else (None, None)
                        action = validate_action_step(
                            predicted,
                            lower=lower,
                            upper=upper,
                            gripper_limit_tolerance_m=args.gripper_limit_tolerance_m,
                        )
                        original_action = np.asarray(predicted, dtype=np.float32)
                        correction = action - original_action
                        if np.any(correction):
                            _diagnostic(
                                "action_safety_postprocess",
                                f"index={task.take_action_cnt} policy={policy.policy_type} "
                                f"original={original_action.tolist()} "
                                f"corrected={action.tolist()} correction={correction.tolist()} "
                                f"gripper_limit_tolerance_m={args.gripper_limit_tolerance_m}",
                            )
                        action_tensor = torch.as_tensor(
                            action, dtype=torch.float32, device=task.device
                        )
                        _diagnostic(
                            "execute_action",
                            f"index={task.take_action_cnt} qpos={action.tolist()}",
                        )
                        executed, eval_success = task.take_action(action_tensor, action_type="qpos")
                        _diagnostic("observation_post_action", "starting")
                        observation = task._get_observations()
                        if sensor_adapter is not None:
                            observation = sensor_adapter.attach(observation)
                        policy.record_observation(
                            observation, task.instruction, task.take_action_cnt
                        )
                        post_joint = observation["embodiment"]["joint"]
                        if hasattr(post_joint, "detach"):
                            post_joint = post_joint.detach().cpu().numpy()
                        post_joint = np.asarray(post_joint).reshape(-1)[:8]
                        if not np.isfinite(post_joint).all():
                            raise RuntimeError("Post-action joint observation contains NaN/Inf.")
                        _diagnostic(
                            "observation_post_action",
                            f"complete action_count={task.take_action_cnt} joint={post_joint.tolist()}",
                        )
                        if eval_success or task.eval_success:
                            success = True
                            termination_reason = "success"
                            break
                        if not executed or task.check_early_stop():
                            success = False
                            termination_reason = "early_stop"
                            break
                        if task.take_action_cnt >= task.cfg.step_lim:
                            success = False
                            termination_reason = "step_limit"
                            break
                    if (
                        termination_reason != "step_limit"
                        or task.take_action_cnt >= task.cfg.step_lim
                    ):
                        break
            except InvalidActionError as exc:
                success = None
                termination_reason = "invalid_action"
                error = str(exc)
                _diagnostic("episode_invalid_action", f"seed={seed} error={error}")
            except Exception:
                success = None
                termination_reason = "exception"
                error = traceback.format_exc()
                _diagnostic("episode_exception", f"seed={seed}\n{error}")
            finally:
                if reset_started or getattr(task.video_handler, "ffmpeg", None) is not None:
                    try:
                        video_path = str(
                            _finish_video(
                                task,
                                monitor=video_monitor,
                                output_dir=args.output_dir,
                                task_name=args.task,
                                seed=seed,
                                success=success,
                                termination_reason=termination_reason,
                            )
                        )
                    except Exception:
                        video_error = traceback.format_exc()
                        error = f"{error}\nVideo finalization error:\n{video_error}".strip()
                        if success is not None:
                            success = None
                            termination_reason = "exception"
                        _diagnostic("video_error", video_error)

            result = EpisodeResult(
                task=args.task,
                seed=seed,
                episode_index=episode_index,
                success=success,
                num_steps=int(getattr(task, "take_action_cnt", 0)) if reset_started else 0,
                sim_steps=int(getattr(task, "step_count", 0)) if reset_started else 0,
                execution_horizon=args.execution_horizon,
                checkpoint=args.checkpoint,
                instruction=instruction,
                elapsed_seconds=time.perf_counter() - start,
                termination_reason=termination_reason,
                video_path=video_path,
                error=error,
            )
            episodes.append(result)
            _write_worker_results(
                args.result_json,
                episodes,
                metadata,
                completed=False,
                phase=f"episode_{episode_index}_complete",
            )
            _diagnostic(
                "episode_complete",
                f"count={episode_index + 1}/{args.episodes} task={args.task} seed={seed} "
                f"result={termination_reason} actions={result.num_steps} "
                f"elapsed={result.elapsed_seconds:.2f}s video={video_path or 'MISSING'}",
            )
            status = "error" if result.success is None else "success" if result.success else "failure"
            emit_event(
                "episode",
                task=args.task,
                index=episode_index,
                status=status,
            )

        _write_worker_results(
            args.result_json,
            episodes,
            metadata,
            completed=True,
            phase="complete",
        )
    finally:
        if policy is not None:
            _diagnostic("policy_close", "starting")
            policy.close()
            _diagnostic("policy_close", "complete")
        if task is not None:
            _diagnostic("task_close", "starting")
            task.close()
            _diagnostic("task_close", "complete")


def main() -> None:
    faulthandler.enable(all_threads=True)
    args = _parse_args()
    root = args.univtac_root.resolve()
    task_config, task_config_path = _load_yaml(args.task_config, root)
    metadata: dict[str, Any] = {}
    simulation_app = None
    try:
        metadata.update(_preflight(args, task_config, task_config_path))
        _write_worker_results(
            args.result_json,
            [],
            metadata,
            completed=False,
            phase="app_launcher_initialization",
        )
        _diagnostic("app_launcher_initialization", "starting")
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        _diagnostic("app_launcher_initialization", "complete")
        _run(args, task_config, task_config_path, metadata)
    except BaseException as exc:
        if isinstance(exc, UnsupportedRuntimeError):
            metadata.update(exc.metadata)
        fatal_error = traceback.format_exc()
        _diagnostic("fatal_error", fatal_error)
        episodes, saved_metadata = _read_partial_worker_results(args.result_json)
        saved_metadata.update(metadata)
        _write_worker_results(
            args.result_json,
            episodes,
            saved_metadata,
            completed=False,
            phase="fatal_error",
            fatal_error=fatal_error,
        )
        raise
    finally:
        if simulation_app is not None:
            _diagnostic("app_close", "starting")
            simulation_app.close()
            _diagnostic("app_close", "complete")


if __name__ == "__main__":
    main()
