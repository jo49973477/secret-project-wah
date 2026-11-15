"""Checkpoint-driven UniVTAC sensor history and calibrated scene conversion."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Any

import numpy as np
from scene_pointcloud import (
    ScenePreprocessConfig,
    build_scene_frame,
    pose_wxyz_to_matrix,
    validate_intrinsics,
    validate_transform,
)

from .core import _to_numpy, format_univtac_observation


UNIVTAC_TACTILE_RGB_ENCODING = "univtac_tactile_rgb"
UINT8_RGB_ENCODING = "uint8_0_255"


def field(config: Any, name: str) -> Any:
    return config.get(name) if isinstance(config, dict) else getattr(config, name, None)


def detect_policy(configs: dict[str, Any], requested: str = "auto") -> str:
    if requested not in {"auto", "rgb", "wah"}:
        raise ValueError(f"Unknown policy {requested!r}")
    kind = "wah" if {"pointcloud", "tactile"}.intersection(configs) else "rgb"
    if requested != "auto" and requested != kind:
        raise ValueError(
            f"Requested --policy {requested}, but checkpoint requires {kind} modalities."
        )
    return kind


def validate_contract(configs: dict[str, Any], execution_horizon: int, requested: str) -> str:
    kind = detect_policy(configs, requested)
    expected = {
        "video": ["head", "wrist"],
        "state": ["joint"],
        "action": ["joint"],
        "language": ["annotation.human.task_description"],
        "pointcloud": ["scene"],
        "tactile": ["rgb"],
    }
    missing = set(expected).difference(configs)
    if missing.intersection({"video", "state", "action", "language"}):
        raise ValueError(f"Checkpoint modality config is missing {sorted(missing)}")
    if kind == "wah" and "pointcloud" not in configs:
        raise ValueError("WAH rollout needs pointcloud.scene.")
    for name, config in configs.items():
        if name not in expected or field(config, "modality_keys") != expected[name]:
            raise ValueError(
                f"Unsupported checkpoint {name} keys: {field(config, 'modality_keys')}"
            )
        delta = field(config, "delta_indices")
        if not delta or any(not isinstance(i, int) for i in delta):
            raise ValueError(f"Invalid {name}.delta_indices: {delta}")
        if name not in {"action", "language"} and (delta[-1] != 0 or any(i > 0 for i in delta)):
            raise ValueError(f"{name}.delta_indices must be past offsets ending at zero: {delta}")
    horizon = len(field(configs["action"], "delta_indices"))
    if not 1 <= execution_horizon <= horizon:
        raise ValueError(
            f"execution_horizon={execution_horizon} exceeds checkpoint action horizon {horizon}."
        )
    if len(field(configs["language"], "delta_indices")) != 1:
        raise ValueError("GR00T inference supports one language timestamp.")
    return kind


def load_sensor_contract(
    path: str | Path | None, *, require_tactile: bool = True
) -> dict[str, Any]:
    if path is None:
        raise ValueError(
            "WAH requires --sensor-contract with the training dataset meta/modality.json. "
            "It records point sampling and, when required, the actual tactile source."
        )
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    scene = data.get("pointcloud", {}).get("scene")
    if not isinstance(scene, dict):
        raise ValueError("Sensor contract needs a pointcloud.scene entry.")
    for key in ("num_points", "feature_dim", "voxel_size_m", "coordinate_frame"):
        if key not in scene:
            raise ValueError(f"Sensor contract pointcloud.scene is missing {key}.")
    if scene["feature_dim"] != 6 or scene["coordinate_frame"] != "robot_base":
        raise ValueError("Scene contract must be robot_base XYZRGB with feature_dim=6.")
    if not isinstance(scene["num_points"], int) or scene["num_points"] < 1:
        raise ValueError("Sensor contract pointcloud.scene.num_points must be positive.")
    if not isinstance(scene["voxel_size_m"], (int, float)) or scene["voxel_size_m"] <= 0:
        raise ValueError("Sensor contract pointcloud.scene.voxel_size_m must be positive.")
    source_cameras = scene.get("source_cameras")
    if source_cameras is not None and source_cameras != ["head", "wrist"]:
        raise ValueError(
            "WAH sensor contract source_cameras must be ordered ['head', 'wrist']; "
            f"got {source_cameras!r}."
        )
    if not require_tactile:
        return data
    tactile = data.get("tactile", {}).get("rgb")
    if not isinstance(tactile, dict):
        raise ValueError("Sensor contract needs a tactile.rgb entry.")
    source_key = tactile.get("source_key")
    source_parts = source_key.split("/") if isinstance(source_key, str) else []
    if (
        not isinstance(source_key, str)
        or len(source_parts) not in {2, 3}
        or source_parts[0] != "tactile"
        or source_parts[-1] not in {"rgb", "rgb_marker"}
        or any(not part for part in source_parts)
    ):
        raise ValueError(f"Unsupported tactile training source_key {tactile.get('source_key')!r}.")
    return data


def add_scene_calibration(contract: dict[str, Any], path: str | Path | None) -> dict[str, Any]:
    if path is not None:
        calibration = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(calibration, dict) or not isinstance(calibration.get("cameras"), dict):
            raise ValueError("Scene calibration JSON must contain a cameras mapping.")
        cameras = calibration["cameras"]
        unknown = set(cameras).difference({"head", "wrist"})
        if unknown:
            raise ValueError(f"Scene calibration has unknown cameras: {sorted(unknown)}")
        if "T_base_camera" in cameras.get("wrist", {}):
            raise ValueError(
                "Static wrist T_base_camera in scene calibration is invalid. Supply "
                "T_ee_camera or use the live per-frame camera pose."
            )
        for name, camera in cameras.items():
            if not isinstance(camera, dict):
                raise ValueError(f"Scene calibration cameras.{name} must be a mapping.")
            if "intrinsics" in camera:
                validate_intrinsics(camera["intrinsics"], source=f"scene calibration {name}")
            for transform_name in ("T_base_camera", "T_ee_camera"):
                if transform_name in camera:
                    validate_transform(
                        camera[transform_name], source=f"scene calibration {name}.{transform_name}"
                    )
            if "depth_scale" in camera and (
                not isinstance(camera["depth_scale"], (int, float))
                or not np.isfinite(camera["depth_scale"])
                or camera["depth_scale"] <= 0
            ):
                raise ValueError(f"Scene calibration {name}.depth_scale must be positive.")
        contract = dict(contract)
        contract["calibration"] = calibration
    return contract


def scene_frame(
    observation: dict[str, Any], contract: dict[str, Any], timestamp: int
) -> np.ndarray:
    meta = contract["pointcloud"]["scene"]
    sensors = observation.get("observation", {})
    prebuilt = sensors.get("pointcloud", {}).get("scene")
    if prebuilt is not None:
        reported_frame = sensors["pointcloud"].get("coordinate_frame")
        units = sensors["pointcloud"].get("xyz_units")
        if reported_frame != "robot_base" or units != "m":
            raise ValueError(
                "Prebuilt observation.pointcloud.scene must declare coordinate_frame='robot_base' "
                f"and xyz_units='m'; got {reported_frame!r}, {units!r}."
            )
        preprocessing = sensors["pointcloud"].get("preprocessing", {})
        for key in ("num_points", "voxel_size_m", "sampling_seed"):
            if preprocessing.get(key) != meta.get(key, 0 if key == "sampling_seed" else None):
                raise ValueError(
                    f"Prebuilt scene preprocessing {key}={preprocessing.get(key)!r} does not "
                    f"match the training contract {meta.get(key)!r}."
                )
        points = np.asarray(_to_numpy(prebuilt, name="scene point cloud"), dtype=np.float32)
        if (
            points.shape != (meta["num_points"], 6)
            or not np.isfinite(points).all()
            or (points[:, 3:] < 0).any()
            or (points[:, 3:] > 255).any()
        ):
            raise ValueError(
                f"Prebuilt scene must be finite [{meta['num_points']},6] robot_base XYZRGB."
            )
        return np.ascontiguousarray(points)
    calibration = contract.get("calibration", {}).get("cameras", {})
    if "T_base_camera" in calibration.get("wrist", {}):
        raise ValueError(
            "Static wrist T_base_camera in scene calibration is invalid. Supply "
            "T_ee_camera plus current embodiment.ee, or per-frame observation.wrist.T_base_camera."
        )
    camera_data = {}
    for camera in ("head", "wrist"):
        camera_data[camera] = dict(calibration.get(camera, {}))
        camera_data[camera].update(sensors.get(camera, {}))
    wrist = camera_data["wrist"]
    if (
        "T_base_camera" not in wrist
        and "T_ee_camera" in wrist
        and "ee" in observation.get("embodiment", {})
    ):
        ee_pose = _to_numpy(observation["embodiment"]["ee"], name="end-effector pose").reshape(-1)
        wrist["T_base_camera"] = pose_wxyz_to_matrix(
            ee_pose, source="embodiment.ee"
        ) @ validate_transform(
            _to_numpy(wrist["T_ee_camera"], name="wrist extrinsic"),
            source="wrist T_ee_camera",
        )
    required = ("rgb", "depth", "intrinsics", "depth_scale", "T_base_camera")
    missing = [
        f"observation.{camera}.{key}"
        for camera in ("head", "wrist")
        for key in required
        if key not in camera_data[camera]
    ]
    if missing:
        raise ValueError(
            f"WAH scene cloud needs calibrated head and wrist RGB-D in robot_base; missing {missing}. "
            "Enable depth and expose intrinsics, metric depth scale, and per-frame wrist extrinsics."
        )
    config = ScenePreprocessConfig(
        workspace_min=meta.get("workspace_min_m"),
        workspace_max=meta.get("workspace_max_m"),
        voxel_size=meta["voxel_size_m"],
        max_points=meta["num_points"],
        seed=meta.get("sampling_seed", 0),
    )
    head, wrist = camera_data["head"], camera_data["wrist"]
    for name, sensor in (("head", head), ("wrist", wrist)):
        for key in required:
            sensor[key] = _to_numpy(sensor[key], name=f"{name}.{key}")
    points, _ = build_scene_frame(
        head_rgb=head["rgb"],
        head_depth=head["depth"],
        head_intrinsics=head["intrinsics"],
        head_depth_scale=float(head["depth_scale"]),
        transform_base_head=head["T_base_camera"],
        wrist_rgb=wrist["rgb"],
        wrist_depth=wrist["depth"],
        wrist_intrinsics=wrist["intrinsics"],
        wrist_depth_scale=float(wrist["depth_scale"]),
        transform_base_wrist=wrist["T_base_camera"],
        config=config,
        frame_index=timestamp,
    )
    return points


def _tactile_encoding(source_key: str, adapted: dict[str, Any]) -> str:
    """Return an encoding from sensor provenance, never from one frame's extrema."""
    encoding = adapted.get("encoding")
    if encoding is not None:
        if encoding not in {UNIVTAC_TACTILE_RGB_ENCODING, UINT8_RGB_ENCODING}:
            raise ValueError(f"Unsupported tactile encoding {encoding!r} for {source_key}.")
        return encoding
    # UniVTAC's public ``rgb`` buffer is float32. Its reset path receives the optical
    # simulator's normalized [0, 1] output, while its update path assigns quantized
    # [0, 255] values back into that same float32 buffer. ``rgb_marker`` is uint8.
    return UNIVTAC_TACTILE_RGB_ENCODING if source_key.endswith("/rgb") else UINT8_RGB_ENCODING


def _tactile_rgb_to_uint8(value: Any, *, source_key: str, encoding: str) -> np.ndarray:
    """Match the uint8 video frames produced by the UniVTAC training converter."""
    image = _to_numpy(value, name=source_key.replace("/", "."))
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
        raise ValueError(f"{source_key} must have shape [H,W,3]; got {image.shape}.")

    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(
            f"{source_key} must be uint8 or a documented UniVTAC floating RGB stream; "
            f"got dtype {image.dtype}."
        )
    if encoding != UNIVTAC_TACTILE_RGB_ENCODING:
        raise ValueError(
            f"{source_key} uses encoding {encoding!r}, which requires uint8; got {image.dtype}."
        )
    if not np.isfinite(image).all():
        raise ValueError(f"{source_key} contains NaN or Inf.")

    minimum = float(image.min())
    maximum = float(image.max())
    if minimum < 0.0 or maximum > 255.0:
        raise ValueError(
            f"{source_key} values must be in [0,1] or [0,255]; got min={minimum:.8g}, "
            f"max={maximum:.8g}."
        )

    # The encoding is selected from UniVTAC sensor provenance above, not inferred from
    # arbitrary image statistics. Within that documented dual-range public buffer, values
    # above one unambiguously identify the update-path [0, 255] representation; otherwise
    # the official normalized optical-simulator representation is used.
    scaled = image * 255.0 if maximum <= 1.0 else image
    return np.ascontiguousarray(np.rint(scaled).astype(np.uint8))


def tactile_frame(observation: dict[str, Any], contract: dict[str, Any]) -> np.ndarray:
    source_key = contract["tactile"]["rgb"]["source_key"]
    source_parts = source_key.split("/")[1:]
    sensors = observation.get("tactile", {})
    value: Any = sensors
    for part in source_parts:
        if not isinstance(value, dict) or part not in value:
            value = None
            break
        value = value[part]
    adapted = sensors.get("_wah_contract_source", {}) if isinstance(sensors, dict) else {}
    if value is None:
        if adapted.get("source_key") == source_key:
            value = adapted.get("image")
    if value is None:
        raise ValueError(
            f"WAH tactile.rgb was trained from {source_key}; observation has "
            f"{sorted(sensors) if isinstance(sensors, dict) else type(sensors).__name__}. "
            "Request that exact UniVTAC tactile sensor stream. Camera RGB cannot substitute."
        )
    encoding = _tactile_encoding(
        source_key,
        adapted if adapted.get("source_key") == source_key else {},
    )
    return _tactile_rgb_to_uint8(value, source_key=source_key, encoding=encoding)


class ObservationHistory:
    """Synchronized per-inference samples with episode-start repeat padding."""

    def __init__(self, configs: dict[str, Any], sensor_contract: dict[str, Any] | None):
        if {"pointcloud", "tactile"}.intersection(configs) and sensor_contract is None:
            raise ValueError("Active pointcloud/tactile modalities require a sensor contract.")
        self.configs = configs
        self.sensor_contract = sensor_contract
        self.capacity = 1 + max(
            -min(field(c, "delta_indices")) for name, c in configs.items() if name != "action"
        )
        self.frames: deque[dict[str, dict[str, Any]]] = deque(maxlen=self.capacity)
        self.timestamp = -1

    def reset(self) -> None:
        self.frames.clear()
        self.timestamp = -1

    def append(
        self, observation: dict[str, Any], instruction: str, timestamp: int
    ) -> dict[str, dict[str, Any]]:
        if timestamp <= self.timestamp:
            raise ValueError(f"Observation timestamp {timestamp} is not after {self.timestamp}.")
        rgb = format_univtac_observation(observation, instruction)
        frame = {
            name: {key: value[0, 0] for key, value in rgb[name].items()}
            for name in ("video", "state")
        }
        if "pointcloud" in self.configs:
            frame["pointcloud"] = {
                "scene": scene_frame(observation, self.sensor_contract, timestamp)
            }
        if "tactile" in self.configs:
            frame["tactile"] = {"rgb": tactile_frame(observation, self.sensor_contract)}
        self.frames.append(frame)
        self.timestamp = timestamp
        result: dict[str, dict[str, Any]] = {}
        for name, config in self.configs.items():
            if name in {"action", "language"}:
                continue
            deltas = field(config, "delta_indices")
            result[name] = {}
            for key in field(config, "modality_keys"):
                values = [
                    self.frames[max(0, len(self.frames) - 1 + offset)][name][key]
                    for offset in deltas
                ]
                result[name][key] = np.ascontiguousarray(np.stack(values)[None])
        result["language"] = {field(self.configs["language"], "modality_keys")[0]: [[instruction]]}
        if "pointcloud" in result and len(field(self.configs["pointcloud"], "delta_indices")) > 1:
            result["temporal_valid"] = np.asarray(
                [
                    [
                        len(self.frames) - 1 + delta >= 0
                        for delta in field(self.configs["pointcloud"], "delta_indices")
                    ]
                ],
                dtype=np.bool_,
            )
        for name, values in result.items():
            if name == "temporal_valid":
                continue
            for key, value in values.items():
                if isinstance(value, np.ndarray) and (
                    value.shape[0] != 1 or not np.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid {name}.{key}: {value.shape} {value.dtype}.")
        return result
