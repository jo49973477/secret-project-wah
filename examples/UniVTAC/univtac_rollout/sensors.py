"""Runtime calibration adapter for the unmodified UniVTAC sensor objects."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scene_pointcloud import validate_intrinsics, validate_transform


OPTICAL_FRAME_CONVENTION = "ros_optical_x_right_y_down_z_forward"
UNIVTAC_TACTILE_RGB_ENCODING = "univtac_tactile_rgb"
UINT8_RGB_ENCODING = "uint8_0_255"

logger = logging.getLogger(__name__)


def _to_numpy(value: Any, *, name: str) -> np.ndarray:
    """Copy a simulator value to CPU NumPy without assuming its tensor backend."""
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive context for simulator APIs
        raise TypeError(f"{name} cannot be converted to NumPy: {type(value)}") from exc


def pose_wxyz_matrix(position: Any, quaternion: Any, *, source: str) -> np.ndarray:
    """Build ``T_world_frame`` from Isaac Lab position and WXYZ quaternion data."""
    position_array = _to_numpy(position, name=f"{source} position").reshape(-1)
    quaternion_array = _to_numpy(quaternion, name=f"{source} quaternion").reshape(-1)
    if position_array.shape != (3,) or quaternion_array.shape != (4,):
        raise ValueError(
            f"{source} pose must contain position [3] and WXYZ quaternion [4]; got "
            f"{position_array.shape} and {quaternion_array.shape}."
        )
    if not np.isfinite(position_array).all() or not np.isfinite(quaternion_array).all():
        raise ValueError(f"{source} pose contains NaN or Inf.")
    norm = float(np.linalg.norm(quaternion_array))
    if not np.isclose(norm, 1.0, atol=1e-4):
        raise ValueError(f"{source} quaternion is not normalized; norm={norm:.8g}.")
    w, x, y, z = quaternion_array / norm
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform[:3, 3] = position_array
    return validate_transform(transform, source=source)


def _first(value: Any, *, name: str) -> np.ndarray:
    array = _to_numpy(value, name=name)
    if array.shape[0] != 1:
        raise ValueError(f"WAH rollout requires exactly one {name}; got shape {array.shape}.")
    return array[0]


def _isaac_depth_scale_to_meters() -> float:
    """Verify the live stage is metric before declaring Isaac depth to be meters."""
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Cannot read RGB-D units because the Isaac USD stage is unavailable.")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not np.isclose(meters_per_unit, 1.0, atol=1e-9):
        raise RuntimeError(
            "UniVTAC RGB-D depth is only validated for an Isaac stage using meters; "
            f"UsdGeom.GetStageMetersPerUnit returned {meters_per_unit}. Supply verified "
            "metric calibration instead of guessing a depth scale."
        )
    # Isaac Lab's `depth` output is distance_to_image_plane in the metric stage.
    return 1.0


def _tactile_runtime_mapping(task: Any, source_key: str) -> tuple[str, str]:
    parts = source_key.split("/")
    if len(parts) == 2:
        raise ValueError(
            f"Sensor contract {source_key!r} does not identify which UniVTAC tactile sensor "
            "produced the training stream. Use tactile/<sensor>/<rgb|rgb_marker>."
        )
    if len(parts) != 3 or parts[0] != "tactile":
        raise ValueError(f"Unsupported tactile source_key {source_key!r}.")
    contract_sensor, stream = parts[1:]
    if stream not in {"rgb", "rgb_marker"}:
        raise ValueError(f"Unsupported tactile stream in source_key {source_key!r}.")

    tactile_manager = getattr(task, "_tactile_manager", None)
    runtime_sensors = getattr(tactile_manager, "tactiles", None)
    if not isinstance(runtime_sensors, dict):
        raise RuntimeError("UniVTAC task does not expose _tactile_manager.tactiles.")
    if contract_sensor in runtime_sensors:
        return contract_sensor, stream

    sensor_type = str(getattr(getattr(task, "cfg", None), "tactile_sensor_type", ""))
    candidates = []
    for side in ("left", "right"):
        if contract_sensor == f"{side}_{sensor_type}":
            candidates.append(f"{side}_tactile")
    matches = [name for name in candidates if name in runtime_sensors]
    if len(matches) != 1:
        raise ValueError(
            f"Training tactile source {source_key!r} does not match the live UniVTAC sensors "
            f"{sorted(runtime_sensors)} with tactile_sensor_type={sensor_type!r}. No tactile "
            "camera stream will be substituted."
        )
    return matches[0], stream


class UniVTACSensorAdapter:
    """Attach only the live sensor metadata and streams required by the checkpoint."""

    def __init__(
        self,
        task: Any,
        sensor_contract: dict[str, Any],
        *,
        require_pointcloud: bool = True,
        require_tactile: bool = True,
    ):
        if not require_pointcloud and not require_tactile:
            raise ValueError("UniVTACSensorAdapter needs an active sensor modality.")
        self.require_pointcloud = require_pointcloud
        self.require_tactile = require_tactile
        self.task = task
        self.cameras: dict[str, Any] = {}
        self.depth_scale: float | None = None
        if require_pointcloud:
            manager = getattr(task, "_camera_manager", None)
            cameras = getattr(manager, "cameras", None)
            if not isinstance(cameras, dict) or set(cameras) != {"head", "wrist"}:
                raise RuntimeError(
                    "PointCloud rollout needs live UniVTAC head and wrist camera objects; found "
                    f"{sorted(cameras) if isinstance(cameras, dict) else None}."
                )
            self.cameras = cameras
            self.depth_scale = _isaac_depth_scale_to_meters()
        if require_tactile:
            source_key = sensor_contract["tactile"]["rgb"]["source_key"]
            self.tactile_source_key = source_key
            self.tactile_runtime_sensor, self.tactile_stream = _tactile_runtime_mapping(
                task, source_key
            )
            self.tactile_encoding = (
                UNIVTAC_TACTILE_RGB_ENCODING if self.tactile_stream == "rgb" else UINT8_RGB_ENCODING
            )
        self._tactile_diagnostic_logged = False
        self._head_transform: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        if require_pointcloud:
            self.metadata.update(
                depth_units="m",
                depth_scale=self.depth_scale,
                depth_semantics="distance_to_image_plane",
                camera_frame=OPTICAL_FRAME_CONVENTION,
                base_frame_source="live articulation root-link pose",
                camera_pose_source="live TiledCamera.data.pos_w + quat_w_ros",
                camera_prim_paths={
                    name: str(getattr(getattr(camera, "cfg", None), "prim_path", ""))
                    for name, camera in self.cameras.items()
                },
            )

        if require_tactile:
            self.metadata.update(
                tactile_source_key=self.tactile_source_key,
                tactile_runtime_sensor=self.tactile_runtime_sensor,
                tactile_runtime_encoding=self.tactile_encoding,
            )

    def _base_transform(self) -> np.ndarray:
        manager = getattr(self.task, "_robot_manager", None)
        robot = getattr(manager, "robot", None)
        data = getattr(robot, "data", None)
        if (
            data is None
            or not hasattr(data, "root_link_pos_w")
            or not hasattr(data, "root_link_quat_w")
        ):
            raise RuntimeError(
                "UniVTAC robot does not expose its live articulation root-link pose."
            )
        return pose_wxyz_matrix(
            _first(data.root_link_pos_w, name="robot root-link positions"),
            _first(data.root_link_quat_w, name="robot root-link quaternions"),
            source="T_world_robot_base",
        )

    def _camera_calibration(self, name: str, transform_world_base: np.ndarray) -> dict[str, Any]:
        camera = self.cameras[name]
        data = getattr(camera, "data", None)
        if data is None:
            raise RuntimeError(f"UniVTAC {name} camera has no live CameraData.")
        intrinsics = validate_intrinsics(
            _first(data.intrinsic_matrices, name=f"{name} camera intrinsics"),
            source=f"live UniVTAC {name} CameraData.intrinsic_matrices",
        )
        transform_world_camera = pose_wxyz_matrix(
            _first(data.pos_w, name=f"{name} camera positions"),
            _first(data.quat_w_ros, name=f"{name} camera ROS optical quaternions"),
            source=f"T_world_{name}_camera_ros_optical",
        )
        transform_base_camera = validate_transform(
            np.linalg.inv(transform_world_base) @ transform_world_camera,
            source=f"live {name} T_base_camera",
        )
        if name == "head":
            if self._head_transform is None:
                self._head_transform = transform_base_camera.copy()
            elif not np.allclose(self._head_transform, transform_base_camera, atol=1e-5):
                raise RuntimeError(
                    "The configured fixed head camera moved relative to robot_base; refusing "
                    "to use inconsistent scene calibration."
                )
        return {
            "intrinsics": np.ascontiguousarray(intrinsics, dtype=np.float64),
            "depth_scale": self.depth_scale,
            "T_base_camera": np.ascontiguousarray(transform_base_camera, dtype=np.float64),
            "depth_units": "m",
            "depth_semantics": "distance_to_image_plane",
            "camera_frame": OPTICAL_FRAME_CONVENTION,
        }

    def attach(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.require_pointcloud:
            sensors = observation.get("observation")
            if not isinstance(sensors, dict):
                raise ValueError("UniVTAC observation is missing its observation sensor mapping.")
            transform_world_base = self._base_transform()
            calibrations = {}
            for name in ("head", "wrist"):
                sensor = sensors.get(name)
                if not isinstance(sensor, dict) or "rgb" not in sensor or "depth" not in sensor:
                    raise ValueError(
                        f"Live UniVTAC {name} observation must contain aligned RGB and depth."
                    )
                calibration = self._camera_calibration(name, transform_world_base)
                sensor.update(calibration)
                calibrations[name] = calibration
                rgb = _to_numpy(sensor["rgb"], name=f"{name} RGB")
                depth = _to_numpy(sensor["depth"], name=f"{name} depth")
                if depth.ndim == 3 and depth.shape[-1] == 1:
                    depth = depth[..., 0]
                if rgb.ndim != 3 or rgb.shape[-1] != 3 or depth.shape != rgb.shape[:2]:
                    raise ValueError(
                        f"Live {name} RGB-D is not pixel aligned: "
                        f"rgb={rgb.shape}, depth={depth.shape}."
                    )

            if "head_intrinsics" not in self.metadata:
                self.metadata.update(
                    {
                        "head_intrinsics": calibrations["head"]["intrinsics"].tolist(),
                        "head_T_base_camera": calibrations["head"]["T_base_camera"].tolist(),
                        "wrist_intrinsics": calibrations["wrist"]["intrinsics"].tolist(),
                    }
                )
            self.metadata["last_wrist_T_base_camera"] = calibrations["wrist"][
                "T_base_camera"
            ].tolist()

        if not self.require_tactile:
            return observation

        tactile = observation.get("tactile")
        runtime_sensor = (
            tactile.get(self.tactile_runtime_sensor) if isinstance(tactile, dict) else None
        )
        if not isinstance(runtime_sensor, dict) or self.tactile_stream not in runtime_sensor:
            available = sorted(runtime_sensor) if isinstance(runtime_sensor, dict) else []
            raise ValueError(
                f"Live UniVTAC tactile sensor {self.tactile_runtime_sensor!r} does not expose "
                f"the trained {self.tactile_stream!r} stream; available={available}."
            )
        tactile_image = runtime_sensor[self.tactile_stream]
        if not self._tactile_diagnostic_logged:
            raw_image = _to_numpy(tactile_image, name=self.tactile_source_key.replace("/", "."))
            try:
                raw_min = float(np.min(raw_image))
                raw_max = float(np.max(raw_image))
            except (TypeError, ValueError):
                raw_min = raw_max = float("nan")
            logger.warning(
                "UniVTAC tactile first observation: source_key=%s runtime_sensor=%s "
                "stream=%s shape=%s dtype=%s min=%.8g max=%.8g encoding=%s",
                self.tactile_source_key,
                self.tactile_runtime_sensor,
                self.tactile_stream,
                raw_image.shape,
                raw_image.dtype,
                raw_min,
                raw_max,
                self.tactile_encoding,
            )
            self._tactile_diagnostic_logged = True
        tactile["_wah_contract_source"] = {
            "source_key": self.tactile_source_key,
            "encoding": self.tactile_encoding,
            "image": tactile_image,
        }
        return observation
