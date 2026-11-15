"""CPU-only contracts for unified RGB/WAH and deployment orchestration."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "examples" / "UniVTAC"))
sys.path.insert(0, str(ROOT / "scripts"))

from gr00t.policy.gr00t_policy import validate_wah_restoration  # noqa: E402
from univtac_deployment import (  # noqa: E402
    build_rollout_command,
    check_port,
    print_rollout_summary,
    probe,
    wait_ready,
)
from univtac_rollout.checkpoint import checkpoint_identity  # noqa: E402
from univtac_rollout.multimodal import (  # noqa: E402
    ObservationHistory,
    add_scene_calibration,
    detect_policy,
    load_sensor_contract,
    scene_frame,
    tactile_frame,
    validate_contract,
)
from univtac_rollout.policy import Gr00tUniVTACPolicy  # noqa: E402
from univtac_rollout.sensors import UniVTACSensorAdapter  # noqa: E402


def configs(temporal: bool = False) -> dict:
    result = {
        "video": {"delta_indices": [0], "modality_keys": ["head", "wrist"]},
        "state": {
            "delta_indices": [-4, -3, -2, -1, 0] if temporal else [0],
            "modality_keys": ["joint"],
        },
        "action": {"delta_indices": list(range(16)), "modality_keys": ["joint"]},
        "language": {"delta_indices": [0], "modality_keys": ["annotation.human.task_description"]},
    }
    if temporal:
        result["pointcloud"] = {"delta_indices": [-4, -3, -2, -1, 0], "modality_keys": ["scene"]}
        result["tactile"] = {"delta_indices": [-5, -4, -3, -2, -1, 0], "modality_keys": ["rgb"]}
    return result


def observation(value: int = 0) -> dict:
    return {
        "observation": {
            "head": {"rgb": np.full((2, 2, 3), value, dtype=np.uint8)},
            "wrist": {"rgb": np.full((2, 2, 3), value + 1, dtype=np.uint8)},
            "pointcloud": {
                "scene": np.full((4, 6), value, dtype=np.float32),
                "coordinate_frame": "robot_base",
                "xyz_units": "m",
                "preprocessing": {"num_points": 4, "voxel_size_m": 0.01, "sampling_seed": 0},
            },
        },
        "embodiment": {"joint": np.full(8, value, dtype=np.float32)},
        "tactile": {"rgb_marker": np.full((2, 2, 3), value, dtype=np.uint8)},
    }


def contract() -> dict:
    return {
        "pointcloud": {
            "scene": {
                "num_points": 4,
                "feature_dim": 6,
                "voxel_size_m": 0.01,
                "coordinate_frame": "robot_base",
                "sampling_seed": 0,
            }
        },
        "tactile": {"rgb": {"source_key": "tactile/rgb_marker"}},
    }


def fake_checkpoint(path: Path, weight: bytes = b"weights") -> Path:
    path.mkdir()
    (path / "config.json").write_text("{}")
    (path / "processor_config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(weight)
    return path


def test_rgb_baseline_contract_and_observation() -> None:
    assert validate_contract(configs(), 1, "auto") == "rgb"
    frame = ObservationHistory(configs(), None).append(observation(), "lift can", 0)
    assert frame["video"]["head"].shape == (1, 1, 2, 2, 3)
    assert frame["state"]["joint"].shape == (1, 1, 8)


def test_wah_detection_uses_each_checkpoint_sensor_modality() -> None:
    assert detect_policy(configs(True)) == "wah"
    with pytest.raises(ValueError, match="requires wah"):
        validate_contract(configs(True), 1, "rgb")
    tactile_only = configs(True)
    del tactile_only["pointcloud"]
    assert validate_contract(tactile_only, 1, "auto") == "wah"
    point_only = configs(True)
    del point_only["tactile"]
    assert validate_contract(point_only, 1, "auto") == "wah"


def test_history_padding_reset_and_validity() -> None:
    history = ObservationHistory(configs(True), contract())
    first = history.append(observation(1), "lift can", 0)
    assert first["state"]["joint"].shape == (1, 5, 8)
    assert first["tactile"]["rgb"].shape == (1, 6, 2, 2, 3)
    assert first["temporal_valid"].tolist() == [[False, False, False, False, True]]
    history.append(observation(2), "lift can", 1)
    history.reset()
    next_episode = history.append(observation(9), "lift can", 0)
    assert np.all(next_episode["state"]["joint"] == 9)
    with pytest.raises(ValueError, match="not after"):
        history.append(observation(), "lift can", 0)


def test_scene_projection_uses_robot_base_extrinsics() -> None:
    obs = observation()
    obs["observation"].pop("pointcloud")
    for name in ("head", "wrist"):
        sensor = obs["observation"][name]
        sensor.update(
            depth=np.ones((2, 2), dtype=np.float32),
            intrinsics=np.array([1, 1, 0, 0]),
            depth_scale=1.0,
            T_base_camera=np.eye(4),
        )
    obs["observation"]["wrist"]["T_base_camera"][0, 3] = 2.0
    points = scene_frame(obs, contract(), 0)
    assert points.shape == (4, 6)
    assert np.any(points[:, 0] >= 2)
    assert np.all(points[:, 3:] <= 255)


def test_missing_scene_sensor_rejected() -> None:
    obs = observation()
    obs["observation"].pop("pointcloud")
    with pytest.raises(ValueError, match="missing.*depth"):
        scene_frame(obs, contract(), 0)


def test_prebuilt_scene_requires_frame_and_metric_units() -> None:
    obs = observation()
    del obs["observation"]["pointcloud"]["xyz_units"]
    with pytest.raises(ValueError, match="xyz_units='m'"):
        scene_frame(obs, contract(), 0)


def test_tactile_source_and_uint8_dtype() -> None:
    image = tactile_frame(observation(), contract())
    assert image.shape == (2, 2, 3) and image.dtype == np.uint8
    wrong = contract()
    wrong["tactile"]["rgb"]["source_key"] = "tactile/rgb"
    with pytest.raises(ValueError, match="trained from tactile/rgb"):
        tactile_frame(observation(), wrong)


def _gsmini_observation(image: np.ndarray) -> tuple[dict, dict]:
    obs = observation()
    obs["tactile"] = {"left_gsmini": {"rgb": image}}
    sensor_contract = contract()
    sensor_contract["tactile"]["rgb"]["source_key"] = "tactile/left_gsmini/rgb"
    return obs, sensor_contract


def test_tactile_float32_normalized_is_scaled_and_rounded() -> None:
    source = np.asarray([[[0.0, 0.5, 1.0], [0.25, 0.75, 0.1]]], dtype=np.float32)
    obs, sensor_contract = _gsmini_observation(source)
    converted = tactile_frame(obs, sensor_contract)
    expected = np.rint(source * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(converted, expected)


def test_tactile_float32_byte_range_is_rounded_without_rescaling() -> None:
    source = np.asarray([[[0.0, 12.4, 254.6], [1.4, 127.5, 255.0]]], dtype=np.float32)
    obs, sensor_contract = _gsmini_observation(source)
    converted = tactile_frame(obs, sensor_contract)
    np.testing.assert_array_equal(converted, np.rint(source).astype(np.uint8))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_tactile_rejects_nan_and_inf(bad_value: float) -> None:
    source = np.zeros((2, 2, 3), dtype=np.float32)
    source[0, 0, 0] = bad_value
    obs, sensor_contract = _gsmini_observation(source)
    with pytest.raises(ValueError, match="NaN or Inf"):
        tactile_frame(obs, sensor_contract)


@pytest.mark.parametrize("bad_value", [-0.01, 255.01])
def test_tactile_rejects_out_of_range_values(bad_value: float) -> None:
    source = np.zeros((2, 2, 3), dtype=np.float32)
    source[0, 0, 0] = bad_value
    obs, sensor_contract = _gsmini_observation(source)
    with pytest.raises(ValueError, match=r"\[0,1\] or \[0,255\]"):
        tactile_frame(obs, sensor_contract)


@pytest.mark.parametrize("shape", [(2, 2), (2, 2, 1), (2, 2, 2, 3)])
def test_tactile_rejects_wrong_shape(shape: tuple[int, ...]) -> None:
    obs, sensor_contract = _gsmini_observation(np.zeros(shape, dtype=np.float32))
    with pytest.raises(ValueError, match=r"shape \[H,W,3\]"):
        tactile_frame(obs, sensor_contract)


def test_scene_calibration_json_is_validated_without_static_wrist(tmp_path: Path) -> None:
    import json

    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "cameras": {
                    "head": {
                        "intrinsics": [2.0, 2.0, 1.0, 1.0],
                        "depth_scale": 1.0,
                        "T_base_camera": np.eye(4).tolist(),
                    },
                    "wrist": {"T_ee_camera": np.eye(4).tolist()},
                }
            }
        )
    )
    calibrated = add_scene_calibration(contract(), path)
    assert calibrated["calibration"]["cameras"]["head"]["depth_scale"] == 1.0

    path.write_text(json.dumps({"cameras": {"wrist": {"T_base_camera": np.eye(4).tolist()}}}))
    with pytest.raises(ValueError, match="Static wrist"):
        add_scene_calibration(contract(), path)


@pytest.mark.parametrize("require_tactile", [False, True])
def test_live_univtac_sensor_adapter_uses_per_frame_optical_camera_poses(
    monkeypatch, require_tactile
) -> None:
    class CameraData:
        def __init__(self, position):
            self.intrinsic_matrices = np.asarray(
                [[[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]]
            )
            self.pos_w = np.asarray([position], dtype=np.float64)
            self.quat_w_ros = np.asarray([[1.0, 0.0, 0.0, 0.0]])

    class Camera:
        def __init__(self, name, position):
            self.cfg = SimpleNamespace(prim_path=f"/World/{name}")
            self.data = CameraData(position)

    root_data = SimpleNamespace(
        root_link_pos_w=np.asarray([[1.0, 0.0, 0.0]]),
        root_link_quat_w=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )
    task = SimpleNamespace(
        cfg=SimpleNamespace(tactile_sensor_type="gsmini"),
        _robot_manager=SimpleNamespace(robot=SimpleNamespace(data=root_data)),
        _camera_manager=SimpleNamespace(
            cameras={
                "head": Camera("head", [2.0, 0.0, 0.0]),
                "wrist": Camera("wrist", [1.0, 1.0, 0.0]),
            }
        ),
        _tactile_manager=SimpleNamespace(
            tactiles={"left_tactile": object(), "right_tactile": object()}
        ),
    )
    sensor_contract = contract()
    sensor_contract["tactile"]["rgb"]["source_key"] = "tactile/left_gsmini/rgb"
    diagnostic_messages = []
    monkeypatch.setattr(
        "univtac_rollout.sensors.logger.warning",
        lambda message, *args: diagnostic_messages.append(message % args),
    )
    monkeypatch.setattr("univtac_rollout.sensors._isaac_depth_scale_to_meters", lambda: 1.0)
    if not require_tactile:
        del task._tactile_manager
        sensor_contract.pop("tactile")
    adapter = UniVTACSensorAdapter(task, sensor_contract, require_tactile=require_tactile)

    def live_observation(value=0.5):
        return {
            "observation": {
                name: {
                    "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                    "depth": np.ones((2, 2, 1), dtype=np.float32),
                }
                for name in ("head", "wrist")
            },
            "tactile": {
                "left_tactile": {"rgb": np.full((2, 2, 3), value, dtype=np.float32)},
                "right_tactile": {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)},
            },
        }

    if not require_tactile:
        obs = live_observation()
        obs.pop("tactile")
        attached = adapter.attach(obs)
        assert "tactile" not in attached
        assert not any(key.startswith("tactile") for key in adapter.metadata)
        assert scene_frame(attached, sensor_contract, 0).shape == (4, 6)
        del obs["observation"]["head"]["depth"]
        with pytest.raises(ValueError, match="aligned RGB and depth"):
            adapter.attach(obs)
        return

    first = adapter.attach(live_observation())
    np.testing.assert_allclose(first["observation"]["head"]["T_base_camera"][:3, 3], [1, 0, 0])
    np.testing.assert_allclose(first["observation"]["wrist"]["T_base_camera"][:3, 3], [0, 1, 0])
    assert first["observation"]["wrist"]["depth_scale"] == 1.0
    assert np.all(tactile_frame(first, sensor_contract) == 128)

    task._camera_manager.cameras["wrist"].data.pos_w[0] = [1.0, 2.0, 0.0]
    second = adapter.attach(live_observation())
    np.testing.assert_allclose(second["observation"]["wrist"]["T_base_camera"][:3, 3], [0, 2, 0])
    assert len(diagnostic_messages) == 1
    assert "shape=(2, 2, 3) dtype=float32 min=0.5 max=0.5" in diagnostic_messages[0]

    task._camera_manager.cameras["head"].data.pos_w[0, 0] += 0.1
    with pytest.raises(RuntimeError, match="fixed head camera moved"):
        adapter.attach(live_observation())


def test_tactile_only_sensor_adapter_does_not_initialize_rgbd(monkeypatch) -> None:
    task = SimpleNamespace(
        cfg=SimpleNamespace(tactile_sensor_type="gsmini"),
        _tactile_manager=SimpleNamespace(tactiles={"left_tactile": object()}),
    )
    sensor_contract = {
        "tactile": {"rgb": {"source_key": "tactile/left_gsmini/rgb_marker"}}
    }
    monkeypatch.setattr(
        "univtac_rollout.sensors._isaac_depth_scale_to_meters",
        lambda: pytest.fail("tactile-only mode must not initialize RGB-D calibration"),
    )
    adapter = UniVTACSensorAdapter(
        task,
        sensor_contract,
        require_pointcloud=False,
        require_tactile=True,
    )
    observation = {
        "tactile": {
            "left_tactile": {
                "rgb_marker": np.full((2, 2, 3), 7, dtype=np.uint8),
            }
        }
    }
    attached = adapter.attach(observation)
    assert adapter.cameras == {}
    assert not any(key.startswith("depth") for key in adapter.metadata)
    assert np.all(tactile_frame(attached, sensor_contract) == 7)


def test_checkpoint_identity_is_path_independent_and_mismatch(monkeypatch, tmp_path: Path) -> None:
    first = fake_checkpoint(tmp_path / "a")
    second = fake_checkpoint(tmp_path / "b", b"different")
    clone = fake_checkpoint(tmp_path / "c")
    assert checkpoint_identity(first) == checkpoint_identity(clone)
    assert checkpoint_identity(first) != checkpoint_identity(second)
    (clone / "statistics.json").write_text('{"joint": 1}')
    assert checkpoint_identity(first) != checkpoint_identity(clone)

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def ping(self):
            return True

        def get_model_info(self):
            return {"checkpoint_identity": checkpoint_identity(second)}

        def get_modality_config(self):
            return configs()

        def close(self):
            pass

    monkeypatch.setattr("univtac_rollout.policy.MinimalPolicyClient", Client)
    with pytest.raises(ValueError, match="Checkpoint identity mismatch"):
        Gr00tUniVTACPolicy("localhost", 5555, checkpoint=str(first))


def test_wah_policy_sends_real_multimodal_history(monkeypatch, tmp_path: Path) -> None:
    import json

    sensor_file = tmp_path / "modality.json"
    sensor_file.write_text(json.dumps(contract()))
    expected = "sha256:" + "a" * 64

    class Client:
        sent = None

        def __init__(self, *args, **kwargs):
            pass

        def ping(self):
            return True

        def get_model_info(self):
            return {
                "checkpoint_identity": expected,
                "action_horizon": 16,
                "dit_type": "sidecar_conditioned_dit",
                "sidecar_type": "temporal",
                "dit_training_mode": "lora",
                "dit_lora_enabled": True,
            }

        def get_modality_config(self):
            return configs(True)

        def reset(self):
            pass

        def close(self):
            pass

        def get_action(self, data):
            self.sent = data
            return ({"joint": np.zeros((1, 16, 8), dtype=np.float32)}, {})

    monkeypatch.setattr("univtac_rollout.policy.MinimalPolicyClient", Client)
    with Gr00tUniVTACPolicy(
        "localhost",
        5555,
        checkpoint="/remote/checkpoint",
        expected_checkpoint_identity=expected,
        sensor_contract=str(sensor_file),
    ) as policy:
        policy.reset()
        action, formatted = policy.predict_chunk(observation(3), "lift can", timestamp=0)
        assert action.shape == (1, 8)
        assert policy.client.sent["pointcloud"]["scene"].shape == (1, 5, 4, 6)
        assert policy.client.sent["tactile"]["rgb"].shape == (1, 6, 2, 2, 3)
        assert np.all(formatted["pointcloud"]["scene"] == 3)
        policy.record_observation(observation(4), "lift can", 1)
        _, formatted = policy.predict_chunk(observation(4), "lift can", timestamp=1)
        assert formatted["state"]["joint"][0, -1, 0] == 4
        policy.reset()
        _, formatted = policy.predict_chunk(observation(9), "lift can", timestamp=0)
        assert np.all(formatted["state"]["joint"] == 9)


def test_pointcloud_only_checkpoint_loads_and_sends_no_tactile(monkeypatch, tmp_path: Path) -> None:
    import json

    sensor_contract = contract()
    sensor_contract.pop("tactile")
    sensor_file = tmp_path / "modality.json"
    sensor_file.write_text(json.dumps(sensor_contract))
    expected = "sha256:" + "b" * 64
    point_only_configs = configs()
    point_only_configs["pointcloud"] = {
        "delta_indices": [0],
        "modality_keys": ["scene"],
    }

    class Client:
        sent = None

        def __init__(self, *args, **kwargs):
            pass

        def ping(self):
            return True

        def get_model_info(self):
            return {"checkpoint_identity": expected, "action_horizon": 16}

        def get_modality_config(self):
            return point_only_configs

        def reset(self):
            pass

        def close(self):
            pass

        def get_action(self, data):
            self.sent = data
            return ({"joint": np.zeros((1, 16, 8), dtype=np.float32)}, {})

    monkeypatch.setattr("univtac_rollout.policy.MinimalPolicyClient", Client)
    obs = observation(5)
    obs.pop("tactile")
    with Gr00tUniVTACPolicy(
        "localhost",
        5555,
        expected_checkpoint_identity=expected,
        sensor_contract=str(sensor_file),
    ) as policy:
        action, formatted = policy.predict_chunk(obs, "lift can", timestamp=0)

    assert action.shape == (1, 8)
    assert formatted["pointcloud"]["scene"].shape == (1, 1, 4, 6)
    assert "tactile" not in formatted
    assert "tactile" not in policy.client.sent


def test_deployment_probe_reports_checkpoint_sensor_modalities(monkeypatch, tmp_path) -> None:
    import univtac_deployment as deployment

    expected = "sha256:" + "c" * 64
    point_only_configs = configs()
    point_only_configs["pointcloud"] = {
        "delta_indices": [0],
        "modality_keys": ["scene"],
    }

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def ping(self):
            return True

        def get_model_info(self):
            return {"checkpoint_identity": expected}

        def get_modality_config(self):
            return point_only_configs

    monkeypatch.setattr(deployment, "MinimalPolicyClient", Client)
    info = probe("localhost", 5555, "/remote/checkpoint", "auto", 1, expected)
    assert info["policy_type"] == "wah"
    assert info["sensor_modalities"] == ["pointcloud"]
    point_contract = contract()
    point_contract.pop("tactile")
    path = tmp_path / "point-only-modality.json"
    path.write_text(__import__("json").dumps(point_contract))
    assert deployment._load_checkpoint_sensor_contract(info, path) == point_contract


def test_sidecar_lora_missing_weights_rejected() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(dit_type="sidecar_conditioned_dit", dit_training_mode="lora"),
        state_dict=lambda: {
            "action_head.model.sidecar_stages.0.weight": 1,
            "action_head.model.sidecar_pool.weight": 1,
            "action_head.model.transformer_blocks.0.lora_A": 1,
        },
    )
    validate_wah_restoration(model, {"missing_keys": []})
    with pytest.raises(RuntimeError, match="did not restore"):
        validate_wah_restoration(model, {"missing_keys": ["action_head.model.sidecar_pool.weight"]})
    with pytest.raises(RuntimeError, match="did not restore"):
        validate_wah_restoration(
            model, {"missing_keys": ["action_head.model.transformer_blocks.0.lora_A"]}
        )


def test_split_local_rollout_command_parity_and_gpu_argument(tmp_path: Path) -> None:
    base = dict(
        checkpoint="checkpoint",
        policy="auto",
        server_host="127.0.0.1",
        server_port=5555,
        execution_horizon=1,
        sim_device="cuda:1",
        sensor_contract=None,
        scene_calibration=None,
    )
    split = SimpleNamespace(**base, deployment="split")
    local = SimpleNamespace(**base, deployment="local")
    command = build_rollout_command(split, tmp_path, ["--mode", "smoke"])
    assert command == build_rollout_command(local, tmp_path, ["--mode", "smoke"])
    assert command[command.index("--device") + 1] == "cuda:1"


def test_deployment_terminal_summary_reports_all_outcomes(tmp_path: Path, capsys) -> None:
    import json

    output = tmp_path / "output"
    logs = output / "logs"
    logs.mkdir(parents=True)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "total_successes": 2,
                    "total_failures": 3,
                    "total_errors": 1,
                }
            }
        )
    )
    summary = print_rollout_summary(output, logs)
    assert summary["total_errors"] == 1
    terminal = capsys.readouterr().out
    assert "success=2 failure=3 error=1" in terminal
    assert f"Logs: {logs}" in terminal


def test_local_port_conflict_does_not_kill_service(monkeypatch) -> None:
    import univtac_deployment as deployment

    calls = []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append("closed")

        def connect_ex(self, address):
            return 0

    monkeypatch.setattr(deployment.socket, "socket", Socket)
    with pytest.raises(RuntimeError, match="occupied"):
        check_port("127.0.0.1", 5555)
    assert calls == ["closed"]


def test_server_readiness_timeout_and_crash(monkeypatch) -> None:
    import univtac_deployment as deployment

    args = SimpleNamespace(
        startup_timeout=0.01,
        retry_interval=0.001,
        server_host="127.0.0.1",
        server_port=1,
        checkpoint="none",
        policy="auto",
        execution_horizon=1,
    )
    monkeypatch.setattr(
        deployment, "probe", lambda *a: (_ for _ in ()).throw(ConnectionError("offline"))
    )
    with pytest.raises(TimeoutError, match="readiness timed out"):
        wait_ready(args, None)
    crashed = SimpleNamespace(poll=lambda: 9, returncode=9)
    with pytest.raises(RuntimeError, match="exited before readiness"):
        wait_ready(args, crashed)


def test_local_startup_and_worker_crash_preserve_metadata(monkeypatch, tmp_path: Path) -> None:
    import univtac_deployment as deployment

    checkpoint = fake_checkpoint(tmp_path / "checkpoint")
    (tmp_path / "envs").mkdir()
    (tmp_path / "envs" / "_base_task.py").touch()
    output = tmp_path / "results"
    args = SimpleNamespace(
        deployment="local",
        checkpoint=str(checkpoint),
        policy="auto",
        server_host="127.0.0.1",
        server_port=5555,
        model_python=ROOT / ".venv" / "bin" / "python",
        model_device="cuda:2",
        sim_device="cuda:1",
        model_visible_devices="2",
        sim_visible_devices="1",
        startup_timeout=1,
        retry_interval=0.01,
        output_dir=output,
        sensor_contract=None,
        scene_calibration=None,
        execution_horizon=1,
    )
    monkeypatch.setenv("UNIVTAC_EXPECTED_PYTHON", "/usr/bin/python3")
    monkeypatch.setenv("UNIVTAC_ROOT", str(tmp_path))
    monkeypatch.setattr(deployment, "validate_python", lambda *a: None)
    monkeypatch.setattr(deployment, "check_port", lambda *a: None)
    monkeypatch.setattr(deployment, "gpu_memory", lambda: [])
    monkeypatch.setattr(
        deployment,
        "wait_ready",
        lambda *a: {"checkpoint_identity": checkpoint_identity(checkpoint), "policy_type": "rgb"},
    )
    monkeypatch.setattr(deployment, "git_commit", lambda: "abc123")
    stopped = []
    monkeypatch.setattr(deployment, "stop_owned", lambda process: stopped.append(process))
    spawned = []

    class Process:
        returncode = None

        def __init__(self, command, **kwargs):
            self.command = command
            self.is_server = len(spawned) == 0
            spawned.append(self)

        def poll(self):
            return None if self.is_server else 7

        def wait(self):
            self.returncode = 7
            return 7

    monkeypatch.setattr(deployment.subprocess, "Popen", Process)
    assert deployment.run(args, ["--mode", "smoke"]) == 7
    assert "--device" in spawned[0].command
    assert spawned[0].command[spawned[0].command.index("--device") + 1] == "cuda:2"
    assert spawned[1].command[spawned[1].command.index("--device") + 1] == "cuda:1"
    assert spawned[1].command[-2:] == ["--mode", "smoke"]
    assert spawned[0] in stopped and spawned[1] in stopped
    metadata = __import__("json").loads((output / "runtime_metadata.json").read_text())
    assert metadata["deployment_mode"] == "local"
    assert metadata["model_visible_devices"] == "2"


def test_owned_process_cleanup_uses_only_its_process_group(monkeypatch) -> None:
    import univtac_deployment as deployment

    signals = []
    monkeypatch.setattr(deployment.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    class Process:
        pid = 1234

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    deployment.stop_owned(Process())
    assert signals == [(1234, deployment.signal.SIGTERM)]


@pytest.mark.parametrize("signum", [__import__("signal").SIGINT, __import__("signal").SIGTERM])
def test_signal_cleanup_stops_owned_server_and_rollout(monkeypatch, tmp_path: Path, signum) -> None:
    import univtac_deployment as deployment

    checkpoint = fake_checkpoint(tmp_path / "checkpoint")
    (tmp_path / "envs").mkdir()
    (tmp_path / "envs" / "_base_task.py").touch()
    args = SimpleNamespace(
        deployment="local",
        checkpoint=str(checkpoint),
        policy="auto",
        server_host="127.0.0.1",
        server_port=5555,
        model_python=ROOT / ".venv" / "bin" / "python",
        model_device="cuda:0",
        sim_device="cuda:0",
        model_visible_devices=None,
        sim_visible_devices=None,
        startup_timeout=1,
        retry_interval=0.01,
        output_dir=tmp_path / "results",
        sensor_contract=None,
        scene_calibration=None,
        execution_horizon=1,
    )
    monkeypatch.setenv("UNIVTAC_EXPECTED_PYTHON", "/usr/bin/python3")
    monkeypatch.setenv("UNIVTAC_ROOT", str(tmp_path))
    monkeypatch.setattr(deployment, "validate_python", lambda *a: None)
    monkeypatch.setattr(deployment, "check_port", lambda *a: None)
    monkeypatch.setattr(deployment, "gpu_memory", lambda: [])
    monkeypatch.setattr(
        deployment,
        "wait_ready",
        lambda *a: {"checkpoint_identity": checkpoint_identity(checkpoint), "policy_type": "rgb"},
    )
    monkeypatch.setattr(deployment, "git_commit", lambda: None)
    spawned = []
    stopped = []

    class Process:
        def __init__(self, command, **kwargs):
            self.command = command
            self.returncode = None
            spawned.append(self)

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    def stop(process):
        if process is not None:
            stopped.append(process)
            process.returncode = -int(signum)

    fired = False

    def fire_signal(seconds):
        nonlocal fired
        if not fired:
            fired = True
            deployment.signal.getsignal(signum)(signum, None)

    monkeypatch.setattr(deployment.subprocess, "Popen", Process)
    monkeypatch.setattr(deployment, "stop_owned", stop)
    monkeypatch.setattr(deployment.time, "sleep", fire_signal)
    assert deployment.run(args, []) == 130
    assert spawned[0] in stopped and spawned[1] in stopped
    assert (args.output_dir / "runtime_metadata.json").is_file()


@pytest.mark.parametrize("mode", ["rgb", "point", "tactile", "point_tactile"])
def test_active_modality_contract_and_history(mode, tmp_path):
    import json

    active = configs()
    sensor_contract = contract()
    obs = observation()
    needs_pointcloud = mode in {"point", "point_tactile"}
    needs_tactile = mode in {"tactile", "point_tactile"}
    if needs_pointcloud:
        active["pointcloud"] = {"delta_indices": [0], "modality_keys": ["scene"]}
    else:
        sensor_contract.pop("pointcloud")
    if needs_tactile:
        active["tactile"] = {"delta_indices": [0], "modality_keys": ["rgb"]}
    else:
        sensor_contract.pop("tactile")
        obs.pop("tactile")
    assert validate_contract(active, 1, "auto") == ("rgb" if mode == "rgb" else "wah")
    path = tmp_path / "modality.json"
    path.write_text(json.dumps(sensor_contract))
    loaded = (
        load_sensor_contract(
            path,
            require_pointcloud=needs_pointcloud,
            require_tactile=needs_tactile,
        )
        if mode != "rgb"
        else None
    )
    result = ObservationHistory(active, loaded).append(obs, "lift can", 0)
    assert set(result) == set(active) - {"action"}
    if needs_pointcloud:
        assert result["pointcloud"]["scene"].shape == (1, 1, 4, 6)
        assert "temporal_valid" not in result
    else:
        assert "pointcloud" not in result
    if needs_tactile:
        assert result["tactile"]["rgb"].dtype == np.uint8
        obs.pop("tactile")
        with pytest.raises(ValueError, match="trained from"):
            ObservationHistory(active, loaded).append(obs, "task", 0)
    else:
        assert "tactile" not in result
    if needs_pointcloud:
        obs["observation"].pop("pointcloud")
        with pytest.raises(ValueError, match="calibrated head and wrist RGB-D"):
            ObservationHistory(active, loaded).append(obs, "task", 0)
        sensor_contract.pop("pointcloud")
        path.write_text(json.dumps(sensor_contract))
        with pytest.raises(ValueError, match="pointcloud.scene"):
            load_sensor_contract(path, require_tactile=needs_tactile)


@pytest.mark.parametrize("mode", ["rgb", "point", "tactile", "point_tactile"])
def test_worker_requests_policy_and_renderer_tactile_streams(mode):
    from run_closed_loop_worker import _configure_tactile_observations

    active = configs()
    if mode in {"point", "point_tactile"}:
        active["pointcloud"] = {"delta_indices": [0], "modality_keys": ["scene"]}
    sensor_contract = None
    if mode in {"tactile", "point_tactile"}:
        active["tactile"] = {"delta_indices": [0], "modality_keys": ["rgb"]}
        sensor_contract = contract()
        sensor_contract["tactile"]["rgb"]["source_key"] = "tactile/left_gsmini/rgb"
    streams = {"camera": ["rgb", "depth"], "tactile": ["rgb_marker", "depth"]}
    _configure_tactile_observations(streams, active, sensor_contract)
    assert streams["camera"] == ["rgb", "depth"]
    if mode in {"tactile", "point_tactile"}:
        assert streams["tactile"] == ["rgb", "rgb_marker"]
    else:
        assert streams["tactile"] == ["rgb_marker"]
