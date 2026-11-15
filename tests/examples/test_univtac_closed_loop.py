# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest


UNIVTAC_EXAMPLE_ROOT = Path(__file__).parents[2] / "examples" / "UniVTAC"
sys.path.insert(0, str(UNIVTAC_EXAMPLE_ROOT))

import run_closed_loop_eval as closed_loop_eval  # noqa: E402
from run_closed_loop_eval import (  # noqa: E402
    _load_worker_result,
    _worker_failure_result,
    evaluation_exit_code,  # noqa: E402
)
from run_closed_loop_worker import (  # noqa: E402
    _apply_task_config,
    _config_profile,
    _configure_tactile_observations,
    _finish_video,
    _isaac_lab_release_matches,
    _validate_mp4,
    _validate_video_renderer_observation,
    _write_worker_results,
)
from univtac_rollout.client import CompatibleMsgSerializer  # noqa: E402
from univtac_rollout.core import (  # noqa: E402
    CANONICAL_TASKS,
    EpisodeResult,
    InvalidActionError,
    aggregate_results,
    classify_result,
    extract_action_chunk,
    format_univtac_observation,
    resolve_tasks,
    validate_action_step,
    video_output_path,
    write_results,
)
from univtac_rollout.policy import validate_rgb_checkpoint_contract  # noqa: E402


def _observation() -> dict:
    return {
        "observation": {
            "head": {"rgb": np.zeros((270, 480, 3), dtype=np.uint8)},
            "wrist": {"rgb": np.ones((270, 480, 3), dtype=np.uint8)},
        },
        "embodiment": {"joint": np.arange(9, dtype=np.float64)},
    }


def _episode(
    task: str,
    index: int,
    success: bool | None,
    reason: str,
    *,
    steps: int = 10,
) -> EpisodeResult:
    return EpisodeResult(
        task=task,
        seed=1_000_000 + index,
        episode_index=index,
        success=success,
        num_steps=steps,
        execution_horizon=1,
        checkpoint="checkpoint-2000",
        instruction="do the task",
        elapsed_seconds=1.5,
        termination_reason=reason,
        video_path=f"videos/{task}-{index}.mp4",
    )


def _modality_configs(action_horizon: int = 16) -> dict:
    return {
        "video": {"delta_indices": [0], "modality_keys": ["head", "wrist"]},
        "state": {"delta_indices": [0], "modality_keys": ["joint"]},
        "action": {
            "delta_indices": list(range(action_horizon)),
            "modality_keys": ["joint"],
        },
        "language": {
            "delta_indices": [0],
            "modality_keys": ["annotation.human.task_description"],
        },
    }


def test_observation_format_matches_rgb_checkpoint_contract() -> None:
    formatted = format_univtac_observation(_observation(), "lift the can")
    assert formatted["video"]["head"].shape == (1, 1, 270, 480, 3)
    assert formatted["video"]["wrist"].shape == (1, 1, 270, 480, 3)
    assert formatted["video"]["head"].dtype == np.uint8
    assert formatted["state"]["joint"].shape == (1, 1, 8)
    assert formatted["state"]["joint"].dtype == np.float32
    np.testing.assert_array_equal(formatted["state"]["joint"][0, 0], np.arange(8, dtype=np.float32))
    assert formatted["language"] == {"annotation.human.task_description": [["lift the can"]]}


def test_observation_format_rejects_wrong_rgb_dtype_and_short_joint_state() -> None:
    observation = _observation()
    observation["observation"]["head"]["rgb"] = np.zeros((2, 3, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="uint8"):
        format_univtac_observation(observation, "task")

    observation = _observation()
    observation["embodiment"]["joint"] = np.zeros(7, dtype=np.float32)
    with pytest.raises(ValueError, match="at least 8"):
        format_univtac_observation(observation, "task")


def test_action_extraction_accepts_one_batch_and_rejects_malformed_actions() -> None:
    chunk = np.arange(4 * 8, dtype=np.float32).reshape(1, 4, 8)
    extracted = extract_action_chunk([{"joint": chunk}, {}])
    assert extracted.shape == (4, 8)
    assert extracted.dtype == np.float32

    with pytest.raises(InvalidActionError, match="8 dimensions"):
        extract_action_chunk([{"joint": np.zeros((1, 4, 7), dtype=np.float32)}, {}])
    chunk[0, 2, 3] = np.nan
    with pytest.raises(InvalidActionError, match="NaN/Inf"):
        extract_action_chunk([{"joint": chunk}, {}])


def test_action_limit_validation_rejects_without_clipping() -> None:
    action = np.zeros(8, dtype=np.float32)
    lower = np.full(8, -1.0, dtype=np.float32)
    upper = np.full(8, 1.0, dtype=np.float32)
    np.testing.assert_array_equal(validate_action_step(action, lower=lower, upper=upper), action)
    action[6] = 1.1
    with pytest.raises(InvalidActionError, match="dof 6"):
        validate_action_step(action, lower=lower, upper=upper)
    assert action[6] == pytest.approx(1.1)


def test_gripper_action_at_exact_limits_is_unchanged() -> None:
    lower = np.full(8, -1.0, dtype=np.float32)
    upper = np.full(8, 1.0, dtype=np.float32)
    lower[7], upper[7] = 0.0, 0.04

    for boundary in (lower[7], upper[7]):
        action = np.zeros(8, dtype=np.float32)
        action[7] = boundary
        corrected = validate_action_step(action, lower=lower, upper=upper)
        np.testing.assert_array_equal(corrected, action)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-2.67964e-5, 0.0), (-1e-4, 0.0), (0.04005, 0.04)],
)
def test_tiny_gripper_limit_overshoot_is_corrected(value: float, expected: float) -> None:
    action = np.zeros(8, dtype=np.float32)
    action[7] = value
    original = action.copy()
    lower = np.full(8, -1.0, dtype=np.float32)
    upper = np.full(8, 1.0, dtype=np.float32)
    lower[7], upper[7] = 0.0, 0.04

    corrected = validate_action_step(
        action,
        lower=lower,
        upper=upper,
        gripper_limit_tolerance_m=1e-4,
    )

    assert corrected[7] == pytest.approx(expected)
    np.testing.assert_array_equal(action, original)


def test_large_gripper_overshoot_and_other_joint_overshoot_remain_invalid() -> None:
    lower = np.full(8, -1.0, dtype=np.float32)
    upper = np.full(8, 1.0, dtype=np.float32)
    lower[7], upper[7] = 0.0, 0.04

    action = np.zeros(8, dtype=np.float32)
    action[7] = -1.01e-4
    with pytest.raises(InvalidActionError, match="dof 7"):
        validate_action_step(action, lower=lower, upper=upper)

    action = np.zeros(8, dtype=np.float32)
    action[3] = upper[3] + 1e-5
    with pytest.raises(InvalidActionError, match="dof 3"):
        validate_action_step(action, lower=lower, upper=upper)


def test_gripper_limit_tolerance_is_configurable() -> None:
    action = np.zeros(8, dtype=np.float32)
    action[7] = -2.67964e-5
    lower = np.full(8, -1.0, dtype=np.float32)
    upper = np.full(8, 1.0, dtype=np.float32)
    lower[7], upper[7] = 0.0, 0.04

    with pytest.raises(InvalidActionError, match="dof 7"):
        validate_action_step(
            action,
            lower=lower,
            upper=upper,
            gripper_limit_tolerance_m=1e-5,
        )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_gripper_nan_and_inf_remain_invalid(invalid: float) -> None:
    action = np.zeros(8, dtype=np.float32)
    action[7] = invalid
    lower = np.full(8, -1.0, dtype=np.float32)
    upper = np.full(8, 1.0, dtype=np.float32)

    with pytest.raises(InvalidActionError, match="NaN or Inf"):
        validate_action_step(action, lower=lower, upper=upper)


def test_checkpoint_contract_requires_rgb_schema_and_sufficient_horizon() -> None:
    validate_rgb_checkpoint_contract(_modality_configs(), execution_horizon=16)
    with pytest.raises(ValueError, match="exceeds checkpoint action horizon"):
        validate_rgb_checkpoint_contract(_modality_configs(4), execution_horizon=8)
    multimodal = _modality_configs()
    multimodal["tactile"] = {"delta_indices": [0], "modality_keys": ["rgb"]}
    with pytest.raises(ValueError, match="RGB rollout adapter"):
        validate_rgb_checkpoint_contract(multimodal, execution_horizon=1)


def test_tactile_observation_requests_keep_policy_and_renderer_streams_separate() -> None:
    observations = {"camera": ["rgb"], "tactile": ["depth", "pose"]}
    _configure_tactile_observations(observations, _modality_configs(), None)
    assert observations["tactile"] == ["rgb_marker"]

    tactile_modalities = _modality_configs()
    tactile_modalities["tactile"] = {"delta_indices": [0], "modality_keys": ["rgb"]}
    contract = {"tactile": {"rgb": {"source_key": "tactile/left_gsmini/rgb"}}}
    _configure_tactile_observations(observations, tactile_modalities, contract)
    assert observations["tactile"] == ["rgb", "rgb_marker"]


def test_video_cleanup_with_no_frames_preserves_primary_episode_error(tmp_path: Path) -> None:
    source = tmp_path / "episode_error.mp4"
    source.touch()

    class FakeTask:
        save_video_path = tmp_path / "episode.mp4"
        video_handler = SimpleNamespace(ffmpeg=SimpleNamespace(pid=123, returncode=0))

        def clean_cache(self, *, result: str) -> None:
            assert result == "error"

    finalized = _finish_video(
        FakeTask(),
        monitor={"frames": 0},
        output_dir=tmp_path,
        task_name="lift_can",
        seed=7,
        success=None,
        termination_reason="exception",
    )

    assert finalized is None
    assert not source.exists()


def test_video_renderer_schema_rejects_missing_real_tactile_stream() -> None:
    observation = _observation()
    observation["tactile"] = {"left_tactile": {"rgb_marker": np.zeros((16, 16, 3), dtype=np.uint8)}}

    with pytest.raises(ValueError, match="tactile.right_tactile.rgb_marker"):
        _validate_video_renderer_observation(observation)

    observation["tactile"]["right_tactile"] = {"rgb_marker": np.zeros((16, 16, 3), dtype=np.uint8)}
    _validate_video_renderer_observation(observation)


def test_minimal_serializer_round_trips_requests_and_decodes_msgpack_numpy() -> None:
    request = {"observation": {"joint": np.arange(8, dtype=np.float32)}}
    decoded = CompatibleMsgSerializer.from_bytes(CompatibleMsgSerializer.to_bytes(request))
    np.testing.assert_array_equal(decoded["observation"]["joint"], request["observation"]["joint"])

    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    wire = msgpack.packb(
        {
            b"nd": True,
            b"type": array.dtype.str,
            b"kind": array.dtype.kind,
            b"shape": array.shape,
            b"data": array.tobytes(),
        },
        use_bin_type=True,
    )
    np.testing.assert_array_equal(CompatibleMsgSerializer.from_bytes(wire), array)


def test_task_resolution_and_result_classification() -> None:
    assert resolve_tasks(["all"]) == list(CANONICAL_TASKS)
    assert resolve_tasks(["lift_can", "insert_HDMI"]) == ["lift_can", "insert_HDMI"]
    with pytest.raises(ValueError, match="duplicate"):
        resolve_tasks(["lift_can", "lift_can"])
    assert classify_result(True, "success") == "success"
    assert classify_result(False, "early_stop") == "failure"
    assert classify_result(None, "invalid_action") == "error"
    assert classify_result(None, "worker_failure") == "error"


def test_video_output_path_separates_success_failure_and_error(tmp_path: Path) -> None:
    success = video_output_path(
        tmp_path,
        task="lift_can",
        seed=7,
        success=True,
        termination_reason="success",
    )
    failure = video_output_path(
        tmp_path,
        task="lift_can",
        seed=8,
        success=False,
        termination_reason="step_limit",
    )
    error = video_output_path(
        tmp_path,
        task="lift_can",
        seed=9,
        success=None,
        termination_reason="exception",
    )
    assert success == tmp_path / "videos/success/lift_can/lift_can_seed_7_success.mp4"
    assert failure == tmp_path / "videos/failure/lift_can/lift_can_seed_8_failure.mp4"
    assert error == tmp_path / "videos/error/lift_can/lift_can_seed_9_error.mp4"


def test_aggregation_uses_macro_average_and_excludes_errors() -> None:
    episodes = [
        _episode("lift_bottle", 0, True, "success"),
        _episode("lift_bottle", 1, False, "step_limit"),
        _episode("lift_can", 0, True, "success"),
        _episode("lift_can", 1, True, "success"),
        _episode("lift_can", 2, True, "success"),
        _episode("lift_can", 3, None, "exception"),
    ]
    aggregate = aggregate_results(episodes, ["lift_bottle", "lift_can"])
    assert aggregate["macro_average_percent"] == pytest.approx(75.0)
    assert aggregate["micro_average_percent"] == pytest.approx(80.0)
    assert aggregate["total_errors"] == 1
    assert aggregate["episode_counts_differ"] is True
    assert evaluation_exit_code(aggregate) == 1


def test_task_failure_is_not_an_evaluation_process_error() -> None:
    aggregate = aggregate_results([_episode("lift_can", 0, False, "step_limit")], ["lift_can"])
    assert aggregate["total_failures"] == 1
    assert evaluation_exit_code(aggregate) == 0


def test_macro_average_is_not_reported_when_a_task_has_no_valid_episode() -> None:
    episodes = [
        _episode("lift_bottle", 0, True, "success"),
        _episode("lift_can", 0, None, "invalid_action"),
    ]
    aggregate = aggregate_results(episodes, ["lift_bottle", "lift_can"])
    assert aggregate["macro_average"] is None
    assert aggregate["total_errors"] == 1


def test_result_files_contain_episode_rows_and_macro_summary(tmp_path: Path) -> None:
    episodes = [
        _episode("lift_bottle", 0, True, "success", steps=4),
        _episode("lift_bottle", 1, False, "early_stop", steps=8),
    ]
    aggregate = write_results(
        tmp_path,
        episodes,
        ["lift_bottle"],
        checkpoint="/models/checkpoint-2000",
        execution_horizon=1,
        episodes_per_task=2,
    )
    assert aggregate["macro_average_percent"] == pytest.approx(50.0)
    for filename in ("episodes.csv", "summary.csv", "summary.json", "summary.md", "summary.txt"):
        assert (tmp_path / filename).is_file()
    text = (tmp_path / "summary.txt").read_text()
    assert "Lift Bottle" in text
    assert "Average" in text
    assert "50.00%" in text
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["summary"]["tasks"][0]["successes"] == 1
    assert payload["episodes"][1]["termination_reason"] == "early_stop"


def test_worker_result_requires_explicit_completion_marker(tmp_path: Path) -> None:
    path = tmp_path / "worker.json"
    episode = _episode("lift_can", 0, False, "step_limit")
    _write_worker_results(
        path,
        [episode],
        {"version": "test"},
        completed=False,
        phase="task_construction",
        fatal_error="boom",
    )
    episodes, metadata, completed, phase, fatal_error = _load_worker_result(path)
    assert episodes == [episode]
    assert metadata == {"version": "test"}
    assert completed is False
    assert phase == "task_construction"
    assert fatal_error == "boom"


def test_missing_worker_episode_is_an_explicit_error() -> None:
    args = SimpleNamespace(
        seed_start=1_000_000,
        execution_horizon=1,
        checkpoint="checkpoint-10000",
    )
    result = _worker_failure_result(
        args,
        task="lift_can",
        episode_index=0,
        error="worker exited without a result",
    )
    assert result.success is None
    assert result.termination_reason == "worker_failure"
    assert result.video_path == ""


def test_main_and_isaac51_config_schemas_are_distinguished(tmp_path: Path) -> None:
    main_config = {"observations": {"camera": ["rgb"]}}
    assert _config_profile(main_config) == ("main", "4.5", "2.1.1")

    isaac51_config = {
        "env_settings": {
            "frequencies": {
                "physical": 120,
                "collect": 120,
                "save": 60,
                "video": 60,
                "eval": 60,
                "render": 0,
            },
            "random_texture": False,
            "sensor_type": "gsmini",
            "optical_backend": "taxim",
        },
        "observation_settings": {
            "camera": ["rgb"],
            "tactile": ["rgb_marker"],
            "embodiment": ["joint"],
        },
        "task_overrides": {},
    }
    assert _config_profile(isaac51_config) == ("isaac51", "5.1", "2.3.0")
    env_cfg = SimpleNamespace(
        save_dir=None,
        tactile_sensor_type=None,
        tactile_optical_backend=None,
        random_texture=None,
        obs_data_type=None,
        decimation=None,
        save_frequency=None,
        video_frequency=None,
        render_frequency=None,
        sim=SimpleNamespace(dt=None, render_interval=None),
        uipc_sim=SimpleNamespace(dt=None),
        cameras=[SimpleNamespace(update_period=None)],
    )
    profile, observations = _apply_task_config(
        env_cfg,
        isaac51_config,
        task_name="lift_can",
        output_dir=tmp_path,
    )
    assert profile == "isaac51"
    assert observations == isaac51_config["observation_settings"]
    assert env_cfg.decimation == 2
    assert env_cfg.video_frequency == 1
    assert env_cfg.render_frequency == 0
    assert env_cfg.sim.dt == pytest.approx(1 / 120)
    assert env_cfg.uipc_sim.dt == pytest.approx(1 / 120)


def test_official_isaac_lab_release_identities_are_recognized() -> None:
    assert _isaac_lab_release_matches("main", "0.41.3", "v2.1.1")
    assert _isaac_lab_release_matches("main", "0.41.3", None)
    assert _isaac_lab_release_matches("isaac51", "2.3.0", None)
    assert not _isaac_lab_release_matches("main", "12.0.0", "perf-2026-07-06-127-g04c2f318b6")


def test_mp4_validation_rejects_empty_artifact(tmp_path: Path) -> None:
    path = tmp_path / "empty.mp4"
    path.touch()
    with pytest.raises(RuntimeError, match="missing or empty"):
        _validate_mp4(path)


def test_zero_episode_worker_is_reported_as_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    univtac_root = tmp_path / "UniVTAC"
    (univtac_root / "envs").mkdir(parents=True)
    (univtac_root / "envs/_base_task.py").touch()
    (univtac_root / "task_config").mkdir()
    (univtac_root / "instructions").mkdir()
    output_dir = tmp_path / "output"

    class FakePolicy:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(closed_loop_eval, "Gr00tUniVTACPolicy", FakePolicy)
    monkeypatch.setattr(
        closed_loop_eval,
        "_stream_process",
        lambda *args, **kwargs: closed_loop_eval.ProcessOutcome(0, None, 0.1),
    )
    args = SimpleNamespace(
        univtac_root=univtac_root,
        checkpoint="checkpoint-10000",
        server_host="127.0.0.1",
        server_port=5555,
        mode="smoke",
        episodes_per_task=1,
        tasks=["lift_can"],
        execution_horizon=1,
        seed_start=1_000_000,
        task_config="clean",
        instruction_type="seen",
        step_limit=None,
        request_timeout_seconds=1.0,
        gripper_limit_tolerance_m=1e-4,
        output_dir=output_dir,
        python_command=sys.executable,
        expected_python=Path(sys.executable),
        device="cpu",
        headless=True,
        debug_first_rollout=True,
    )
    with pytest.raises(RuntimeError, match="did not complete a valid rollout"):
        closed_loop_eval.run(args)

    payload = json.loads((output_dir / "summary.json").read_text())
    assert payload["metadata"]["evaluation_status"] == "incomplete"
    assert payload["summary"]["total_errors"] == 1
    assert payload["episodes"][0]["termination_reason"] == "worker_failure"
