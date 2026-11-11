# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test Gr00tPolicy: observation validation and inference pipeline.

Uses mocked model and processor to avoid downloading checkpoints.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gr00t.data.types import ModalityConfig
from gr00t.policy.gr00t_policy import validate_model_processor_compatibility
import numpy as np
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "processor_config"
EMBODIMENT = "libero_sim"

VIDEO_KEYS = ["observation.images.rgb.head_256_256", "observation.images.rgb.left_wrist_256_256"]
STATE_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
LANGUAGE_KEY = "annotation.human.action.task_description"


def _build_modality_configs():
    return {
        EMBODIMENT: {
            "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS),
            "state": ModalityConfig(delta_indices=[0], modality_keys=STATE_KEYS),
            "action": ModalityConfig(delta_indices=list(range(16)), modality_keys=ACTION_KEYS),
            "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE_KEY]),
        }
    }


def _sensor_modalities(*names: str, state_window: int = 1) -> dict[str, ModalityConfig]:
    modalities = {
        "state": ModalityConfig(delta_indices=list(range(state_window)), modality_keys=["joint"])
    }
    if "pointcloud" in names:
        modalities["pointcloud"] = ModalityConfig(delta_indices=[0], modality_keys=["scene"])
    if "tactile" in names:
        modalities["tactile"] = ModalityConfig(delta_indices=[0], modality_keys=["rgb"])
    return modalities


class TestModelProcessorCompatibility:
    @pytest.mark.parametrize("dit_type", ["alternate_vl_dit", "dit"])
    def test_rgb_architecture_ignores_unused_multimodal_defaults(self, dit_type):
        config = SimpleNamespace(
            dit_type=dit_type,
            use_point_conditioning=True,
            use_tactile_conditioning=True,
            temporal_window_size=0,
        )

        validate_model_processor_compatibility(config, _sensor_modalities())

    def test_multimodal_requires_configured_pointcloud(self):
        config = SimpleNamespace(
            dit_type="multimodal_conditioned_dit",
            use_point_conditioning=True,
            use_tactile_conditioning=False,
            temporal_window_size=0,
        )

        with pytest.raises(ValueError, match="use_point_conditioning=True"):
            validate_model_processor_compatibility(config, _sensor_modalities())

    def test_multimodal_rejects_unexpected_pointcloud(self):
        config = SimpleNamespace(
            dit_type="multimodal_conditioned_dit",
            use_point_conditioning=False,
            use_tactile_conditioning=False,
            temporal_window_size=0,
        )

        with pytest.raises(ValueError, match="use_point_conditioning=False"):
            validate_model_processor_compatibility(config, _sensor_modalities("pointcloud"))

    def test_sidecar_requires_configured_tactile(self):
        config = SimpleNamespace(
            dit_type="sidecar_conditioned_dit",
            use_point_conditioning=False,
            use_tactile_conditioning=True,
            temporal_window_size=0,
        )

        with pytest.raises(ValueError, match="use_tactile_conditioning=True"):
            validate_model_processor_compatibility(config, _sensor_modalities())

    def test_sidecar_accepts_matching_sensor_contract(self):
        config = SimpleNamespace(
            dit_type="sidecar_conditioned_dit",
            use_point_conditioning=True,
            use_tactile_conditioning=True,
            temporal_window_size=0,
        )

        validate_model_processor_compatibility(
            config, _sensor_modalities("pointcloud", "tactile")
        )

    def test_temporal_window_validation_remains_enforced(self):
        config = SimpleNamespace(
            dit_type="sidecar_conditioned_dit",
            use_point_conditioning=True,
            use_tactile_conditioning=True,
            temporal_window_size=2,
        )
        modalities = _sensor_modalities("pointcloud", "tactile", state_window=2)

        with pytest.raises(ValueError, match="pointcloud window 2, processor declares 1"):
            validate_model_processor_compatibility(config, modalities)


@pytest.fixture
def policy():
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.device = torch.device("cpu")
    mock_model.dtype = torch.bfloat16
    mock_model.config.use_point_conditioning = False
    mock_model.config.use_tactile_conditioning = False
    mock_model.config.temporal_window_size = 0

    mock_model.get_action = MagicMock(
        return_value=BatchFeature(data={"action_pred": torch.randn(1, 16, 7)})
    )

    mock_processor = MagicMock()
    mock_processor.modality_configs = _build_modality_configs()
    mock_processor.get_modality_configs.return_value = _build_modality_configs()
    mock_processor.state_action_processor = MagicMock()
    mock_processor.action_dim = {EMBODIMENT: 7}
    mock_processor.max_action_dim = 128
    mock_processor.max_action_horizon = 50
    mock_processor.eval = MagicMock()
    mock_processor.training = False
    mock_processor.collator = MagicMock()
    structural_mask = np.zeros((50, 128), dtype=np.float32)
    structural_mask[:16, :7] = 1
    mock_processor.side_effect = lambda messages: {"action_mask": structural_mask.copy()}
    mock_processor.collator.side_effect = lambda features: BatchFeature(
        data={
            "inputs": {
                "action_mask": torch.from_numpy(
                    np.stack([feature["action_mask"] for feature in features])
                )
            }
        }
    )

    def fake_process_observation(observation, embodiment_tag):
        return BatchFeature(
            data={
                "state": torch.randn(1, 1, 128),
                "action_mask": torch.ones(1, 16, 128),
                "embodiment_id": torch.zeros(1, dtype=torch.long),
                "input_ids": torch.ones(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
                "pixel_values": torch.randn(1, 3, 256, 256),
                "image_grid_thw": torch.tensor([[1, 16, 16]]),
            }
        )

    mock_processor.process_observation = MagicMock(side_effect=fake_process_observation)

    def fake_decode_action(action, embodiment_tag, state=None):
        return {k: np.zeros((1, 16, 1), dtype=np.float32) for k in ACTION_KEYS}

    mock_processor.decode_action = MagicMock(side_effect=fake_decode_action)

    # Patch both AutoModel and AutoProcessor, and also the processor_config.json check
    with (
        patch("gr00t.policy.gr00t_policy.AutoModel") as MockAutoModel,
        patch("gr00t.policy.gr00t_policy.AutoProcessor") as MockAutoProcessor,
        patch("pathlib.Path.is_dir", return_value=False),
        patch("pathlib.Path.exists", return_value=True),
        patch(
            "examples.UniVTAC.univtac_rollout.checkpoint.checkpoint_identity",
            return_value="sha256:fixture",
        ),
    ):
        MockAutoModel.from_pretrained.return_value = (mock_model, {"missing_keys": []})
        MockAutoProcessor.from_pretrained.return_value = mock_processor

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        p = Gr00tPolicy(
            embodiment_tag=EMBODIMENT,
            model_path="/fake/path",
            device="cpu",
        )
    return p


def _make_observation(batch_size=1):
    return {
        "video": {
            k: np.random.randint(0, 255, (batch_size, 1, 256, 256, 3), dtype=np.uint8)
            for k in VIDEO_KEYS
        },
        "state": {
            k: np.random.randn(batch_size, 1, 1).astype(np.float32)
            for k in STATE_KEYS[:-1]  # all except gripper
        }
        | {"gripper": np.random.randn(batch_size, 1, 2).astype(np.float32)},
        "language": {
            LANGUAGE_KEY: [["pick up the apple"]] * batch_size,
        },
    }


class TestGr00tPolicyInit:
    def test_policy_has_model_and_processor(self, policy):
        assert policy.model is not None
        assert policy.processor is not None

    def test_policy_embodiment_tag(self, policy):
        assert policy.embodiment_tag is not None


class TestGr00tPolicyCheckObservation:
    def test_valid_observation_passes(self, policy):
        obs = _make_observation()
        policy.check_observation(obs)

    def test_missing_video_key_raises(self, policy):
        obs = _make_observation()
        del obs["video"][VIDEO_KEYS[0]]
        with pytest.raises(AssertionError):
            policy.check_observation(obs)

    def test_wrong_video_dtype_raises(self, policy):
        obs = _make_observation()
        obs["video"][VIDEO_KEYS[0]] = obs["video"][VIDEO_KEYS[0]].astype(np.float32)
        with pytest.raises(AssertionError):
            policy.check_observation(obs)

    def test_multimodal_observation_reaches_vla_step_data(self, policy):
        policy.modality_configs.update(
            {
                "pointcloud": ModalityConfig(delta_indices=[0], modality_keys=["xyz"]),
                "tactile": ModalityConfig(delta_indices=[0], modality_keys=["rgb"]),
            }
        )
        obs = _make_observation(batch_size=2)
        obs["pointcloud"] = {
            "xyz": np.random.randn(2, 1, 32, 3).astype(np.float32),
        }
        obs["tactile"] = {
            "rgb": np.random.randint(0, 256, (2, 1, 24, 32, 3), dtype=np.uint8),
        }

        policy.check_observation(obs)
        unbatched = policy._unbatch_observation(obs)
        sample = policy._to_vla_step_data(unbatched[1])

        np.testing.assert_array_equal(sample.pointclouds["xyz"], obs["pointcloud"]["xyz"][1])
        np.testing.assert_array_equal(sample.tactile["rgb"], obs["tactile"]["rgb"][1])

    def test_multimodal_observation_rejects_wrong_dtype(self, policy):
        policy.modality_configs["pointcloud"] = ModalityConfig(
            delta_indices=[0], modality_keys=["xyz"]
        )
        obs = _make_observation()
        obs["pointcloud"] = {"xyz": np.zeros((1, 1, 32, 3), dtype=np.float64)}

        with pytest.raises(AssertionError, match="float32"):
            policy.check_observation(obs)

    def test_tactile_history_repeats_after_reset_then_uses_previous(self, policy):
        policy.modality_configs["tactile"] = ModalityConfig(
            delta_indices=[-1, 0], modality_keys=["rgb"]
        )
        first = _make_observation()
        first_frame = np.full((1, 1, 24, 32, 3), 17, dtype=np.uint8)
        first["tactile"] = {"rgb": first_frame}

        policy.check_observation(first)
        first_pair = policy._with_tactile_history(first)["tactile"]["rgb"]
        np.testing.assert_array_equal(first_pair[:, 0], first_frame[:, 0])
        np.testing.assert_array_equal(first_pair[:, 1], first_frame[:, 0])

        second = _make_observation()
        second_frame = np.full((1, 1, 24, 32, 3), 29, dtype=np.uint8)
        second["tactile"] = {"rgb": second_frame}
        second_pair = policy._with_tactile_history(second)["tactile"]["rgb"]
        np.testing.assert_array_equal(second_pair[:, 0], first_frame[:, 0])
        np.testing.assert_array_equal(second_pair[:, 1], second_frame[:, 0])

        policy.reset()
        reset_pair = policy._with_tactile_history(second)["tactile"]["rgb"]
        np.testing.assert_array_equal(reset_pair[:, 0], second_frame[:, 0])


class TestGr00tPolicyGetAction:
    def test_get_action_returns_tuple(self, policy):
        obs = _make_observation()
        result = policy.get_action(obs)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_action_returns_dict(self, policy):
        obs = _make_observation()
        action, info = policy.get_action(obs)
        assert isinstance(action, dict)
        assert isinstance(info, dict)

    def test_action_mask_survives_processor_collator_model_path(self, policy):
        policy.get_action(_make_observation())

        model_inputs = policy.model.get_action.call_args.kwargs["inputs"]
        mask = model_inputs["action_mask"]
        assert mask.shape == (1, 50, 128)
        assert mask.sum().item() == 16 * 7


class _NumpyLanguageSimPolicy:
    def __init__(self):
        self.modality_configs = {
            "video": ModalityConfig(delta_indices=[0], modality_keys=["camera"]),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=["state"],
            ),
            "action": ModalityConfig(delta_indices=[0], modality_keys=["action"]),
            "language": ModalityConfig(
                delta_indices=[0],
                modality_keys=["annotation.human.action.task_description"],
            ),
        }
        self.last_observation = None

    def get_modality_config(self):
        return self.modality_configs

    def get_action(self, observation, options=None):
        self.last_observation = observation
        return {"action": np.zeros((1, 1, 2), dtype=np.float32)}, {}

    def reset(self, options=None):
        return {}


def test_sim_policy_wrapper_accepts_numpy_language_batches():
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

    policy = _NumpyLanguageSimPolicy()
    wrapper = Gr00tSimPolicyWrapper(policy)
    observation = {
        "video.camera": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "state.state": np.zeros((1, 1, 3), dtype=np.float32),
        "annotation.human.action.task_description": np.array(["follow the instruction"]),
    }

    action, info = wrapper.get_action(observation)

    assert policy.last_observation["language"][LANGUAGE_KEY] == [["follow the instruction"]]
    assert "action.action" in action
    assert info == {}
