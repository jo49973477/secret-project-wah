"""Real parquet/Hydra regressions for custom relative statistics; CPU only."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import MagicMock

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.configs.hydra_config import build_gr00t_config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.data.stats import (
    STATS_FINGERPRINTS_KEY,
    RelativeActionLoader,
    _compute_relative_action_fingerprint,
    calculate_stats_for_key,
    generate_rel_stats,
    generate_stats,
)
from gr00t.data.types import ActionFormat, ActionRepresentation, ActionType, EmbodimentTag
from hydra import compose, initialize_config_dir
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[3]
TAG = EmbodimentTag.NEW_EMBODIMENT


def hydra_config(name="univtac_full", overrides=()):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name=f"experiment/{name}", overrides=list(overrides))
    return build_gr00t_config(cfg)


@pytest.fixture
def episode(tmp_path):
    """No sensor files needed: stats read state/action, initialization reads metadata."""
    meta = tmp_path / "meta"
    data = tmp_path / "data/chunk-000"
    meta.mkdir()
    data.mkdir(parents=True)
    t = np.arange(60, dtype=np.float32)
    states = np.stack((2 * t, -t), axis=1)
    actions = np.stack((t**2 + 7, 3 * t + 11), axis=1)
    pd.DataFrame({"observation.state": list(states), "action": list(actions)}).to_parquet(
        data / "episode_000000.parquet"
    )
    (meta / "info.json").write_text(
        json.dumps(
            {
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "chunks_size": 1000,
                "fps": 30,
                "features": {
                    key: {"dtype": "float32", "shape": [2]}
                    for key in ("observation.state", "action")
                },
            }
        )
    )
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": len(t), "tasks": ["test task"]}) + "\n"
    )
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "test task"}) + "\n")
    (meta / "modality.json").write_text(
        json.dumps(
            {
                "state": {"joint": {"start": 0, "end": 2}},
                "action": {"joint": {"start": 0, "end": 2}},
                "video": {"head": {}, "wrist": {}},
            }
        )
    )
    generate_stats(tmp_path)
    return tmp_path, states, actions


def read_relative(path):
    return json.loads((path / "meta/relative_stats.json").read_text())


@pytest.mark.parametrize("name,horizon", [("univtac_rgb", 16), ("univtac_full", 16)])
def test_hydra_factory_initializes_custom_dataset(episode, monkeypatch, name, horizon):
    path, states, actions = episode
    monkeypatch.delitem(MODALITY_CONFIGS, TAG.value, raising=False)
    config = hydra_config(
        name, [f"data.datasets.0.dataset_paths=[{path}]", "data.num_shards_per_epoch=2"]
    )
    modalities = config.data.modality_configs[TAG.value]
    original = deepcopy(modalities)
    assert modalities["state"].modality_keys == ["joint"]
    assert modalities["action"].modality_keys == ["joint"]
    assert modalities["action"].delta_indices == list(range(horizon))
    action = modalities["action"].action_configs[0]
    assert (action.rep, action.type, action.format, action.state_key) == (
        ActionRepresentation.RELATIVE,
        ActionType.NON_EEF,
        ActionFormat.DEFAULT,
        "joint",
    )
    processor = MagicMock()
    training, evaluation = DatasetFactory(config).build(processor)
    assert evaluation is None
    assert training.datasets[0].modality_configs is modalities
    assert training.datasets[0].action_horizon == horizon
    assert TAG.value not in MODALITY_CONFIGS
    assert modalities == original
    relative = np.stack(
        [actions[i : i + horizon] - states[i] for i in range(len(states) - horizon + 1)]
    )
    stats = training.get_dataset_statistics()[TAG.value]["relative_action"]["joint"]
    np.testing.assert_allclose(stats["mean"], relative.mean(axis=0))
    assert np.asarray(stats["mean"]).shape == (horizon, 2)
    assert read_relative(path)[STATS_FINGERPRINTS_KEY][
        "joint"
    ] == _compute_relative_action_fingerprint(TAG, "joint", modality_configs=modalities)
    processor.set_statistics.assert_called_once()


@pytest.mark.parametrize("horizon", [1, 7, 16, 40])
def test_relative_loader_and_stats_respect_resolved_horizon(episode, monkeypatch, horizon):
    path, states, actions = episode
    monkeypatch.delitem(MODALITY_CONFIGS, TAG.value, raising=False)
    config = hydra_config(overrides=[f"data.modalities.action.horizon={horizon}"])
    modalities = config.data.modality_configs[TAG.value]
    loader = RelativeActionLoader(path, TAG, "joint", modality_configs=modalities)
    trajectories = np.stack(loader.load_relative_actions(0))
    expected = np.stack(
        [actions[i : i + horizon] - states[i] for i in range(len(states) - horizon + 1)]
    )
    np.testing.assert_array_equal(trajectories, expected)
    stats = calculate_stats_for_key(path, TAG, "joint", modality_configs=modalities)
    for key, expected_stat in {
        "mean": expected.mean(axis=0),
        "std": expected.std(axis=0),
        "min": expected.min(axis=0),
        "max": expected.max(axis=0),
        "q01": np.quantile(expected, 0.01, axis=0),
        "q99": np.quantile(expected, 0.99, axis=0),
    }.items():
        np.testing.assert_allclose(stats[key], expected_stat)


def test_custom_config_cache_hit_and_horizon_invalidation(episode, monkeypatch):
    path, _, _ = episode
    monkeypatch.delitem(MODALITY_CONFIGS, TAG.value, raising=False)
    modalities = hydra_config("univtac_rgb").data.modality_configs[TAG.value]
    generate_rel_stats(path, TAG, modality_configs=modalities)
    first = read_relative(path)
    original = calculate_stats_for_key
    calls = []

    def tracked(*args, **kwargs):
        calls.append(kwargs["modality_configs"])
        return original(*args, **kwargs)

    monkeypatch.setattr("gr00t.data.stats.calculate_stats_for_key", tracked)
    generate_rel_stats(path, TAG, modality_configs=deepcopy(modalities))
    assert not calls
    changed = deepcopy(modalities)
    changed["action"].delta_indices = list(range(7))
    generate_rel_stats(path, TAG, modality_configs=changed)
    assert calls == [changed]
    new = read_relative(path)
    assert new[STATS_FINGERPRINTS_KEY]["joint"] != first[STATS_FINGERPRINTS_KEY]["joint"]
    assert np.asarray(new["joint"]["mean"]).shape == (7, 2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rep", ActionRepresentation.ABSOLUTE),
        ("type", ActionType.EEF),
        ("format", ActionFormat.XYZ_ROTVEC),
        ("state_key", "other_joint"),
    ],
)
def test_explicit_semantics_invalidate_fingerprint(field, value):
    modalities = hydra_config("univtac_rgb").data.modality_configs[TAG.value]
    before = _compute_relative_action_fingerprint(TAG, "joint", modality_configs=modalities)
    changed = deepcopy(modalities)
    changed["action"].action_configs[0] = replace(
        changed["action"].action_configs[0], **{field: value}
    )
    assert _compute_relative_action_fingerprint(TAG, "joint", modality_configs=changed) != before


def test_builtin_registry_fingerprint_unchanged():
    tag = EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
    modalities = MODALITY_CONFIGS[tag.value]
    for key in ("eef_9d", "joint_position"):
        assert _compute_relative_action_fingerprint(
            tag, key
        ) == _compute_relative_action_fingerprint(tag, key, modality_configs=modalities)


def test_explicit_builtin_override_used_by_loader_and_fingerprint(episode):
    path, _, _ = episode
    tag = EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
    modalities = hydra_config("univtac_rgb").data.modality_configs[TAG.value]
    # This built-in tag has no 'joint' group; an explicit config must win.
    generate_rel_stats(path, tag, modality_configs=modalities)
    assert np.asarray(read_relative(path)["joint"]["mean"]).shape == (16, 2)


def test_builtin_relative_loader_without_explicit_config(episode):
    path, states, actions = episode
    tag = EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
    meta_path = path / "meta/modality.json"
    metadata = json.loads(meta_path.read_text())
    for modality in ("state", "action"):
        metadata[modality]["joint_position"] = {"start": 0, "end": 2}
    meta_path.write_text(json.dumps(metadata))
    horizon = len(MODALITY_CONFIGS[tag.value]["action"].delta_indices)
    expected = np.stack(
        [actions[i : i + horizon] - states[i] for i in range(len(states) - horizon + 1)]
    )
    # Original positional call signatures must continue to use the registry.
    trajectories = RelativeActionLoader(path, tag, "joint_position").load_relative_actions(0)
    np.testing.assert_array_equal(np.stack(trajectories), expected)
    stats = calculate_stats_for_key(path, tag, "joint_position", 1)
    np.testing.assert_allclose(stats["mean"], expected.mean(axis=0))
