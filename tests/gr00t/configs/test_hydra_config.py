"""CPU configuration tests: no weights, downloads, CUDA or distributed initialization."""

from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from gr00t.configs.base_config import Config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.hydra_config import (
    build_gr00t_config,
    inherit_checkpoint_architecture,
    preflight,
)
from gr00t.configs.training.training_config import TrainingConfig
from gr00t.data.types import ActionRepresentation, ModalityConfig
from gr00t.experiment.hydra_launch import (
    find_latest,
    launch,
    resolve_resume,
    validate_checkpoint,
    validate_dataset_inputs,
    worker_command,
    write_snapshots,
)
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
import pytest


ROOT = Path(__file__).resolve().parents[3]


def composed(name="univtac_full", overrides=()):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(config_name=f"experiment/{name}", overrides=list(overrides))


@pytest.mark.parametrize(
    "name", ["univtac_full", "univtac_rgb", "univtac_overfit5", "so100_baseline"]
)
def test_composition_and_construction(name, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cfg = composed(name)
    result = build_gr00t_config(cfg)
    assert isinstance(result, Config)
    assert isinstance(result.training, TrainingConfig)
    assert result.training.num_gpus == 2
    assert result.data.datasets[0].embodiment_tag == "new_embodiment"
    assert isinstance(result.data.modality_configs["new_embodiment"]["action"], ModalityConfig)


def test_overrides_hardware_and_rates(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cfg = composed(
        overrides=[
            "hardware=single_3090",
            "training.max_steps=2000",
            "training.global_batch_size=2",
            "training.gradient_accumulation_steps=16",
            "training.action_head_learning_rate=5e-5",
            "training.point_encoder_learning_rate=1e-5",
        ]
    )
    result = build_gr00t_config(cfg)
    assert result.training.max_steps == 2000
    assert result.training.num_gpus == 1
    summary = preflight(cfg, result)
    assert summary["effective_batch_size"] == 32
    assert summary["per_device_batch_size"] == 2
    assert summary["requested_learning_rates"] == {
        "vlm": 1e-4,
        "action_head_including_sensor_projections": 5e-5,
        "point_backbone": 1e-5,
        "tactile_backbone": 5e-5,
    }


@pytest.mark.parametrize(
    "name",
    [
        "temporal_sidecar_ditfrozen-overfit5",
        "temporal_sidecar_ditlora-overfit5",
        "temporal_sidecar_ditfull-overfit5",
        "moe_sidecar_ditfrozen-overfit5",
        "moe_sidecar_ditlora-overfit5",
        "moe_sidecar_ditfull-overfit5",
        "temporal_sidecar_ditlora-full",
        "moe_sidecar_ditlora-full",
        "temporal_sidecar_ditfull-full",
        "moe_sidecar_ditfull-full",
    ],
)
def test_florence_default_optimizer_uses_existing_zero3_profile(name, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    default = build_gr00t_config(composed(name))
    zero = default.get_deepspeed_config()["zero_optimization"]
    assert zero["stage"] == 3
    assert "offload_optimizer" not in zero
    assert "offload_param" not in zero
    assert default.training.optim == "adamw_torch_fused"
    assert default.training.gradient_accumulation_steps == 8
    assert default.training.accumulated_batch_size == 32

    offloaded = build_gr00t_config(composed(name, ["hardware=quad_3090_zero3_cpu_offload"]))
    assert offloaded.get_deepspeed_config()["zero_optimization"]["offload_optimizer"] == {
        "device": "cpu",
        "pin_memory": False,
    }
    assert offloaded.training.accumulated_batch_size == default.training.accumulated_batch_size


@pytest.mark.parametrize(
    "name,family,dit_mode",
    [
        ("temporal_sidecar_ditlora-full", "temporal", "lora"),
        ("moe_sidecar_ditlora-full", "moe", "lora"),
        ("temporal_sidecar_ditfull-full", "temporal", "full"),
        ("moe_sidecar_ditfull-full", "moe", "full"),
    ],
)
def test_florence_2x2_full_dataset_experiment_modes(name, family, dit_mode, monkeypatch):
    monkeypatch.setenv("FLORENCE_DATASET", "/test/full_univtac")
    cfg = composed(name)
    result = build_gr00t_config(cfg)
    assert cfg.name == name
    assert cfg.data.get("expected_episode_ids") is None
    assert result.data.datasets[0].dataset_paths == ["/test/full_univtac"]
    assert result.training.max_steps == 10000
    assert result.training.warmup_ratio == 0.05
    assert result.model.dit_type == "sidecar_conditioned_dit"
    assert result.model.dit_training_mode == dit_mode
    assert result.model.freeze_pretrained_dit == (dit_mode == "lora")
    assert result.model.tune_diffusion_model == (dit_mode == "full")
    assert result.model.tune_projector
    assert not result.model.tune_vlln
    assert result.model.state_dropout_prob == pytest.approx(0.2)
    assert result.model.action_horizon == 40
    assert result.data.modality_configs["new_embodiment"]["action"].delta_indices == list(range(16))
    assert result.model.tactile_fusion_mode == "none"
    assert result.model.contact_prediction_loss_weight == 0.0
    assert result.model.use_point_conditioning
    assert result.model.use_tactile_conditioning
    if family == "temporal":
        assert result.model.temporal_window_size == 5
        assert result.model.contact_input_mode == "none"
        assert result.data.modality_configs["new_embodiment"]["pointcloud"].delta_indices == [
            -4,
            -3,
            -2,
            -1,
            0,
        ]
    else:
        assert result.model.temporal_window_size == 0
        assert result.model.contact_input_mode == "moe_soft"
        assert result.data.modality_configs["new_embodiment"]["pointcloud"].delta_indices == [0]


@pytest.mark.parametrize("mode", ["frozen", "lora", "full"])
def test_multilevel_local_uses_full_dataset_base(mode, monkeypatch):
    monkeypatch.setenv("FLORENCE_DATASET", "/test/full_univtac")
    monkeypatch.delenv("FLORENCE_OVERFIT_DATASET", raising=False)
    cfg = composed(f"wah_multilevel_local_{mode}")
    result = build_gr00t_config(cfg)

    assert cfg.data.get("expected_episode_ids") is None
    assert result.data.datasets[0].dataset_paths == ["/test/full_univtac"]
    assert result.training.max_steps == 10000
    assert result.training.warmup_ratio == 0.05
    assert result.training.action_head_learning_rate == pytest.approx(1e-4)
    assert result.training.dit_learning_rate == pytest.approx(1e-5)
    assert result.training.gradient_accumulation_steps == 8
    assert result.training.global_batch_size == 4

    assert result.model.sidecar_variant == "multilevel_local"
    assert result.model.sidecar_num_stages == 4
    assert result.model.sidecar_stage_indices is None
    assert result.model.sidecar_point_stage_names == ["enc1", "enc2", "enc3", "enc4"]
    assert result.model.sidecar_local_patch_size == 64
    assert result.model.sidecar_window_chunk_size == 16
    assert result.model.temporal_window_size == 0
    assert result.model.contact_input_mode == "none"
    assert result.model.tactile_fusion_mode == "none"
    assert result.model.contact_prediction_loss_weight == 0.0
    assert result.model.tune_projector
    assert not result.model.tune_vlln
    assert result.model.state_dropout_prob == pytest.approx(0.2)
    assert result.model.action_horizon == 40
    assert result.data.modality_configs["new_embodiment"]["action"].delta_indices == list(range(16))

    assert result.model.dit_training_mode == mode
    assert result.model.freeze_pretrained_dit == (mode != "full")
    assert result.model.tune_diffusion_model == (mode == "full")


def test_multilevel_local_default_dataset_has_no_overfit_dependency(monkeypatch):
    monkeypatch.delenv("FLORENCE_DATASET", raising=False)
    monkeypatch.delenv("FLORENCE_OVERFIT_DATASET", raising=False)
    for mode in ("frozen", "lora", "full"):
        cfg = composed(f"wah_multilevel_local_{mode}")
        assert cfg.data.get("expected_episode_ids") is None
        assert list(cfg.data.datasets[0].dataset_paths) == ["data/univtac"]
        assert "overfit" not in OmegaConf.to_yaml(cfg).lower()


def test_multilevel_local_dataset_validation_is_explicit(tmp_path, monkeypatch):
    missing = tmp_path / "missing-full-univtac"
    monkeypatch.setenv("FLORENCE_DATASET", str(missing))
    cfg = composed("wah_multilevel_local_frozen")
    with pytest.raises(ValueError, match=r"Dataset directory does not exist.*FLORENCE_DATASET"):
        validate_dataset_inputs(cfg)

    dataset = tmp_path / "full-univtac"
    metadata = dataset / "meta"
    metadata.mkdir(parents=True)
    (metadata / "info.json").write_text("{}")
    (metadata / "modality.json").write_text("{}")
    monkeypatch.setenv("FLORENCE_DATASET", str(dataset))
    validate_dataset_inputs(composed("wah_multilevel_local_full"))


def test_existing_overfit_and_legacy_defaults_are_unchanged(monkeypatch):
    monkeypatch.setenv("FLORENCE_OVERFIT_DATASET", "/test/independent-overfit5")
    overfit = composed("univtac_overfit5")
    assert list(overfit.data.expected_episode_ids) == [0, 1, 2, 3, 4]
    assert list(overfit.data.datasets[0].dataset_paths) == ["/test/independent-overfit5"]
    legacy = build_gr00t_config(composed("univtac_full"))
    assert legacy.model.sidecar_variant == "legacy"


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("model.base_model_path", None, "base_model_path"),
        ("model.unused", 1, "unsupported keys"),
        ("training.typo", 1, "unsupported keys"),
        ("data.preprocessing.typo", 1, "unsupported keys"),
        ("model.hidden_size", 8, "checkpoint architecture"),
        ("training.max_steps", "wrong", "training"),
        ("training.transformers_access_token", "secret", "unsupported keys"),
        ("data.modalities.video.modality_keys", [], "non-empty"),
        ("data.modalities.action.horizon", 41, "horizon"),
        ("data.modalities.action.horizon", 0, "positive integer"),
        ("model.point_input_dim", 3, "XYZRGB"),
        ("data.modalities.pointcloud.modality_keys", ["head", "wrist"], "pointcloud.scene"),
        ("model.tactile_input_channels", 6, "RGB channels"),
        ("data.modalities.tactile.delta_indices", [0, -1], "chronological"),
        ("data.modalities.tactile.delta_indices", [0], "chronological"),
        ("model.tactile_encoder_cfg", "unknown", "unsupported tactile encoder"),
        ("model.point_encoder_cfg", "unknown", "unsupported point encoder"),
        ("hardware.num_gpus", 3, "CUDA_VISIBLE_DEVICES"),
        ("hardware.cuda_visible_devices", "0,0", "distinct"),
        ("training.global_batch_size", 3, "divisible"),
        ("hardware.bf16", True, None),
        ("resume.enabled", True, "specify checkpoint_path"),
        ("resume.checkpoint_path", "/fake/checkpoint-1", "resume.enabled"),
    ],
)
def test_validation(key, value, message, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cfg = composed()
    OmegaConf.update(cfg, key, value, force_add=True)
    if message:
        with pytest.raises((ValueError, TypeError), match=message):
            build_gr00t_config(cfg)
    else:
        build_gr00t_config(cfg)


def test_required_fields_and_unknown_added_keys():
    cfg = composed()
    del cfg.data.datasets
    with pytest.raises(ValueError, match="at least one"):
        build_gr00t_config(cfg)
    cfg = composed()
    del cfg.data.modalities.tactile
    with pytest.raises(ValueError, match="tactile.rgb"):
        build_gr00t_config(cfg)
    cfg = composed()
    OmegaConf.update(cfg, "extra", 2, force_add=True)
    with pytest.raises(ValueError, match="unsupported keys"):
        build_gr00t_config(cfg)


def test_action_conversion_and_horizon():
    cfg = composed(overrides=["data.modalities.action.horizon=16"])
    result = build_gr00t_config(cfg)
    action = result.data.modality_configs["new_embodiment"]["action"]
    assert action.delta_indices == list(range(16))
    assert result.model.action_horizon == 40
    assert action.action_configs[0].rep is ActionRepresentation.RELATIVE
    assert action.action_configs[0].state_key == "joint"
    so = build_gr00t_config(composed("so100_baseline"))
    assert [a.rep for a in so.data.modality_configs["new_embodiment"]["action"].action_configs] == [
        ActionRepresentation.RELATIVE,
        ActionRepresentation.ABSOLUTE,
    ]


def test_modality_regression_against_legacy():
    path = ROOT / "examples/UniVTAC/univtac_config_common.py"
    spec = importlib.util.spec_from_file_location("univtac_common_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, multimodal, horizon in [("univtac_rgb", False, 16), ("univtac_full", True, 16)]:
        actual = build_gr00t_config(composed(name)).data.modality_configs["new_embodiment"]
        expected = module.build_univtac_config(multimodal=multimodal, action_horizon=horizon)
        assert {k: asdict(v) for k, v in actual.items()} == {
            k: asdict(v) for k, v in expected.items()
        }


def test_legacy_flags_precision_and_loading_policy():
    old = FinetuneConfig(
        base_model_path="base", dataset_path="dataset", embodiment_tag="new_embodiment"
    )
    new = build_gr00t_config(composed("univtac_rgb"))
    for key in (
        "tune_llm",
        "tune_visual",
        "tune_projector",
        "tune_diffusion_model",
        "tune_vlln",
        "tune_point_encoder",
        "tune_tactile_encoder",
        "tune_multimodal_adapter",
        "use_lora",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_bias",
        "state_dropout_prob",
    ):
        assert getattr(new.model, key) == getattr(old, key)
    for key in (
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "save_steps",
        "save_total_limit",
        "save_only_model",
        "shard_size",
        "allow_padding",
        "num_shards_per_epoch",
    ):
        obj = new.data if hasattr(new.data, key) else new.training
        assert getattr(obj, key) == getattr(old, key)
    assert new.training.bf16 and new.training.tf32 and not new.training.fp16
    assert not new.model.load_bf16
    assert new.model.backbone_trainable_params_fp32
    assert new.model.use_relative_action
    assert new.model.model_name == "nvidia/Cosmos-Reason2-2B"
    assert new.training.optim == "adamw_torch_fused"
    assert new.training.resume_from_checkpoint is False
    assert not new.training.skip_weight_loading
    assert new.data.episode_sampling_rate == 1  # full shell overrides CLI .1


def checkpoint(path, ds=False):
    path.mkdir()
    (path / "trainer_state.json").write_text(
        json.dumps({"global_step": int(path.name.split("-")[1])})
    )
    for name in (
        ["zero_optim_states.pt", "zero_model_states.pt"]
        if ds
        else ["optimizer.pt", "scheduler.pt", "model.safetensors"]
    ):
        (path / name).touch()
    (path / "rng_state.pth").touch()
    return path


@pytest.mark.parametrize("ds", [False, True])
def test_checkpoint_families_and_latest(tmp_path, ds):
    first = checkpoint(tmp_path / "checkpoint-10", ds)
    validate_checkpoint(first)
    second = checkpoint(tmp_path / "checkpoint-20", ds)
    partial = tmp_path / "checkpoint-30"
    partial.mkdir()
    (partial / "model.safetensors").touch()
    with pytest.raises(ValueError, match="not resumable"):
        validate_checkpoint(partial)
    assert find_latest(tmp_path) == second
    (second / "rng_state.pth").unlink()
    assert find_latest(tmp_path) == first


def test_snapshots_resume_and_compatibility(tmp_path):
    output = tmp_path / "run"
    cfg = composed(overrides=[f"output_dir={output}"])
    config = build_gr00t_config(cfg)
    snapshot = write_snapshots(cfg, config)
    frozen = OmegaConf.load(snapshot)
    assert not OmegaConf.is_interpolation(frozen, "run_id")
    assert (output / ".hydra/hydra.yaml").is_file()
    assert (output / ".hydra/overrides.yaml").is_file()
    assert (output / "metadata.json").is_file()
    check = checkpoint(output / "checkpoint-100", ds=True)
    cfg.resume.enabled = True
    cfg.resume.auto_find_latest = True
    cfg.training.max_steps = config.training.max_steps + 100
    assert resolve_resume(cfg, build_gr00t_config(cfg)) == check
    cfg.training.action_head_learning_rate = 5e-5
    with pytest.raises(ValueError, match="training.action_head_learning_rate"):
        resolve_resume(cfg, build_gr00t_config(cfg))
    cfg.training.action_head_learning_rate = None
    cfg.data.modalities.action.horizon = 15
    with pytest.raises(ValueError, match="data.modalities"):
        resolve_resume(cfg, build_gr00t_config(cfg))


def test_resume_requires_stable_output_and_provenance(tmp_path):
    path = checkpoint(tmp_path / "checkpoint-100")
    cfg = composed(overrides=["resume.enabled=true", f"resume.checkpoint_path={path}"])
    with pytest.raises(ValueError, match="original output directory"):
        resolve_resume(cfg, build_gr00t_config(cfg))
    cfg.output_dir = str(tmp_path)
    with pytest.raises(ValueError, match="missing"):
        resolve_resume(cfg, build_gr00t_config(cfg))


def test_resume_mode_conflicts():
    cfg = composed(
        overrides=[
            "resume.enabled=true",
            "resume.auto_find_latest=true",
            "training.save_only_model=true",
        ]
    )
    with pytest.raises(ValueError, match="save_only_model"):
        build_gr00t_config(cfg)
    cfg.training.save_only_model = False
    cfg.resume.checkpoint_path = "checkpoint-100"
    with pytest.raises(ValueError, match="OR"):
        build_gr00t_config(cfg)


def test_checkpoint_architecture_and_embedded_sensors():
    cfg = composed()
    config = build_gr00t_config(cfg)
    fake = SimpleNamespace(action_horizon=40, dit_type="alternate_vl_dit", hidden_size=99)
    inherit_checkpoint_architecture(cfg, config, fake)
    assert config.model.hidden_size == 99
    fake.action_horizon = 16
    with pytest.raises(ValueError, match="architecture"):
        inherit_checkpoint_architecture(cfg, config, fake)
    fake.action_horizon = 40
    fake.dit_type = "multimodal_conditioned_dit"
    fake.use_point_conditioning = True
    fake.point_encoder_cfg = "pointnet2"
    fake.point_input_dim = 6
    with pytest.raises(ValueError, match="Embedded checkpoint sensor"):
        inherit_checkpoint_architecture(cfg, config, fake)


def test_dry_run_never_imports_training_or_loads_checkpoint(tmp_path):
    # Guard imports in a fresh interpreter, not just a mocked call in this process.
    code = """
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'gr00t.experiment.experiment', 'gr00t.model.gr00t_n1d7.setup'}:
            raise AssertionError('training imported during preflight')
sys.meta_path.insert(0, Guard())
import torch
import torch.distributed
def forbidden(*args, **kwargs):
    raise AssertionError('CUDA, distributed, model or checkpoint initialized')
torch.cuda.init = forbidden
torch.distributed.init_process_group = forbidden
from transformers import AutoModel, AutoConfig, AutoProcessor
AutoModel.from_pretrained = forbidden
AutoConfig.from_pretrained = forbidden
AutoProcessor.from_pretrained = forbidden
import runpy
runpy.run_path('scripts/train_hydra.py', run_name='__main__')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "--config-name",
            "experiment/univtac_full",
            "dry_run=true",
            f"output_dir={tmp_path}/unused",
        ],
        cwd=ROOT,
        env=dict(os.environ, HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="0,1"),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"effective_batch_size": 32' in result.stdout
    assert not (tmp_path / "unused").exists()


def test_worker_command_uses_array_and_single_snapshot(tmp_path):
    cfg = composed()
    path = tmp_path / "space in run" / "config.yaml"
    command = worker_command(cfg, path)
    assert command[-1] == str(path)
    assert command[1:3] == ["-m", "torch.distributed.run"]
    assert "--nproc-per-node=2" in command


def test_dry_run_direct_does_not_launch(monkeypatch):
    cfg = composed(
        overrides=["dry_run=true", "resume.enabled=true", "resume.auto_find_latest=true"]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("subprocess launched")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    assert launch(cfg) == 0


def test_scalar_coercion_is_used_by_launcher_not_just_dataclasses():
    cfg = composed()
    cfg.dry_run = "false"
    cfg.resume.enabled = "false"
    cfg.hardware.num_gpus = "2"
    cfg.model.use_point_conditioning = "false"
    cfg.model.use_tactile_conditioning = "false"
    build_gr00t_config(cfg)
    assert cfg.dry_run is False
    assert cfg.resume.enabled is False
    assert cfg.hardware.num_gpus == 2
    assert cfg.model.use_point_conditioning is False


@pytest.mark.parametrize("field", ["preprocessing", "modalities"])
def test_required_data_mappings(field):
    cfg = composed()
    with open_dict(cfg.data):
        del cfg.data[field]
    with pytest.raises(ValueError, match=f"data.{field} is a required mapping"):
        build_gr00t_config(cfg)


def test_ineffective_legacy_controls_fail_early():
    for field, value in [
        ("data.shuffle", False),
        ("training.eval_set_split_ratio", 0.2),
        ("hardware.eval_bf16", False),
        ("training.upload_checkpoints", True),
    ]:
        cfg = composed()
        OmegaConf.update(cfg, field, value, force_add=True)
        with pytest.raises(ValueError, match="unsupported keys"):
            build_gr00t_config(cfg)
    cfg = composed(overrides=["training.eval_strategy=steps"])
    with pytest.raises(ValueError, match="separate open_loop_eval"):
        build_gr00t_config(cfg)


@pytest.mark.parametrize("model", ["gr00t_multimodal", "gr00t_sidecar"])
@pytest.mark.parametrize("enabled", [True, False])
def test_pretrained_loading_override(enabled, model):
    cfg = composed(
        overrides=[f"model={model}", f"model.optimize_pretrained_loading={str(enabled).lower()}"]
    )
    assert build_gr00t_config(cfg).model.optimize_pretrained_loading is enabled


@pytest.mark.parametrize("mode", ["standard", "koyori", "sopia", "roboco", "neuro"])
def test_trainctl_configuration_is_separate_from_training(mode):
    cfg = composed(overrides=[f"trainctl.message_mode={mode}"])
    config = build_gr00t_config(cfg)
    assert cfg.trainctl.enabled
    assert cfg.trainctl.message_mode == mode
    assert not hasattr(config.training, "trainctl")
    assert config.training.wandb_name == f"{cfg.name}-{cfg.run_id}"


def test_trainctl_rejects_secret_configuration_and_invalid_mode():
    cfg = composed()
    OmegaConf.update(cfg, "trainctl.webhook_url", "private", force_add=True)
    with pytest.raises(ValueError, match="unsupported keys"):
        build_gr00t_config(cfg)
    cfg = composed(overrides=["trainctl.message_mode=wrong"])
    with pytest.raises(ValueError, match="trainctl.message_mode"):
        build_gr00t_config(cfg)


def test_notification_changes_preserve_zero3_resume_and_identity(tmp_path):
    output = tmp_path / "original"
    cfg = composed(overrides=[f"output_dir={output}"])
    config = build_gr00t_config(cfg)
    original_name = config.training.wandb_name
    write_snapshots(cfg, config)
    saved = checkpoint(output / "checkpoint-100", ds=True)
    cfg.resume.enabled = True
    cfg.resume.auto_find_latest = True
    cfg.run_id = "ignored-new-id"
    cfg.trainctl.message_mode = "neuro"
    cfg.trainctl.notify_checkpoints = False
    cfg.trainctl.enabled = False
    cfg.training.max_steps += 100
    resumed = build_gr00t_config(cfg)
    assert resolve_resume(cfg, resumed) == saved
    assert resumed.training.wandb_name == original_name
    assert resumed.training.resume_from_checkpoint == str(saved)
