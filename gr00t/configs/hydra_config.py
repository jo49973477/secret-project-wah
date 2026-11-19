"""Hydra composition → validated existing GR00T dataclasses.

This module never imports the training pipeline, loads weights, or initializes CUDA.
Internal field types are reused rather than copying the GR00T schema.
"""

from dataclasses import dataclass, fields, make_dataclass
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from omegaconf import MISSING, DictConfig, OmegaConf

from gr00t.configs.base_config import Config
from gr00t.configs.data.data_config import DataConfig, SingleDatasetConfig
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.configs.training.training_config import TrainingConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Only fields that setup.py actually applies to pretrained models are exposed.
MODEL_FIELDS = frozenset(
    """
    sidecar_variant sidecar_point_stage_names sidecar_local_patch_size sidecar_window_chunk_size
    sidecar_num_stages sidecar_stage_indices sidecar_hidden_dim sidecar_num_heads
    sidecar_num_point_tokens sidecar_point_gate_init sidecar_action_gate_init
    sidecar_gate_init_seed sidecar_gate_mode sidecar_adaptive_gate_hidden_dim
    sidecar_adaptive_gate_init_action sidecar_adaptive_gate_init_point
    tactile_fusion_mode contact_num_tokens contact_hidden_dim
    contact_prediction_loss_weight freeze_pretrained_dit freeze_contact_teacher
    contact_injection_enabled contact_predictor_warmup
    dit_training_mode dit_lora_rank dit_lora_alpha dit_lora_dropout temporal_window_size
    contact_input_mode
    dit_type use_point_conditioning use_tactile_conditioning point_input_dim
    tactile_input_channels tactile_encoder_cfg tactile_pretrained_model
    tactile_checkpoint_filename tactile_background_path point_encoder_cfg
    point_encoder_checkpoint_path point_encoder_repo_id point_encoder_download_root
    point_encoder_grid_size point_encoder_enable_flash tune_llm tune_visual
    tune_projector tune_diffusion_model tune_vlln tune_point_encoder tune_tactile_encoder
    tune_multimodal_adapter use_lora lora_r lora_alpha lora_dropout lora_bias
    state_dropout_prob action_head_dropout_override vl_self_attention_dropout_override
    load_bf16 backbone_trainable_params_fp32 model_revision optimize_pretrained_loading
    use_flash_attention
""".split()
)
PREPROCESSING_FIELDS = frozenset(
    """
    random_rotation_angle color_jitter_params shortest_image_edge crop_fraction
    image_crop_size image_target_size extra_augmentation_config use_percentiles
    use_relative_action use_mean_std exclude_state letter_box_transform formalize_language
    apply_sincos_state_encoding use_albumentations_transforms
""".split()
)
HARDWARE_FIELDS = frozenset(
    """
    num_gpus deepspeed_stage use_ddp tf32 fp16 bf16 gradient_checkpointing
    dataloader_num_workers zero3_offload_optimizer zero3_offload_param zero3_sub_group_size
    ddp_bucket_cap_mb
""".split()
)
BATCH_FIELDS = {"global_batch_size", "gradient_accumulation_steps"}

TRAINING_RESERVED = HARDWARE_FIELDS | {
    "wandb_name",
    "output_dir",
    "experiment_name",
    "start_from_checkpoint",
    "resume_from_checkpoint",
    "skip_weight_loading",
    "per_gpu_batch_size",
    "transformers_access_token",
    # Present in the legacy dataclass but unused by this training implementation.
    "eval_bf16",
    "save_vl_model",
    "upload_checkpoints",
    "upload_every",
    "upload_last_n_checkpoints",
    "max_concurrent_uploads",
    "eval_set_split_ratio",
    "max_retries",
    "add_rl_callback",
    "enable_open_loop_eval",
    "open_loop_eval_traj_ids",
    "open_loop_eval_steps_per_traj",
    "open_loop_eval_plot_indices",
}
ARCHITECTURE_FIELDS = frozenset(
    """
    action_horizon max_state_dim max_action_dim hidden_size input_embedding_dim
    backbone_embedding_dim diffusion_model_cfg state_history_length max_seq_len
    add_pos_embed use_vlln attend_text_every_n_blocks
""".split()
)


@dataclass
class Hardware:
    cuda_visible_devices: str = MISSING
    master_port: int = 29500


@dataclass
class Resume:
    enabled: bool = False
    checkpoint_path: str | None = None
    auto_find_latest: bool = False


@dataclass
class Trainctl:
    enabled: bool = True
    message_mode: str = "sopia"
    notify_checkpoints: bool = True


@dataclass
class Experiment:
    name: str = MISSING
    run_id: str = MISSING
    output_dir: str = MISSING
    dry_run: bool = False


def _keys(values: dict, allowed: set | frozenset, path: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"{path}: unsupported keys {sorted(unknown)}; allowed: {sorted(allowed)}")


def _typed(cls: type, values: dict, path: str) -> dict:
    """Validate supplied fields using the existing dataclass types via OmegaConf."""
    hints = get_type_hints(cls, include_extras=True)
    _keys(values, {f.name for f in fields(cls)}, path)
    schema_fields = []
    for key in values:
        annotation = hints[key]
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
        schema_fields.append((key, annotation))
    schema = make_dataclass(f"{cls.__name__}Input", schema_fields)
    try:
        return OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.structured(schema), values),
            resolve=True,
            throw_on_missing=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _modalities(raw: dict) -> dict[str, ModalityConfig]:
    _keys(raw, {"video", "state", "action", "language", "pointcloud", "tactile"}, "data.modalities")
    missing = {"video", "state", "action", "language"} - raw.keys()
    if missing:
        raise ValueError(f"data.modalities: missing required modalities {sorted(missing)}")
    converted = {}
    for name, settings in raw.items():
        settings = dict(settings)
        if name == "action":
            if "delta_indices" in settings:
                raise ValueError("data.modalities.action: use horizon, not delta_indices")
            horizon = settings.pop("horizon", None)
            if type(horizon) is not int or horizon < 1:
                raise ValueError("data.modalities.action.horizon must be a positive integer")
            settings["delta_indices"] = list(range(horizon))
            actions = []
            for action in settings.get("action_configs", []):
                action = dict(action)
                _keys(action, {f.name for f in fields(ActionConfig)}, "action_configs")
                for key, enum in (
                    ("rep", ActionRepresentation),
                    ("type", ActionType),
                    ("format", ActionFormat),
                ):
                    value = action.get(key)
                    try:
                        action[key] = enum[value] if value in enum.__members__ else enum(value)
                    except (KeyError, ValueError, TypeError) as exc:
                        raise ValueError(f"action_configs.{key}: invalid {value!r}") from exc
                actions.append(ActionConfig(**_typed(ActionConfig, action, "action_configs")))
            if len(actions) != len(settings.get("modality_keys", [])):
                raise ValueError("action_configs must have one entry for each action modality key")
            # Validate types before constructing strong enum objects.
            settings.pop("action_configs", None)
            converted[name] = ModalityConfig(
                **_typed(ModalityConfig, settings, f"modalities.{name}"), action_configs=actions
            )
        else:
            converted[name] = ModalityConfig(
                **_typed(ModalityConfig, settings, f"modalities.{name}")
            )
        if not converted[name].delta_indices:
            raise ValueError(f"modalities.{name}.delta_indices must not be empty")
    return converted


def _model_values(raw_model: dict) -> tuple[str, dict, dict]:
    model = dict(raw_model)
    base = model.pop("base_model_path", None)
    if not isinstance(base, str) or not base:
        raise ValueError("model.base_model_path is required")
    expectations = model.pop("checkpoint_expectations", {})
    _keys(expectations, ARCHITECTURE_FIELDS, "model.checkpoint_expectations")
    _keys(
        model,
        MODEL_FIELDS,
        "model (checkpoint architecture is inherited; use checkpoint_expectations for assertions)",
    )
    model = _typed(Gr00tN1d7Config, model, "model")
    model.update(_typed(Gr00tN1d7Config, expectations, "model.checkpoint_expectations"))
    return base, model, expectations


def _data_values(raw_data: dict) -> tuple[dict, dict]:
    data = dict(raw_data)
    for key in ("modalities", "preprocessing"):
        if key not in data or not isinstance(data[key], dict):
            raise ValueError(f"data.{key} is a required mapping")
    expected_episodes = data.pop("expected_episode_ids", None)
    if expected_episodes is not None and (
        not isinstance(expected_episodes, list)
        or not expected_episodes
        or any(type(i) is not int or i < 0 for i in expected_episodes)
        or len(set(expected_episodes)) != len(expected_episodes)
    ):
        raise ValueError(
            "data.expected_episode_ids must be a nonempty list of distinct nonnegative integers; this asserts an existing subset, it does not filter a dataset"
        )
    modalities = _modalities(data.pop("modalities"))
    preprocessing = data.pop("preprocessing")
    _keys(preprocessing, PREPROCESSING_FIELDS, "data.preprocessing")
    preprocessing = _typed(Gr00tN1d7Config, preprocessing, "data.preprocessing")
    data["datasets"] = [
        SingleDatasetConfig(**_typed(SingleDatasetConfig, ds, "data.datasets"))
        for ds in data.get("datasets", [])
    ]
    if not data["datasets"]:
        raise ValueError("data.datasets must contain at least one dataset")
    tags = set()
    for ds in data["datasets"]:
        if not ds.dataset_paths or any(not isinstance(p, str) or not p for p in ds.dataset_paths):
            raise ValueError("data.datasets.dataset_paths must contain nonempty path strings")
        if ds.dataset_type != "physical_embodiment" or ds.val_dataset_path is not None:
            raise ValueError(
                "This sharded dataset factory supports physical_embodiment training only; validation datasets are not implemented"
            )
        ds.embodiment_tag = EmbodimentTag.resolve(ds.embodiment_tag).value
        tags.add(ds.embodiment_tag)
    # Remaining DataConfig fields reuse their own schema; nested datasets handled above.
    datasets = data.pop("datasets")
    _keys(
        data,
        {
            "shard_size",
            "episode_sampling_rate",
            "num_shards_per_epoch",
            "ds_weights_alpha",
            "override_pretraining_statistics",
            "mode",
            "seed",
            "multiprocessing_context",
            "allow_padding",
        },
        "data",
    )
    data = _typed(DataConfig, data, "data")
    data.update(datasets=datasets, modality_configs={tag: modalities for tag in tags})
    return data, preprocessing


def build_gr00t_config(cfg: DictConfig) -> Config:
    """Resolve, validate and construct Config directly, without FinetuneConfig/Tyro."""
    raw = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    _keys(
        raw,
        {
            "name",
            "run_id",
            "output_dir",
            "dry_run",
            "model",
            "data",
            "training",
            "hardware",
            "resume",
            "temporal",
            "dit_training",
            "trainctl",
        },
        "experiment",
    )
    for key in ("name", "run_id", "output_dir", "model", "data", "training", "hardware", "resume"):
        if key not in raw or raw[key] is None:
            raise ValueError(f"experiment.{key} is required")
    _typed(
        Experiment,
        {k: raw[k] for k in ("name", "run_id", "output_dir", "dry_run") if k in raw},
        "experiment",
    )
    monitoring = Trainctl(**_typed(Trainctl, raw.get("trainctl", {}), "trainctl"))
    if monitoring.message_mode not in {"standard", "koyori", "sopia", "roboco", "neuro"}:
        raise ValueError("trainctl.message_mode must be standard, koyori, sopia, roboco, or neuro")
    OmegaConf.update(cfg, "trainctl", OmegaConf.structured(monitoring), force_add=True)
    model_input = dict(raw["model"])
    if "temporal" in raw:
        _keys(raw["temporal"], {"window_size"}, "temporal")
        model_input["temporal_window_size"] = raw["temporal"]["window_size"]
        OmegaConf.update(
            cfg.model, "temporal_window_size", raw["temporal"]["window_size"], force_add=True
        )
    if "dit_training" in raw:
        _keys(raw["dit_training"], {"mode", "rank", "alpha", "dropout"}, "dit_training")
        for field, target in (
            ("mode", "dit_training_mode"),
            ("rank", "dit_lora_rank"),
            ("alpha", "dit_lora_alpha"),
            ("dropout", "dit_lora_dropout"),
        ):
            if field in raw["dit_training"]:
                model_input[target] = raw["dit_training"][field]
                OmegaConf.update(cfg.model, target, raw["dit_training"][field], force_add=True)
    base, model, expectations = _model_values(model_input)
    data, preprocessing = _data_values(raw["data"])
    model.update(preprocessing)
    hardware = dict(raw["hardware"])
    for key in ("cuda_visible_devices", "master_port", "num_gpus"):
        if key not in hardware:
            raise ValueError(f"hardware.{key} is required")
    _keys(
        hardware,
        HARDWARE_FIELDS | BATCH_FIELDS | {"cuda_visible_devices", "master_port"},
        "hardware",
    )
    launch_hw = _typed(
        Hardware, {k: hardware.pop(k) for k in ("cuda_visible_devices", "master_port")}, "hardware"
    )
    training = dict(raw["training"])
    _keys(training, {f.name for f in fields(TrainingConfig)} - TRAINING_RESERVED, "training")
    hardware = _typed(TrainingConfig, hardware, "hardware")
    batch_defaults = {k: hardware.pop(k) for k in BATCH_FIELDS if k in hardware}
    training = {**batch_defaults, **training, **hardware}
    training = _typed(TrainingConfig, training, "training")
    resume = Resume(**_typed(Resume, raw["resume"], "resume"))
    if resume.checkpoint_path and resume.auto_find_latest:
        raise ValueError("resume: choose checkpoint_path OR auto_find_latest")
    if (resume.checkpoint_path or resume.auto_find_latest) and not resume.enabled:
        raise ValueError("resume.enabled must be true when selecting a resume checkpoint")
    if resume.enabled and not (resume.checkpoint_path or resume.auto_find_latest):
        raise ValueError("resume: specify checkpoint_path or auto_find_latest=true")
    if resume.enabled and training.get("save_only_model", False):
        raise ValueError("resume is incompatible with save_only_model=true")
    training.update(
        output_dir=raw["output_dir"],
        experiment_name=None,
        wandb_name=f"{raw['name']}-{raw['run_id']}",
        start_from_checkpoint=base,
        resume_from_checkpoint=resume.checkpoint_path or resume.auto_find_latest,
    )
    config = Config(
        model=Gr00tN1d7Config(**model), data=DataConfig(**data), training=TrainingConfig(**training)
    )
    validate_compatibility(config, launch_hw)
    config.validate()
    # Preserve OmegaConf's normalized scalar types in the parent snapshot too.
    for key, value in _typed(
        Experiment,
        {k: raw[k] for k in ("name", "run_id", "output_dir", "dry_run") if k in raw},
        "experiment",
    ).items():
        cfg[key] = value
    for key in cfg.hardware:
        cfg.hardware[key] = (
            launch_hw[key]
            if key in launch_hw
            else batch_defaults[key]
            if key in BATCH_FIELDS
            else getattr(config.training, key)
        )
    for key in cfg.training:
        cfg.training[key] = getattr(config.training, key)
    for key in raw["model"]:
        if key in MODEL_FIELDS:
            cfg.model[key] = getattr(config.model, key)
    if "checkpoint_expectations" not in cfg.model:
        OmegaConf.update(cfg.model, "checkpoint_expectations", {}, force_add=True)
    for key in expectations:
        cfg.model.checkpoint_expectations[key] = getattr(config.model, key)
    for key in preprocessing:
        cfg.data.preprocessing[key] = getattr(config.model, key)
    for field in fields(Resume):
        OmegaConf.update(cfg.resume, field.name, getattr(resume, field.name), force_add=True)
    if "dry_run" not in cfg:
        OmegaConf.update(cfg, "dry_run", False, force_add=True)
    return config


def validate_compatibility(config: Config, hardware: dict) -> None:
    m, d, t = config.model, config.data, config.training
    if t.eval_strategy != "no":
        raise ValueError(
            "Sharded DatasetFactory does not support training-time evaluation; use the separate open_loop_eval.py workflow"
        )
    if d.mode != "single_turn":
        raise ValueError("This dataset factory supports data.mode=single_turn only")
    if d.shard_size < 1 or d.num_shards_per_epoch < 1 or not 0 < d.episode_sampling_rate <= 1:
        raise ValueError(
            "Data shard sizes must be positive and episode_sampling_rate must be in (0,1]"
        )
    if any(ds.mix_ratio <= 0 for ds in d.datasets):
        raise ValueError("Each dataset mix_ratio must be positive")
    if m.point_encoder_grid_size <= 0:
        raise ValueError("model.point_encoder_grid_size must be positive")
    from gr00t.configs.sidecar_config import validate_variant

    validate_variant(m)
    if m.dit_type == "sidecar_conditioned_dit":
        if m.dit_training_mode not in {"frozen", "lora", "full"}:
            raise ValueError("dit_training.mode must be frozen, lora or full")
        if m.freeze_pretrained_dit != (m.dit_training_mode != "full") or m.tune_diffusion_model != (
            m.dit_training_mode == "full"
        ):
            raise ValueError("DiT trainability flags conflict with dit_training.mode")
        if m.temporal_window_size not in {0, 5, 10}:
            raise ValueError("temporal.window_size must be 5 or 10 when enabled")
        if m.temporal_window_size and m.contact_input_mode != "none":
            raise ValueError("Temporal and Contact MoE are separate architecture families")
        if m.contact_input_mode != "none" and (
            m.contact_input_mode != "moe_soft" or m.tactile_fusion_mode != "none"
        ):
            raise ValueError("Contact MoE requires moe_soft and no duplicate tactile injection")
        if m.temporal_window_size and m.tactile_fusion_mode != "none":
            raise ValueError(
                "Temporal Fusion already includes tactile; disable legacy tactile injection"
            )
    ids = [v.strip() for v in hardware["cuda_visible_devices"].split(",") if v.strip()]
    if t.num_gpus < 1 or len(ids) != t.num_gpus or len(set(ids)) != len(ids):
        raise ValueError(
            "hardware.num_gpus must match the number of distinct CUDA_VISIBLE_DEVICES IDs"
        )
    if not 1 <= hardware["master_port"] <= 65535:
        raise ValueError("hardware.master_port must be between 1 and 65535")
    if t.gradient_accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be >= 1")
    if t.global_batch_size < 1 or t.global_batch_size % t.num_gpus:
        raise ValueError(
            "training.global_batch_size must be positive and divisible by hardware.num_gpus"
        )
    if (
        t.max_steps < 1
        or t.save_steps < 1
        or t.save_total_limit < 1
        or t.dataloader_num_workers < 0
    ):
        raise ValueError("steps/save_total_limit must be positive and workers nonnegative")
    if m.dit_type not in {
        "alternate_vl_dit",
        "multimodal_conditioned_dit",
        "sidecar_conditioned_dit",
    }:
        raise ValueError(
            "model.dit_type: only alternate_vl_dit and its multimodal extension are supported"
        )
    if m.dit_type not in {"multimodal_conditioned_dit", "sidecar_conditioned_dit"} and (
        m.use_point_conditioning or m.use_tactile_conditioning
    ):
        raise ValueError("sensor conditioning requires multimodal_conditioned_dit")
    for name in (
        "state_dropout_prob",
        "lora_dropout",
        "action_head_dropout_override",
        "vl_self_attention_dropout_override",
    ):
        value = getattr(m, name)
        if value is not None and not 0 <= value < 1:
            raise ValueError(f"model.{name} must be in [0,1)")
    if m.use_lora and (m.lora_r < 1 or m.lora_alpha < 1 or m.lora_bias != "none"):
        raise ValueError("LoRA requires positive rank/alpha and lora_bias=none")
    if (m.shortest_image_edge is None) != (m.crop_fraction is None):
        raise ValueError(
            "data.preprocessing.shortest_image_edge and crop_fraction must be set together"
        )
    if m.shortest_image_edge is not None and (
        m.image_crop_size is not None or m.image_target_size is not None
    ):
        raise ValueError(
            "Fractional resize/crop conflicts with legacy image_crop_size/image_target_size; set both legacy sizes to null"
        )
    if m.color_jitter_params is not None:
        _keys(
            m.color_jitter_params,
            {"brightness", "contrast", "saturation", "hue"},
            "data.preprocessing.color_jitter_params",
        )
    if (
        not 0 <= t.warmup_ratio <= 1
        or t.warmup_steps < 0
        or t.weight_decay < 0
        or t.max_grad_norm < 0
    ):
        raise ValueError(
            "warmup_ratio must be in [0,1]; warmup_steps, weight_decay and max_grad_norm must be nonnegative"
        )
    from transformers.trainer_utils import SchedulerType
    from transformers.training_args import OptimizerNames

    try:
        OptimizerNames(t.optim)
        SchedulerType(t.lr_scheduler_type)
    except ValueError as exc:
        raise ValueError(f"Invalid training optimizer/scheduler: {exc}") from exc
    if m.crop_fraction is not None and not 0 < m.crop_fraction <= 1:
        raise ValueError("data.preprocessing.crop_fraction must be in (0,1]")
    for modalities in d.modality_configs.values():
        action = modalities["action"]
        if len(action.delta_indices) > m.action_horizon:
            raise ValueError("dataset action horizon exceeds model maximum action horizon")
        for a in action.action_configs:
            if a.state_key is not None and a.state_key not in modalities["state"].modality_keys:
                raise ValueError(f"action state_key={a.state_key!r} is missing from state modality")
        if m.use_point_conditioning:
            point = modalities.get("pointcloud")
            expected_point = (
                list(range(1 - m.temporal_window_size, 1)) if m.temporal_window_size else [0]
            )
            if (
                point is None
                or point.modality_keys != ["scene"]
                or point.delta_indices != expected_point
            ):
                raise ValueError(
                    f"point conditioning requires pointcloud.scene delta_indices={expected_point}"
                )
            if m.point_input_dim != 6:
                raise ValueError(
                    "pointcloud.scene requires XYZRGB: model.point_input_dim must be 6"
                )
            if m.point_encoder_cfg not in {
                "concerto_small",
                "concerto_base",
                "pointnet2",
                "pointnet",
                "pointnet++",
                "point_transformer",
                "transformer",
            }:
                raise ValueError(f"unsupported point encoder {m.point_encoder_cfg!r}")
        if m.use_tactile_conditioning:
            tactile = modalities.get("tactile")
            if tactile is None or tactile.modality_keys != ["rgb"] or m.tactile_input_channels != 3:
                raise ValueError("tactile conditioning requires tactile.rgb with 3 RGB channels")
            if m.tactile_encoder_cfg not in {"sparsh_dino_base", "resnet18"}:
                raise ValueError(f"unsupported tactile encoder {m.tactile_encoder_cfg!r}")
            indices = tactile.delta_indices
            expected_tactile = (
                list(range(-m.temporal_window_size, 1)) if m.temporal_window_size else [-1, 0]
            )
            if m.tactile_encoder_cfg == "sparsh_dino_base" and indices != expected_tactile:
                raise ValueError(f"Sparsh-DINO requires chronological offsets {expected_tactile}")
            m.tactile_temporal_delta_indices = list(indices)
        if m.temporal_window_size:
            expected_state = list(range(1 - m.temporal_window_size, 1))
            if modalities["state"].delta_indices != expected_state:
                raise ValueError(f"Temporal state offsets must be {expected_state}")


def preflight(cfg: DictConfig, config: Config) -> dict[str, Any]:
    """Requested policy summary; actual tensor ownership is only known after model loading."""
    t, m = config.training, config.model
    return {
        "wandb": {
            "enabled": t.use_wandb,
            "project": t.wandb_project,
            "name": t.wandb_name,
            "group": t.wandb_group,
        },
        "gpu_ids": cfg.hardware.cuda_visible_devices,
        "world_size": t.num_gpus,
        "global_batch_size": t.global_batch_size,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "per_device_batch_size": t.global_batch_size // t.num_gpus,
        "effective_batch_size": t.accumulated_batch_size,
        "requested_learning_rates": {
            "vlm": t.resolved_vlm_learning_rate,
            "action_head_including_sensor_projections": t.resolved_action_head_learning_rate,
            "point_backbone": t.resolved_point_encoder_learning_rate,
            "tactile_backbone": t.resolved_tactile_encoder_learning_rate,
        },
        "sidecar": {
            "variant": m.sidecar_variant,
            "gate_mode": m.sidecar_gate_mode,
            "point_gate_status": (
                "not present in multilevel_local" if m.sidecar_variant == "multilevel_local" else "enabled"
            ),
            "point_stages": m.sidecar_point_stage_names,
            "dit_blocks": m.sidecar_stage_indices,
            "patch_size": m.sidecar_local_patch_size,
            "window_chunk_size": m.sidecar_window_chunk_size,
            "gate_initialization": {
                "point": (
                    None if m.sidecar_variant == "multilevel_local" else m.sidecar_point_gate_init
                ),
                "action": m.sidecar_action_gate_init,
                "seed": m.sidecar_gate_init_seed,
                "adaptive_hidden_dim": m.sidecar_adaptive_gate_hidden_dim,
                "adaptive_action": m.sidecar_adaptive_gate_init_action,
                "adaptive_point": (
                    None
                    if m.sidecar_variant == "multilevel_local"
                    else m.sidecar_adaptive_gate_init_point
                ),
            },
        },
        "conditioning": {
            "dit": m.dit_type,
            "point": m.use_point_conditioning,
            "tactile": m.use_tactile_conditioning,
        },
        "architecture": "Checkpoint-owned; expectations verified by workers before model loading",
        "resume": "Restores optimizer/scheduler LR and counters" if cfg.resume.enabled else None,
    }


def inherit_checkpoint_architecture(cfg: DictConfig, config: Config, checkpoint) -> None:
    """Assert requested constraints and inherit checkpoint-owned widths/horizons.

    Called only in workers. The existing setup loader retains all weight mismatch
    and deferred sensor bootstrap checks.
    """
    for key, expected in OmegaConf.to_container(cfg.model.checkpoint_expectations).items():
        actual = getattr(checkpoint, key, None)
        if actual != expected:
            raise ValueError(
                f"Checkpoint architecture {key}={actual!r} disagrees with checkpoint_expectations.{key}={expected!r}; architecture changes are not supported"
            )
    dit = getattr(checkpoint, "dit_type", "alternate_vl_dit")
    if dit not in {"alternate_vl_dit", "multimodal_conditioned_dit", "sidecar_conditioned_dit"}:
        raise ValueError(f"Incompatible checkpoint DiT architecture: {dit!r}")
    from gr00t.configs.sidecar_config import validate_checkpoint_variant

    validate_checkpoint_variant(checkpoint, config.model)
    if dit in {"multimodal_conditioned_dit", "sidecar_conditioned_dit"}:
        if config.model.dit_type != dit:
            raise ValueError("Cannot remove conditioning from a multimodal checkpoint")
        for sensor in ("point", "tactile"):
            flag = f"use_{sensor}_conditioning"
            if getattr(checkpoint, flag, False):
                keys = [
                    flag,
                    f"{sensor}_encoder_cfg",
                    "point_input_dim" if sensor == "point" else "tactile_input_channels",
                ]
                for key in keys:
                    if getattr(checkpoint, key) != getattr(config.model, key):
                        raise ValueError(
                            f"Embedded checkpoint sensor architecture conflicts with model.{key}"
                        )
    if getattr(checkpoint, "use_lora", False):
        for key in ("use_lora", "lora_r", "lora_alpha", "lora_dropout", "lora_bias"):
            if getattr(checkpoint, key) != getattr(config.model, key):
                raise ValueError(f"Embedded checkpoint LoRA conflicts with model.{key}")
    for key in ARCHITECTURE_FIELDS | {"model_name", "backbone_model_type", "model_dtype"}:
        if hasattr(checkpoint, key):
            setattr(config.model, key, getattr(checkpoint, key))
    validate_compatibility(
        config,
        {
            "cuda_visible_devices": cfg.hardware.cuda_visible_devices,
            "master_port": cfg.hardware.master_port,
        },
    )
