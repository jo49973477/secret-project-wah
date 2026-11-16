#!/usr/bin/env python

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

import json
import logging
import os
from pathlib import Path
import warnings

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from transformers import TrainingArguments, set_seed
import wandb
import yaml

from gr00t.configs.base_config import Config
from gr00t.configs.training.training_config import check_resume_compatibility
from gr00t.experiment.discord_notifier import DiscordNotifier
from gr00t.experiment.telegram_notifier import TelegramNotifier

# Use custom trainer that profiles data loading & forward times
from gr00t.experiment.trainer import FirstStepMemoryCallback, Gr00tTrainer, ProfCallback
from gr00t.experiment.training_memory import log_memory_snapshot, log_trainable_parameter_summary
from gr00t.experiment.training_notification_callback import (
    TrainingNotificationCallback,
    TrainingNotificationContext,
)
from gr00t.experiment.utils import BestMetricCheckpointCallback, CheckpointFormatCallback
from gr00t.model import MODEL_REGISTRY
from gr00t.utils.dist_utils import run_on_rank0, run_or_wait_on_rank0
from gr00t.utils.initial_actions import INITIAL_ACTIONS_FILENAME, save_initial_actions


def _log_zero3_backward_state(model: torch.nn.Module) -> None:
    """Log bounded parameter metadata after a failed opt-in ZeRO-3 backward."""
    prefixes = (
        "point_encoder.backbone",
        "point_encoder.projection",
        "tactile_encoder.backbone",
        "tactile_encoder.projection",
        "point_cross_attention",
        "tactile_cross_attention",
        "point_gates",
        "tactile_gates",
    )
    seen: set[str] = set()
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    for name, parameter in model.named_parameters():
        category = next((prefix for prefix in prefixes if prefix in name), None)
        if category is None:
            continue
        # Include the deep Concerto sparse kernels, especially the 512-channel
        # CPE weights matching the reported backward dimension.
        sparse_kernel = ".cpe.0.weight" in name
        if category in seen and not sparse_kernel:
            continue
        if not sparse_kernel:
            seen.add(category)
        logging.warning(
            "ZeRO-3 backward state rank=%s parameter=%s shape=%s numel=%s "
            "ds_shape=%s ds_numel=%s ds_status=%s requires_grad=%s grad_shape=%s",
            rank,
            name,
            tuple(parameter.shape),
            parameter.numel(),
            getattr(parameter, "ds_shape", None),
            getattr(parameter, "ds_numel", None),
            getattr(parameter, "ds_status", None),
            parameter.requires_grad,
            tuple(parameter.grad.shape) if parameter.grad is not None else None,
        )


def setup_logging(debug: bool = False):
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    # Reduce verbosity of some libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)


def _assert_num_gpus_matches_world_size(num_gpus: int) -> None:
    """``num_gpus`` must equal the launcher's data-parallel world size.

    ``num_gpus`` drives per-device batch size and the DeepSpeed gating, while
    HF ``Trainer`` and dataset sharding use the real ``WORLD_SIZE`` from
    torchrun. A mismatch silently rescales the effective batch (e.g. an 8-rank
    launch with ``num_gpus=1`` trains at 8× the intended batch), so reconcile
    at the launcher trust boundary.
    """
    world_size = os.environ.get("WORLD_SIZE")
    if world_size is not None and int(world_size) != num_gpus:
        raise ValueError(
            f"config.training.num_gpus={num_gpus} does not match the launcher "
            f"WORLD_SIZE={world_size}. num_gpus drives per-device batch size and "
            f"DeepSpeed gating, while HF Trainer and dataset sharding use "
            f"WORLD_SIZE; set num_gpus to {world_size}."
        )


def warn_configs(config: Config):
    # updates to batch size
    _assert_num_gpus_matches_world_size(config.training.num_gpus)
    assert config.training.global_batch_size % config.training.num_gpus == 0, (
        "global_batch_size must be divisible by num_gpus"
    )

    if config.training.gradient_accumulation_steps > 1:
        logging.info(
            "global_batch_size=%d × gradient_accumulation_steps=%d "
            "→ accumulated_batch_size=%d per optimizer step",
            config.training.global_batch_size,
            config.training.gradient_accumulation_steps,
            config.training.accumulated_batch_size,
        )

    if config.training.per_gpu_batch_size is not None:
        warnings.warn(
            "per_gpu_batch_size will be deprecated in the future, please use global_batch_size instead. For now, this will override global_batch_size."
        )

    if config.training.warmup_steps > 0:
        warnings.warn(
            "warmup_steps will be deprecated in the future, please use warmup_ratio instead. For now, this will override warmup_ratio."
        )

    if (
        hasattr(config.model, "backbone_trainable_params_fp32")
        and not config.model.backbone_trainable_params_fp32
    ):
        warnings.warn(
            "backbone_trainable_params_fp32 is not True. This will be deprecated in the future."
        )

    if (
        hasattr(config.model, "use_albumentations_transforms")
        and not config.model.use_albumentations_transforms
    ):
        warnings.warn(
            "use_albumentations_transforms is not True. This will be deprecated in the future."
        )

    if (
        hasattr(config.model, "image_crop_size")
        and hasattr(config.model, "image_target_size")
        and (config.model.image_crop_size is not None or config.model.image_target_size is not None)
    ):
        assert (
            config.model.image_crop_size is not None and config.model.image_target_size is not None
        ), "image_crop_size and image_target_size must be set together"
        warnings.warn(
            "image_crop_size and image_target_size will be deprecated in the future. Please use shortest_image_edge and crop_fraction instead."
        )
        if hasattr(config.model, "shortest_image_edge") and hasattr(config.model, "crop_fraction"):
            assert (
                config.model.shortest_image_edge is None and config.model.crop_fraction is None
            ), (
                "Do not set shortest_image_edge and crop_fraction together with image_crop_size and image_target_size"
            )

    if (
        hasattr(config.model, "shortest_image_edge")
        and hasattr(config.model, "crop_fraction")
        and (config.model.shortest_image_edge is not None or config.model.crop_fraction is not None)
    ):
        assert config.model.use_albumentations_transforms, (
            "use_albumentations_transforms must be True when shortest_image_edge and crop_fraction are set"
        )


def _init_distributed_process_group() -> int:
    """Init the NCCL process group with ``device_id=cuda:LOCAL_RANK`` so NCCL
    binds the communicator eagerly (the PyTorch >=2.4 recommended pattern).
    No-op when ``dist`` is already initialized or ``WORLD_SIZE`` is unset / 1.
    """
    if dist.is_initialized():
        return dist.get_rank()
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        return dist.get_rank()
    return 0


def save_run_config_artifacts(
    save_cfg_dir: Path, output_dir: Path, config: Config, experiment_name: str
):
    """Write ``config.yaml`` / ``conf.yaml`` / ``wandb_config.json``."""
    save_cfg_dir.mkdir(parents=True, exist_ok=True)
    config.save(save_cfg_dir / "config.yaml")
    omegaconf_config = OmegaConf.create(config.__dict__)
    omegaconf_config["max_steps"] = config.training.max_steps
    omegaconf_config["save_steps"] = config.training.save_steps
    OmegaConf.save(omegaconf_config, save_cfg_dir / "conf.yaml", resolve=True)
    wandb_config_file = output_dir / "wandb_config.json"
    with open(wandb_config_file, "w") as f:
        json.dump(
            {
                "project": config.training.wandb_project,
                "run_id": experiment_name,
            },
            f,
        )
    logging.info(f"Saved config to {save_cfg_dir}")


def save_initial_actions_artifact(train_dataset, save_cfg_dir: Path):
    """Write ``initial_actions.npz`` from ``train_dataset``; a falsy payload is a no-op."""
    initial_actions = train_dataset.get_initial_actions()
    if not initial_actions:
        return
    initial_actions_path = save_cfg_dir / INITIAL_ACTIONS_FILENAME
    save_initial_actions(initial_actions, initial_actions_path)
    logging.info(f"Saved {len(initial_actions)} initial actions to {initial_actions_path}")


def _validate_resume_action_config(config: Config, output_dir: Path) -> None:
    """Reject a resume when its saved action semantics differ from this run."""
    resume = config.training.resume_from_checkpoint
    if not resume:
        return
    checkpoint = Path(resume) if isinstance(resume, str) else output_dir
    candidates = (
        checkpoint / "experiment_cfg" / "config.yaml",
        checkpoint.parent / "experiment_cfg" / "config.yaml",
        output_dir / "experiment_cfg" / "config.yaml",
    )
    saved_config_path = next((path for path in candidates if path.is_file()), None)
    if saved_config_path is None:
        raise ValueError(
            "Cannot verify action configuration for requested checkpoint resume. "
            "Start a fresh output directory or resume from a checkpoint that contains "
            "experiment_cfg/config.yaml."
        )
    with saved_config_path.open() as config_file:
        saved_config = yaml.safe_load(config_file)
    embodiment = config.data.datasets[0].embodiment_tag
    saved_action = (
        saved_config.get("data", {}).get("modality_configs", {}).get(embodiment, {}).get("action")
    )
    requested_action = config.data.modality_configs[embodiment]["action"]
    if saved_action is None:
        raise ValueError(
            f"Cannot verify action configuration in {saved_config_path}; refusing resume."
        )
    requested = {
        "delta_indices": list(requested_action.delta_indices),
        "modality_keys": list(requested_action.modality_keys),
        "action_configs": [
            {
                "rep": action.rep.value,
                "type": action.type.value,
                "format": action.format.value,
                "state_key": action.state_key,
            }
            for action in (requested_action.action_configs or [])
        ],
    }
    saved = {
        "delta_indices": saved_action.get("delta_indices"),
        "modality_keys": saved_action.get("modality_keys"),
        "action_configs": saved_action.get("action_configs") or [],
    }
    if saved != requested:
        raise ValueError(
            "Refusing checkpoint resume with incompatible action semantics: "
            f"saved={saved}, requested={requested}. Use a fresh output directory."
        )


def _notification_dataset_path(config: Config) -> str | None:
    paths = [
        str(path)
        for dataset in config.data.datasets
        for path in getattr(dataset, "dataset_paths", [])
    ]
    return os.pathsep.join(paths) if paths else None


def run(config: Config):
    """Main training function."""
    warn_configs(config)
    check_resume_compatibility(config.training)

    global_rank = _init_distributed_process_group()

    # Setup
    setup_logging()
    if global_rank != 0:
        logging.getLogger().setLevel(logging.WARNING)
    set_seed(config.data.seed)

    # Validate config
    config.validate()

    # Create output directory
    if config.training.experiment_name is None:
        output_dir = Path(config.training.output_dir)
        experiment_name = output_dir.name
    else:
        output_dir = Path(config.training.output_dir) / config.training.experiment_name
        experiment_name = config.training.experiment_name

    telegram_notifier = None
    trainctl_active = os.environ.get("GR00T_TRAINCTL_ACTIVE") == "1"
    hydra_launch = bool(os.environ.get("GR00T_HYDRA_CONFIG"))
    if config.training.telegram_on and not (trainctl_active or hydra_launch):
        # The token is resolved only into this runtime object. It is never attached
        # to config, so config/checkpoint/W&B serialization cannot expose it.
        telegram_notifier = TelegramNotifier.from_environment(
            chat_id=config.training.telegram_chat_id
        )

    discord_notifier = None
    if config.training.discord_on and not (trainctl_active or hydra_launch):
        # The credential-bearing URL exists only in this runtime transport object.
        discord_notifier = DiscordNotifier.from_environment()

    run_on_rank0(output_dir.mkdir, parents=True, exist_ok=True, label="output_dir.mkdir")

    save_cfg_dir = output_dir / "experiment_cfg"
    processor_dir = output_dir / "processor"
    _validate_resume_action_config(config, output_dir)

    # Rank-0-only write; wrapped so a rank-0 failure surfaces on every rank
    # instead of stranding peers at the next NCCL collective.
    run_on_rank0(
        save_run_config_artifacts,
        save_cfg_dir,
        output_dir,
        config,
        experiment_name,
    )

    # wandb.init does network I/O; wrap so a rank-0 failure can't strand peers.
    if config.training.use_wandb:
        with run_or_wait_on_rank0(label="wandb.init") as is_rank0:
            if is_rank0:
                config_dict = {
                    **config.__dict__,
                    "git_commit_hash": os.environ.get("GROOT_COMMIT_HASH", "unknown"),
                }

                hydra_path = os.environ.get("GR00T_HYDRA_CONFIG")
                if hydra_path:
                    config_dict["hydra_resolved"] = OmegaConf.to_container(
                        OmegaConf.load(hydra_path), resolve=True
                    )
                florence_preflight = os.environ.get("GR00T_FLORENCE_PREFLIGHT")
                if florence_preflight:
                    report = json.loads(Path(florence_preflight).read_text())
                    config_dict["experiment_architecture"] = (
                        "temporal_sidecar" if config.model.temporal_window_size else "moe_sidecar"
                    )
                    config_dict["dit_training_mode"] = config.model.dit_training_mode
                    config_dict["temporal_window_size"] = config.model.temporal_window_size
                    config_dict["dit_lora_rank"] = config.model.dit_lora_rank
                    config_dict["dit_lora_alpha"] = config.model.dit_lora_alpha
                    config_dict["pretrained_dit_verified"] = False
                    config_dict["pretrained_dit_action_audit_passed"] = (
                        report.get("checks", {}).get("actual_pretrained_dit") == "PASSED"
                    )
                    config_dict["pretrained_model_checkpoint_revision"] = report.get(
                        "actual_pretrained_checkpoint", {}
                    ).get("revision")
                    config_dict["dit_lora_parameter_count"] = report.get(
                        "actual_pretrained_checkpoint", {}
                    ).get("lora_parameter_count")

                wandb.init(
                    project=config.training.wandb_project,
                    name=config.training.wandb_name or experiment_name,
                    **(
                        {"group": config.training.wandb_group}
                        if config.training.wandb_group
                        else {}
                    ),
                    config=config_dict,
                    tags=[config.data.mode],
                )

    # Setup model training pipeline.
    pipeline = MODEL_REGISTRY.get(type(config.model))(config, save_cfg_dir)
    pipeline.setup()
    model = pipeline.return_model()
    if os.environ.get("GR00T_FLORENCE_PREFLIGHT"):
        from gr00t.model.gr00t_n1d7.dit_integrity import compare_shared_tensors

        with run_or_wait_on_rank0(label="runtime pretrained DiT verification") as is_rank0:
            if is_rank0:
                from scripts.verify_pretrained_dit import CheckpointTensors

                parent_report = json.loads(Path(os.environ["GR00T_FLORENCE_PREFLIGHT"]).read_text())
                checkpoint = parent_report["actual_pretrained_checkpoint"]["checkpoint"]
                dit = model.action_head.model
                if any(hasattr(parameter, "ds_id") for parameter in dit.parameters()):
                    raise RuntimeError(
                        "Runtime DiT tensors are already ZeRO partitioned; cannot prove every value without a coordinated per-layer gather"
                    )
                prefix = "action_head.model."
                candidate = {prefix + name: tensor for name, tensor in dit.state_dict().items()}
                original_names = [
                    row["name"]
                    for row in parent_report["actual_pretrained_checkpoint"]["weights"]["tensors"]
                ]
                runtime = compare_shared_tensors(
                    CheckpointTensors(checkpoint), candidate, expected_keys=original_names
                )
                preflight_dir = output_dir / "preflight"
                preflight_dir.mkdir(parents=True, exist_ok=True)
                (preflight_dir / "runtime_dit_verification.json").write_text(
                    json.dumps(runtime, indent=2) + "\n"
                )
                if runtime["status"] != "PASS":
                    raise RuntimeError(
                        "Actual training model DiT differs from pretrained checkpoint"
                    )
                metadata_path = output_dir / "metadata.json"
                metadata = json.loads(metadata_path.read_text())
                metadata["pretrained_dit_verified"] = True
                metadata["runtime_dit_matched_tensors"] = runtime["matched_tensors"]
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                if wandb.run is not None:
                    wandb.config.update(
                        {
                            "pretrained_dit_verified": True,
                            "runtime_dit_matched_tensors": runtime["matched_tensors"],
                        },
                        allow_val_change=True,
                    )
    log_memory_snapshot("after_model_loading")
    log_trainable_parameter_summary(model)
    train_dataset, eval_dataset = pipeline.return_dataset()
    data_collator = pipeline.return_collator()
    processor = pipeline.return_processor()
    # statistics.json here is read by Gr00tPolicy.from_pretrained at deploy time;
    # a torn write silently degrades inference, so gate + broadcast failures.
    run_on_rank0(processor.save_pretrained, processor_dir, label="processor.save_pretrained")

    # deepspeed config
    if config.training.num_gpus > 1 and not config.training.use_ddp:
        deepspeed_config = config.get_deepspeed_config()
        logging.info(
            "Runtime DeepSpeed configuration: %s", json.dumps(deepspeed_config, sort_keys=True)
        )
    else:
        deepspeed_config = None

    # for now we will let per_gpu_batch_size override global_batch_size, in future we will deprecate per_gpu_batch_size
    if config.training.per_gpu_batch_size is None:
        per_device_train_batch_size = config.training.global_batch_size // config.training.num_gpus
    else:
        per_device_train_batch_size = config.training.per_gpu_batch_size

    # Create training arguments
    resolved_vlm_lr = config.training.resolved_vlm_learning_rate
    resolved_action_head_lr = config.training.resolved_action_head_learning_rate
    resolved_point_encoder_lr = config.training.resolved_point_encoder_learning_rate
    resolved_tactile_encoder_lr = config.training.resolved_tactile_encoder_learning_rate
    logging.info(
        "Resolved optimizer learning rates: VLM=%g, Action Head=%g, Point Encoder=%g, "
        "Tactile Encoder=%g "
        "(base learning_rate=%g)",
        resolved_vlm_lr,
        resolved_action_head_lr,
        resolved_point_encoder_lr,
        resolved_tactile_encoder_lr,
        config.training.learning_rate,
    )
    from gr00t.utils.model_load_diagnostics import log_model_load_memory

    log_model_load_memory("before TrainingArguments / ZeRO-3 setup")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.training.max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        warmup_steps=config.training.warmup_steps,
        max_grad_norm=config.training.max_grad_norm,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        save_only_model=config.training.save_only_model,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        tf32=config.training.tf32,
        gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if config.training.gradient_checkpointing else None
        ),
        # With ZeRO CPU optimizer offload, Accelerate maps torch AdamW (including
        # fused AdamW) to DeepSpeedCPUAdam while retaining these parameter groups
        # and AdamW mode. The initialized engine is checked and logged below.
        optim=config.training.optim,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to="wandb" if config.training.use_wandb else "none",
        seed=config.data.seed,
        deepspeed=deepspeed_config,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=config.training.ddp_bucket_cap_mb,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        batch_eval_metrics=True,
        remove_unused_columns=config.training.remove_unused_columns,
        ignore_data_skip=True,
    )

    # Create trainer
    trainer = Gr00tTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        multiprocessing_context=config.data.multiprocessing_context,
        vlm_learning_rate=resolved_vlm_lr,
        action_head_learning_rate=resolved_action_head_lr,
        dit_learning_rate=config.training.resolved_dit_learning_rate,
        point_encoder_learning_rate=resolved_point_encoder_lr,
        tactile_encoder_learning_rate=resolved_tactile_encoder_lr,
        enable_training_dashboard=config.training.enable_training_dashboard,
        dashboard_refresh_seconds=config.training.dashboard_refresh_seconds,
        enable_multimodal_diagnostics=config.training.enable_multimodal_diagnostics,
        multimodal_diagnostics_interval=config.training.multimodal_diagnostics_interval,
        multimodal_diagnostics_level=config.training.multimodal_diagnostics_level,
    )
    if os.environ.get("GR00T_FLORENCE_PREFLIGHT"):
        groups = trainer._create_split_optimizer_groups(model)
        action_head = model.action_head
        dit = action_head.model
        from gr00t.model.extension.dit_lora import original_dit_parameters

        base = list(original_dit_parameters(dit))
        lora = [(name, p) for name, p in dit.named_parameters() if ".lora_" in name]
        mode = config.model.dit_training_mode
        if any(p.requires_grad != (mode == "full") for _, p in base):
            raise RuntimeError("Original DiT requires_grad disagrees with training mode")
        if any(p.requires_grad != (mode == "lora") for _, p in lora):
            raise RuntimeError("DiT LoRA requires_grad disagrees with training mode")
        if mode == "lora" and not lora:
            raise RuntimeError("LoRA mode has no trainable DiT adapters")
        owned = {id(p) for group in groups for p in group["params"]}
        if any(
            id(p) not in owned
            for _, p in (lora if mode == "lora" else base if mode == "full" else [])
        ):
            raise RuntimeError("Intended DiT parameters missing from optimizer groups")
        if global_rank == 0:
            counts = {
                "mode": mode,
                "original_dit_parameters": sum(p.numel() for _, p in base),
                "lora_parameters": sum(p.numel() for _, p in lora),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "frozen_parameters": sum(
                    p.numel() for p in model.parameters() if not p.requires_grad
                ),
                "optimizer_groups": [
                    {
                        "name": g["name"],
                        "learning_rate": g["lr"],
                        "parameters": sum(p.numel() for p in g["params"]),
                    }
                    for g in groups
                ],
            }
            (output_dir / "preflight" / "parameter_ownership.json").write_text(
                json.dumps(counts, indent=2) + "\n"
            )
            if wandb.run is not None:
                wandb.config.update(counts, allow_val_change=True)
    from gr00t.experiment.optimizer_artifacts import EffectiveOptimizerCallback

    trainer.add_callback(EffectiveOptimizerCallback(trainer))
    trainer.add_callback(
        FirstStepMemoryCallback(
            deepspeed_enabled=deepspeed_config is not None,
            zero3_offload_optimizer=config.training.zero3_offload_optimizer,
            zero3_offload_param=config.training.zero3_offload_param,
            zero3_sub_group_size=config.training.zero3_sub_group_size,
        )
    )

    trainer.add_callback(
        CheckpointFormatCallback(
            run_name=experiment_name,
            exp_cfg_dir=save_cfg_dir,
            processor_dir=processor_dir,
        )
    )

    notification_context = TrainingNotificationContext(
        experiment_name=experiment_name,
        output_dir=str(output_dir),
        model_path=config.training.start_from_checkpoint,
        dataset_path=_notification_dataset_path(config),
        num_gpus=config.training.num_gpus,
        global_batch_size=config.training.global_batch_size,
        learning_rate=config.training.learning_rate,
        deepspeed_stage=config.training.deepspeed_stage if deepspeed_config is not None else None,
        sample_exposure=trainer.sample_exposure,
    )
    notification_callbacks: list[TrainingNotificationCallback] = []
    if telegram_notifier is not None:
        notification_callbacks.append(
            TrainingNotificationCallback(
                notifier=telegram_notifier,
                context=notification_context,
                notify_start=config.training.telegram_notify_start,
                notify_save=config.training.telegram_notify_save,
                notify_finish=config.training.telegram_notify_finish,
                notify_error=config.training.telegram_notify_error,
                # HF fires on_train_end before the explicit final save below.
                defer_finish_until_final_save=True,
            )
        )
    if discord_notifier is not None:
        notification_callbacks.append(
            TrainingNotificationCallback(
                notifier=discord_notifier,
                context=notification_context,
                notify_start=config.training.discord_notify_start,
                notify_save=config.training.discord_notify_save,
                notify_finish=config.training.discord_notify_finish,
                notify_error=config.training.discord_notify_error,
                # HF fires on_train_end before the explicit final save below.
                defer_finish_until_final_save=True,
                failure_delivery_marker=os.environ.get("GR00T_FAILURE_NOTIFICATION_MARKER"),
                provider="discord",
            )
        )
    for notification_callback in notification_callbacks:
        trainer.add_callback(notification_callback)

    trainctl_callback = None
    if trainctl_active:
        from gr00t.experiment.trainctl_callback import TrainctlCallback

        trainctl_callback = TrainctlCallback(
            name=config.training.wandb_name or experiment_name,
            total_steps=config.training.max_steps,
            sample_exposure=trainer.sample_exposure,
            notify_checkpoints=os.environ.get("GR00T_TRAINCTL_CHECKPOINTS") == "1",
        )
        trainer.add_callback(trainctl_callback)

    if config.training.save_best_eval_metric_name != "":
        if deepspeed_config is not None and config.training.deepspeed_stage == 3:
            logging.warning(
                "Best-metric model snapshots are disabled under ZeRO-3; use the explicit "
                "checkpoint export tool to consolidate a selected resumable checkpoint."
            )
        else:
            trainer.add_callback(
                BestMetricCheckpointCallback(
                    metric_name=config.training.save_best_eval_metric_name,
                    greater_is_better=config.training.save_best_eval_metric_greater_is_better,
                    exp_cfg_dir=save_cfg_dir,
                    trainer=trainer,
                )
            )

    if hasattr(train_dataset, "get_initial_actions"):
        run_on_rank0(save_initial_actions_artifact, train_dataset, save_cfg_dir)

    # Train and finalize. Keep the original traceback and nonzero process exit.
    # trainctl decides completion/failure only after this entire process exits.
    logging.info("🚀 Starting training...")
    from gr00t.experiment.parameter_layout import audit_parameter_layout

    audit_parameter_layout(trainer.model)
    zero3_debug = os.environ.get("GR00T_ZERO3_DEBUG", "").lower() in {"1", "true", "yes"}
    if zero3_debug:
        logging.warning("ZeRO-3 debug enabled: autograd anomaly detection is active")
        torch.autograd.set_detect_anomaly(True)
    try:
        if config.training.enable_profiling:
            from functools import partial

            logging.info(f"{global_rank} Starting training with profiling...")

            def on_trace_ready_handler(trainer, profile_dir, prof):
                output_path = (
                    profile_dir / f"trace_rank_{global_rank}_iter_{trainer.state.global_step}.json"
                )
                prof.export_chrome_trace(str(output_path))
                logging.info(f"Trace saved to {output_path}")

            profile_dir = output_dir / "profiling"
            run_on_rank0(profile_dir.mkdir, parents=True, exist_ok=True, label="profile_dir.mkdir")

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    skip_first=10, wait=1, warmup=1, active=3, repeat=1
                ),
                # profile_memory=True,
                with_stack=True,
                # record_shapes=True,
                on_trace_ready=partial(on_trace_ready_handler, trainer, profile_dir),
            ) as prof:
                trainer.add_callback(ProfCallback(prof=prof))
                trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
        else:
            trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)

        # A public Trainer.save_model() consolidates ZeRO-3 parameters. Preserve
        # native resumability at the end without doing that full-model gather.
        if deepspeed_config is not None and config.training.deepspeed_stage == 3:
            final_checkpoint = output_dir / f"checkpoint-{trainer.state.global_step}"
            if trainer.state.global_step > 0 and not final_checkpoint.exists():
                logging.info(
                    "Saving final native ZeRO-3 training checkpoint at step %d",
                    trainer.state.global_step,
                )
                trainer._save_checkpoint(trainer.model, trial=None)
            logging.info(
                "Final ZeRO-3 training state is resumable at %s; use "
                "scripts/export_zero3_checkpoint.py for an inference model.",
                final_checkpoint,
            )
            final_artifact = final_checkpoint
        else:
            trainer.save_model()
            logging.info(f"Model saved to {output_dir}")
            final_step_checkpoint = output_dir / f"checkpoint-{trainer.state.global_step}"
            final_artifact = final_step_checkpoint if final_step_checkpoint.is_dir() else None

        if trainctl_callback is not None:
            trainctl_callback.register_checkpoint(
                final_artifact
                if deepspeed_config is not None and config.training.deepspeed_stage == 3
                else output_dir,
                state=trainer.state,
            )

        if config.training.assert_loss_less_than is not None:
            final_loss = trainer.loss
            if final_loss.item() > config.training.assert_loss_less_than:
                raise AssertionError(
                    f"Loss too high: {final_loss.item()} vs {config.training.assert_loss_less_than})"
                )

        if hasattr(train_dataset, "close"):
            train_dataset.close()
        if eval_dataset is not None and hasattr(eval_dataset, "close"):
            eval_dataset.close()

        for notification_callback in notification_callbacks:
            notification_callback.notify_finish(
                args=trainer.args,
                state=trainer.state,
                final_checkpoint=str(final_artifact) if final_artifact is not None else None,
            )
    except Exception as exc:
        if zero3_debug:
            try:
                _log_zero3_backward_state(trainer.model_wrapped or model)
            except Exception as diagnostic_error:
                logging.warning(
                    "ZeRO-3 state diagnostic failed (%s)", type(diagnostic_error).__name__
                )
        for notification_callback in notification_callbacks:
            notification_callback.notify_failure(exc, args=trainer.args, state=trainer.state)
        raise
    logging.info("Training completed!")
