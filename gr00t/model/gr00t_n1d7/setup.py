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

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
    EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
    Gr00tN1d7Processor,
)
from gr00t.model.gr00t_n1d7.sensor_loading import (
    MULTIMODAL_ADAPTER_PREFIXES,
    SIDECAR_ADAPTER_PREFIXES,
    checkpoint_sensor_layout,
    has_legacy_layerwise_attention_keys,
)
from gr00t.model.registry import register_model
from gr00t.utils.dist_utils import (
    is_dist_avail_and_initialized,
    run_or_wait_on_rank0,
    run_serialized_across_ranks,
)
from gr00t.utils.model_load_diagnostics import (
    log_model_load_memory,
    model_load_phase,
    trace_checkpoint_loading,
)


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


def log_parameter_summary(model: Gr00tN1d7) -> None:
    """Log and validate the trainable split used by backbone-LoRA fine-tuning."""
    named_parameters = list(model.named_parameters())
    total_params = sum(parameter.numel() for _, parameter in named_parameters)
    trainable_params = sum(
        parameter.numel() for _, parameter in named_parameters if parameter.requires_grad
    )
    lora_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if name.startswith("backbone.") and "lora_" in name and parameter.requires_grad
    ]
    action_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if name.startswith("action_head.") and parameter.requires_grad
    ]
    frozen_backbone_base = [
        (name, parameter)
        for name, parameter in named_parameters
        if name.startswith("backbone.") and "lora_" not in name and not parameter.requires_grad
    ]

    logging.info("Total parameters: %s", f"{total_params:,}")
    logging.info(
        "Trainable parameters: %s (%.2f%%)",
        f"{trainable_params:,}",
        100 * trainable_params / total_params,
    )
    logging.info("LoRA trainable parameters: %s", f"{sum(p.numel() for _, p in lora_parameters):,}")
    logging.info(
        "Action-head trainable parameters: %s",
        f"{sum(p.numel() for _, p in action_parameters):,}",
    )
    logging.info(
        "Frozen backbone base parameters: %s",
        f"{sum(p.numel() for _, p in frozen_backbone_base):,}",
    )

    if model.config.use_lora:
        if not lora_parameters:
            raise RuntimeError("LoRA is enabled but zero LoRA parameters are trainable")
        if not action_parameters:
            raise RuntimeError(
                "LoRA is enabled but the GR00T action head has no trainable parameters"
            )

        trainable_backbone_base = [
            name
            for name, parameter in named_parameters
            if name.startswith("backbone.") and "lora_" not in name and parameter.requires_grad
        ]
        if trainable_backbone_base:
            raise RuntimeError(
                "LoRA mode requires every backbone base parameter to be frozen; found: "
                f"{trainable_backbone_base[:5]}"
            )

        required_action_modules = [
            "action_head.state_encoder.",
            "action_head.action_encoder.",
            "action_head.action_decoder.",
            "action_head.model.",
        ]
        if model.config.add_pos_embed:
            required_action_modules.append("action_head.position_embedding.")
        if model.config.use_vlln:
            required_action_modules.append("action_head.vlln.")
        missing_action_modules = [
            prefix
            for prefix in required_action_modules
            if not any(name.startswith(prefix) for name, _ in action_parameters)
        ]
        if missing_action_modules:
            raise RuntimeError(
                "LoRA mode requires the complete action head to remain trainable; "
                f"no trainable parameters found under {missing_action_modules}"
            )

        logging.info("Representative trainable LoRA parameter: %s", lora_parameters[0][0])
        if frozen_backbone_base:
            logging.info(
                "Representative frozen backbone base parameter: %s", frozen_backbone_base[0][0]
            )
        for prefix in required_action_modules:
            representative = next(name for name, _ in action_parameters if name.startswith(prefix))
            logging.info(
                "Representative trainable %s parameter: %s",
                prefix.rstrip("."),
                representative,
            )
        logging.info(
            "NEW_EMBODIMENT uses existing projector/category ID %d",
            EMBODIMENT_TAG_TO_PROJECTOR_INDEX["new_embodiment"],
        )


def log_processor_summary(processor: Gr00tN1d7Processor) -> None:
    """Log preprocessing values that materially change training or evaluation."""
    keys = (
        "use_percentiles",
        "use_mean_std",
        "use_relative_action",
        "state_dropout_prob",
        "color_jitter_params",
        "random_rotation_angle",
        "use_albumentations",
        "crop_fraction",
        "shortest_image_edge",
        "image_crop_size",
        "image_target_size",
        "formalize_language",
        "max_action_horizon",
        "max_action_dim",
        "max_state_dim",
    )
    logging.info("Resolved processor configuration:")
    for key in keys:
        logging.info("  %s = %r", key, getattr(processor, key))


class Gr00tN1d7Pipeline(ModelPipeline):
    model_class = Gr00tN1d7
    processor_class = Gr00tN1d7Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""
        skip_weight_loading = getattr(self.config.training, "skip_weight_loading", False)
        if self.config.training.start_from_checkpoint is not None and not skip_weight_loading:
            checkpoint_config = AutoConfig.from_pretrained(
                self.config.training.start_from_checkpoint,
                **self.transformers_loading_kwargs,
            )
            checkpoint_multimodal, checkpoint_has_point, checkpoint_has_tactile = (
                checkpoint_sensor_layout(checkpoint_config)
            )
            requested_multimodal = self.config.model.dit_type in {
                "multimodal_conditioned_dit",
                "sidecar_conditioned_dit",
            }
            checkpoint_dit_type = getattr(checkpoint_config, "dit_type", "alternate_vl_dit")
            requested_dit_type = self.config.model.dit_type
            if checkpoint_multimodal and checkpoint_dit_type != requested_dit_type:
                raise ValueError(
                    "Incompatible adapter architecture: "
                    + checkpoint_dit_type
                    + " -> "
                    + requested_dit_type
                )
            from gr00t.model.extension.multilevel_local_sidecar import validate_checkpoint_variant

            validate_checkpoint_variant(checkpoint_config, self.config.model)
            if checkpoint_dit_type == "sidecar_conditioned_dit":
                for field in (
                    "sidecar_num_stages",
                    "sidecar_stage_indices",
                    "sidecar_hidden_dim",
                    "sidecar_num_heads",
                    "sidecar_num_point_tokens",
                    "contact_hidden_dim",
                    "contact_num_tokens",
                ):
                    if getattr(checkpoint_config, field) != getattr(self.config.model, field):
                        raise ValueError(f"Incompatible Sidecar checkpoint architecture: {field}")
            requested_point = requested_multimodal and self.config.model.use_point_conditioning
            requested_tactile = requested_multimodal and self.config.model.use_tactile_conditioning
            point_model_config = getattr(checkpoint_config, "point_encoder_model_config", None)
            legacy_point_resume = checkpoint_has_point and point_model_config is None
            if legacy_point_resume:
                logging.warning(
                    "This multimodal checkpoint predates persisted Concerto architecture config; "
                    "the official checkpoint will be read once to reconstruct its architecture. "
                    "Newly saved checkpoints do not require this compatibility path."
                )

            point_deferred = requested_point and not checkpoint_has_point
            tactile_deferred = requested_tactile and not checkpoint_has_tactile
            point_load_on_init = (
                legacy_point_resume
                if requested_multimodal
                else self.config.model.point_encoder_pretrained_load_on_init
            )
            tactile_load_on_init = (
                False if requested_multimodal else self.config.model.tactile_pretrained_load_on_init
            )
            dropout_overrides = {
                key: value
                for key, value in (
                    (
                        "action_head_dropout_override",
                        self.config.model.action_head_dropout_override,
                    ),
                    (
                        "vl_self_attention_dropout_override",
                        self.config.model.vl_self_attention_dropout_override,
                    ),
                )
                if value is not None
            }

            from gr00t.model.gr00t_n1d7.pretrained_loading import (
                checkpoint_backbone_shapes,
                finalize_pretrained_model,
                initialize_missing_sidecar,
            )

            def load_base_checkpoint():
                shapes = None
                bf16_shapes = None
                if self.config.model.optimize_pretrained_loading:
                    if self.config.training.bf16 and os.environ.get(
                        "GROOT_MODEL_LOAD_PRESERVE_BF16", "1"
                    ).lower() not in {"0", "false", "no"}:
                        bf16_shapes = {}
                    shapes = checkpoint_backbone_shapes(
                        self.config.training.start_from_checkpoint,
                        self.transformers_loading_kwargs,
                        bf16_parameter_shapes=bf16_shapes,
                    )
                log_model_load_memory("before AutoModel.from_pretrained")
                with (
                    model_load_phase(
                        "GR00T.from_pretrained",
                        checkpoint=self.config.training.start_from_checkpoint,
                    ),
                    trace_checkpoint_loading(),
                ):
                    loaded = AutoModel.from_pretrained(
                        self.config.training.start_from_checkpoint,
                        low_cpu_mem_usage=True,
                        checkpoint_backbone_shapes=shapes,
                        checkpoint_bf16_shapes=bf16_shapes,
                        tune_llm=self.config.model.tune_llm,
                        tune_visual=self.config.model.tune_visual,
                        tune_projector=self.config.model.tune_projector,
                        tune_diffusion_model=self.config.model.tune_diffusion_model,
                        tune_vlln=self.config.model.tune_vlln,
                        state_dropout_prob=self.config.model.state_dropout_prob,
                        backbone_trainable_params_fp32=(
                            self.config.model.backbone_trainable_params_fp32
                        ),
                        load_bf16=self.config.model.load_bf16,
                        use_flash_attention=self.config.model.use_flash_attention,
                        transformers_loading_kwargs=self.transformers_loading_kwargs,
                        dit_type=self.config.model.dit_type,
                        sidecar_variant=self.config.model.sidecar_variant,
                        sidecar_point_stage_names=self.config.model.sidecar_point_stage_names,
                        sidecar_local_patch_size=self.config.model.sidecar_local_patch_size,
                        sidecar_window_chunk_size=self.config.model.sidecar_window_chunk_size,
                        sidecar_num_stages=self.config.model.sidecar_num_stages,
                        sidecar_stage_indices=self.config.model.sidecar_stage_indices,
                        sidecar_hidden_dim=self.config.model.sidecar_hidden_dim,
                        sidecar_num_heads=self.config.model.sidecar_num_heads,
                        sidecar_num_point_tokens=self.config.model.sidecar_num_point_tokens,
                        sidecar_point_gate_init=self.config.model.sidecar_point_gate_init,
                        sidecar_action_gate_init=self.config.model.sidecar_action_gate_init,
                        sidecar_gate_init_seed=self.config.model.sidecar_gate_init_seed,
                        tactile_fusion_mode=self.config.model.tactile_fusion_mode,
                        contact_num_tokens=self.config.model.contact_num_tokens,
                        contact_hidden_dim=self.config.model.contact_hidden_dim,
                        contact_prediction_loss_weight=self.config.model.contact_prediction_loss_weight,
                        freeze_pretrained_dit=self.config.model.freeze_pretrained_dit,
                        freeze_contact_teacher=self.config.model.freeze_contact_teacher,
                        contact_injection_enabled=self.config.model.contact_injection_enabled,
                        contact_predictor_warmup=self.config.model.contact_predictor_warmup,
                        dit_training_mode=self.config.model.dit_training_mode,
                        dit_lora_rank=self.config.model.dit_lora_rank,
                        dit_lora_alpha=self.config.model.dit_lora_alpha,
                        dit_lora_dropout=self.config.model.dit_lora_dropout,
                        temporal_window_size=self.config.model.temporal_window_size,
                        contact_input_mode=self.config.model.contact_input_mode,
                        use_point_conditioning=self.config.model.use_point_conditioning,
                        use_tactile_conditioning=self.config.model.use_tactile_conditioning,
                        point_input_dim=self.config.model.point_input_dim,
                        tactile_input_channels=self.config.model.tactile_input_channels,
                        point_encoder_cfg=self.config.model.point_encoder_cfg,
                        point_encoder_checkpoint_path=self.config.model.point_encoder_checkpoint_path,
                        point_encoder_repo_id=self.config.model.point_encoder_repo_id,
                        point_encoder_download_root=self.config.model.point_encoder_download_root,
                        point_encoder_grid_size=self.config.model.point_encoder_grid_size,
                        point_encoder_enable_flash=self.config.model.point_encoder_enable_flash,
                        point_encoder_pretrained_load_on_init=point_load_on_init,
                        point_encoder_deferred_bootstrap=point_deferred,
                        point_encoder_model_config=point_model_config,
                        tactile_encoder_cfg=self.config.model.tactile_encoder_cfg,
                        tactile_pretrained_model=self.config.model.tactile_pretrained_model,
                        tactile_checkpoint_filename=self.config.model.tactile_checkpoint_filename,
                        tactile_background_path=self.config.model.tactile_background_path,
                        tactile_pretrained_load_on_init=tactile_load_on_init,
                        tactile_encoder_deferred_bootstrap=tactile_deferred,
                        tactile_temporal_delta_indices=(
                            self.config.model.tactile_temporal_delta_indices
                        ),
                        tune_point_encoder=self.config.model.tune_point_encoder,
                        tune_tactile_encoder=self.config.model.tune_tactile_encoder,
                        tune_multimodal_adapter=self.config.model.tune_multimodal_adapter,
                        output_loading_info=True,
                        **dropout_overrides,
                        **self.transformers_loading_kwargs,
                    )
                log_model_load_memory("after AutoModel.from_pretrained")
                return loaded

            serialize_loading = (
                requested_multimodal
                and is_dist_avail_and_initialized()
                and torch.distributed.get_world_size() > 1
                and os.environ.get("GROOT_SERIALIZE_MODEL_LOADING", "1").lower()
                not in {"0", "false", "no"}
            )
            if serialize_loading:
                from transformers.integrations import is_deepspeed_zero3_enabled

                if is_deepspeed_zero3_enabled():
                    raise RuntimeError(
                        "Serialized checkpoint loading cannot run inside active ZeRO-3 initialization: "
                        "its internal collectives require all ranks. Keep this pipeline's existing "
                        "model-before-TrainingArguments order; no initialization order was changed."
                    )
                logging.info("Serializing multimodal model loading across distributed ranks.")
                model, loading_info = run_serialized_across_ranks(
                    load_base_checkpoint,
                    label="GR00T checkpoint load",
                )
            else:
                model, loading_info = load_base_checkpoint()

            missing_keys = loading_info.get("missing_keys", [])
            unexpected_keys = loading_info.get("unexpected_keys", [])
            mismatched_keys = loading_info.get("mismatched_keys", [])
            newly_enabled_prefixes: tuple[str, ...] = ()
            if requested_multimodal and not checkpoint_multimodal:
                newly_enabled_prefixes += (
                    SIDECAR_ADAPTER_PREFIXES
                    if requested_dit_type == "sidecar_conditioned_dit"
                    else MULTIMODAL_ADAPTER_PREFIXES
                )
            if requested_dit_type == "sidecar_conditioned_dit":
                if self.config.model.temporal_window_size:
                    newly_enabled_prefixes += ("action_head.temporal_fusion.",)
                if self.config.model.dit_training_mode == "lora":
                    newly_enabled_prefixes += tuple(
                        name
                        for name, _ in model.named_parameters()
                        if name.startswith("action_head.model.") and ".lora_" in name
                    )
            if point_deferred:
                newly_enabled_prefixes += ("action_head.point_encoder.",)
            if tactile_deferred:
                newly_enabled_prefixes += ("action_head.tactile_encoder.",)
            from gr00t.model.gr00t_n1d7.dit_integrity import validate_loading_info

            validation = validate_loading_info(loading_info, newly_enabled_prefixes)
            model.pretrained_loading_report = {
                **validation,
                "checkpoint": str(self.config.training.start_from_checkpoint),
                "skip_weight_loading": False,
                "dit_type": model.config.dit_type,
                "diffusion_model_cfg": model.config.diffusion_model_cfg,
            }
            other_missing = validation["errors"].get("missing_keys", [])
            initialized_sensor_keys = [
                key
                for key in missing_keys
                if any(key.startswith(prefix) for prefix in newly_enabled_prefixes)
            ]
            if initialized_sensor_keys:
                logging.info(
                    "Accepted %d expected missing tensors from newly enabled multimodal modules.",
                    len(initialized_sensor_keys),
                )
            errors = []
            if other_missing:
                errors.append(f"Missing keys ({len(other_missing)}): {other_missing}")
            if unexpected_keys:
                errors.append(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
            if mismatched_keys:
                errors.append(f"Mismatched keys ({len(mismatched_keys)}): {mismatched_keys}")
            if errors:
                if has_legacy_layerwise_attention_keys(unexpected_keys):
                    errors.append(
                        "Architecture mismatch: this checkpoint uses the experimental "
                        "per-layer multimodal cross-attention layout, which is incompatible "
                        "with the shared-attention architecture."
                    )
                raise RuntimeError(
                    "Checkpoint weight mismatch for "
                    f"{self.config.training.start_from_checkpoint}:\n" + "\n".join(errors)
                )

            with model_load_phase("Sidecar.initialize_missing"):
                initialize_missing_sidecar(model, missing_keys)
            with model_load_phase("GR00T.finalize"):
                finalize_pretrained_model(model)

            if checkpoint_has_point or checkpoint_has_tactile:
                model.action_head.validate_embedded_sensor_backbones()

            if point_deferred or tactile_deferred:

                def bootstrap_sensor_backbones():
                    log_model_load_memory("before external sensor bootstrap")
                    with model_load_phase("sensors.bootstrap"), trace_checkpoint_loading():
                        model.action_head.bootstrap_deferred_sensor_backbones()
                    log_model_load_memory("after external sensor bootstrap")

                if serialize_loading:
                    run_serialized_across_ranks(
                        bootstrap_sensor_backbones,
                        label="external sensor bootstrap",
                    )
                else:
                    bootstrap_sensor_backbones()

            model.pretrained_loading_report["sensors"] = {
                "point": {
                    "status": "PASS" if checkpoint_has_point or point_deferred else "SKIP",
                    "source": "embedded checkpoint"
                    if checkpoint_has_point
                    else "external bootstrap"
                    if point_deferred
                    else "disabled",
                    "checkpoint": self.config.model.point_encoder_checkpoint_path,
                    "repo_id": self.config.model.point_encoder_repo_id,
                },
                "tactile": {
                    "status": "PASS" if checkpoint_has_tactile or tactile_deferred else "SKIP",
                    "source": "embedded checkpoint"
                    if checkpoint_has_tactile
                    else "external bootstrap"
                    if tactile_deferred
                    else "disabled",
                    "model": self.config.model.tactile_pretrained_model,
                    "checkpoint_filename": self.config.model.tactile_checkpoint_filename,
                },
            }

            if self.config.model.use_lora and not model.config.use_lora:
                model.enable_lora(
                    r=self.config.model.lora_r,
                    alpha=self.config.model.lora_alpha,
                    dropout=self.config.model.lora_dropout,
                    bias=self.config.model.lora_bias,
                )
            elif model.config.use_lora:
                logging.info("Loaded LoRA structure and weights directly from checkpoint metadata")

        else:
            model = self.model_class(
                self.config.model,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        if not hasattr(model, "pretrained_loading_report"):
            model.pretrained_loading_report = {
                "status": "SKIP",
                "skip_weight_loading": skip_weight_loading,
                "checkpoint": str(self.config.training.start_from_checkpoint),
                "reason": "Constructed without pretrained weight loading",
            }

        if model.config.dit_type == "sidecar_conditioned_dit":
            dit = model.action_head.model
            model.pretrained_loading_report["sidecar"] = {
                "variant": model.config.sidecar_variant,
                "point_stage_names": model.config.sidecar_point_stage_names,
                "dit_block_indices": list(dit.stage_indices),
                "concerto_enc_mode": (model.config.point_encoder_model_config or {}).get(
                    "enc_mode"
                ),
                "concerto_enc_channels": (model.config.point_encoder_model_config or {}).get(
                    "enc_channels"
                ),
                "local_patch_size": model.config.sidecar_local_patch_size,
                "window_chunk_size": model.config.sidecar_window_chunk_size,
                "gate_initialization": {
                    "point": model.config.sidecar_point_gate_init,
                    "action": model.config.sidecar_action_gate_init,
                    "seed": model.config.sidecar_gate_init_seed,
                },
                "construction_gate_values": [
                    {
                        "stage": index,
                        "point": getattr(stage, "initial_point_gate", None),
                        "action": stage.initial_action_gate,
                    }
                    for index, stage in enumerate(dit.sidecar_stages)
                ],
            }

        logging.debug(f"Model Config: {model.config}")
        with run_or_wait_on_rank0(label="final_model_config.json write") as is_rank0:
            if is_rank0:
                with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                    f.write(model.config.to_filtered_json())
        log_parameter_summary(model)
        logging.debug(f"Model architecture: {model}")

        log_model_load_memory("model loading complete")
        return model

    def _get_statistics(
        self,
    ) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""
        letter_box_transform = self.model_config.letter_box_transform
        logging.info("N1.7 letter_box_transform=%s", letter_box_transform)
        if self.config.training.start_from_checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                letter_box_transform=letter_box_transform,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation overrides
                exclude_state=self.model_config.exclude_state,
                # The action head is the sole stochastic state-dropout site.
                # Keeping processor-side dropout disabled avoids squaring the
                # configured survival probability for each training sample.
                state_dropout_prob=0.0,
                use_mean_std=self.model_config.use_mean_std,
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                letter_box_transform=letter_box_transform,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation
                exclude_state=self.model_config.exclude_state,
                # Preserve legacy model configs while applying the configured
                # probability only once, in the action head.
                state_dropout_prob=0.0,
                use_mean_std=self.model_config.use_mean_std,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        log_processor_summary(processor)
        logging.debug(
            "Processor configs for training: %s",
            json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2),
        )
        with run_or_wait_on_rank0(label="final_processor_config.json write") as is_rank0:
            if is_rank0:
                with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                    json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        with run_or_wait_on_rank0(label="dataset_statistics.json write") as is_rank0:
            if is_rank0:
                stats = train_dataset.get_dataset_statistics()
                stats_dict = convert_tensors_to_lists(stats)
                with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
                    json.dump(stats_dict, f, indent=2)
                logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d7Config, Gr00tN1d7Pipeline)
