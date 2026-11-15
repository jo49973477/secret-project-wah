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
import logging
import os
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.extension.multimodal_dit import MultiModalConditionedDiT
from gr00t.model.extension.point_encoder import ConcertoPointEncoder, build_point_encoder
from gr00t.model.extension.sidecar_dit import SidecarConditionedDiT
from gr00t.model.extension.tactile_encoder import SparshDinoTactileEncoder, build_tactile_encoder
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from gr00t.utils.model_load_diagnostics import log_model_load_memory


logger = logging.getLogger(__name__)


def _expand_action_mask(
    action_mask: torch.Tensor | None, actions: torch.Tensor
) -> torch.Tensor | None:
    if action_mask is None:
        return None
    if action_mask.ndim == 2:
        action_mask = action_mask.unsqueeze(-1)
    if action_mask.ndim != 3 or action_mask.shape[:2] != actions.shape[:2]:
        raise ValueError(
            "action_mask must have shape (batch, horizon) or "
            f"(batch, horizon, action_dim), got {tuple(action_mask.shape)}"
        )
    if action_mask.shape[-1] not in (1, actions.shape[-1]):
        raise ValueError(
            f"action_mask dimension must be 1 or {actions.shape[-1]}, got {action_mask.shape[-1]}"
        )
    return action_mask.to(device=actions.device, dtype=actions.dtype).expand_as(actions)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self._dtype_diagnostics_logged = False
        from gr00t.model.extension.multilevel_local_sidecar import (
            resolve_point_stages,
            validate_variant,
        )

        validate_variant(config)
        point_stage_dims = None
        if config.sidecar_variant == "multilevel_local":
            if config.point_encoder_model_config is None:
                config.point_encoder_model_config = ConcertoPointEncoder.read_architecture_config(
                    model_name=config.point_encoder_cfg,
                    checkpoint_path=config.point_encoder_checkpoint_path,
                    repo_id=config.point_encoder_repo_id,
                    download_root=config.point_encoder_download_root,
                    enable_flash=config.point_encoder_enable_flash,
                )
            count = (
                len(config.sidecar_stage_indices)
                if config.sidecar_stage_indices is not None
                else min(config.sidecar_num_stages, config.diffusion_model_cfg["num_layers"])
            )
            config.sidecar_point_stage_names, point_stage_dims = resolve_point_stages(
                config.point_encoder_model_config, config.sidecar_point_stage_names, count
            )
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.action_head_dropout_override is not None:
            config.diffusion_model_cfg = {
                **config.diffusion_model_cfg,
                "dropout": config.action_head_dropout_override,
            }

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if config.vl_self_attention_dropout_override is not None:
            if not vl_self_attention_cfg or vl_self_attention_cfg.get("num_layers", 0) <= 0:
                logger.warning(
                    "VL self-attention dropout override was requested, but this checkpoint "
                    "does not enable a VL self-attention stack."
                )
            else:
                vl_self_attention_cfg = {
                    **vl_self_attention_cfg,
                    "dropout": config.vl_self_attention_dropout_override,
                }
                config.vl_self_attention_cfg = vl_self_attention_cfg

        logger.info(
            "Resolved action-head diffusion dropout: %s",
            config.diffusion_model_cfg.get("dropout"),
        )
        logger.info(
            "Resolved VL self-attention dropout: %s",
            (
                vl_self_attention_cfg.get("dropout")
                if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0
                else "disabled"
            ),
        )

        if config.dit_type == "alternate_vl_dit":
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        elif config.dit_type == "sidecar_conditioned_dit":
            self.model = SidecarConditionedDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
                sidecar_variant=config.sidecar_variant,
                sidecar_point_stage_names=config.sidecar_point_stage_names,
                sidecar_point_stage_dims=point_stage_dims,
                sidecar_local_patch_size=config.sidecar_local_patch_size,
                sidecar_window_chunk_size=config.sidecar_window_chunk_size,
                sidecar_num_stages=config.sidecar_num_stages,
                sidecar_stage_indices=config.sidecar_stage_indices,
                sidecar_hidden_dim=config.sidecar_hidden_dim,
                sidecar_num_heads=config.sidecar_num_heads,
                sidecar_num_point_tokens=config.sidecar_num_point_tokens,
                tactile_fusion_mode=config.tactile_fusion_mode,
                contact_num_tokens=config.contact_num_tokens,
                contact_hidden_dim=config.contact_hidden_dim,
                freeze_pretrained_dit=config.freeze_pretrained_dit,
                freeze_contact_teacher=config.freeze_contact_teacher,
                contact_injection_enabled=config.contact_injection_enabled,
                dit_training_mode=config.dit_training_mode,
                dit_lora_rank=config.dit_lora_rank,
                dit_lora_alpha=config.dit_lora_alpha,
                dit_lora_dropout=config.dit_lora_dropout,
                contact_input_mode=config.contact_input_mode,
                contact_state_dim=config.max_state_dim,
            )
        elif config.dit_type == "multimodal_conditioned_dit":
            self.model = MultiModalConditionedDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using MultiModalConditionedDiT for diffusion model")
        elif config.dit_type == "dit":
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        else:
            raise ValueError(f"Unsupported dit_type: {config.dit_type!r}")

        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.point_encoder = None
        self.tactile_encoder = None
        if config.dit_type in {"multimodal_conditioned_dit", "sidecar_conditioned_dit"}:
            if config.use_point_conditioning and not config.point_encoder_deferred_bootstrap:
                normalized_point_encoder_cfg = config.point_encoder_cfg.lower().replace("-", "_")
                self.point_encoder = build_point_encoder(
                    config.point_encoder_cfg,
                    input_dim=config.point_input_dim,
                    point_dim=self.model.inner_dim,
                    **(
                        {
                            "checkpoint_path": config.point_encoder_checkpoint_path,
                            "repo_id": config.point_encoder_repo_id,
                            "download_root": config.point_encoder_download_root,
                            "grid_size": config.point_encoder_grid_size,
                            "enable_flash": config.point_encoder_enable_flash,
                            "load_pretrained": config.point_encoder_pretrained_load_on_init,
                            "architecture_config": config.point_encoder_model_config,
                        }
                        if normalized_point_encoder_cfg in {"concerto_small", "concerto_base"}
                        else {}
                    ),
                )
                if (
                    isinstance(self.point_encoder, ConcertoPointEncoder)
                    and self.point_encoder.architecture_config is not None
                ):
                    # The complete backbone is stored in the GR00T checkpoint. Persist its exact
                    # architecture so reload does not need the original Hub/local .pth file.
                    config.point_encoder_model_config = dict(self.point_encoder.architecture_config)
                    config.point_encoder_pretrained_load_on_init = False
            if config.use_tactile_conditioning and not config.tactile_encoder_deferred_bootstrap:
                self.tactile_encoder = build_tactile_encoder(
                    config.tactile_encoder_cfg,
                    input_channels=config.tactile_input_channels,
                    output_dim=self.model.inner_dim,
                    pretrained_model=config.tactile_pretrained_model,
                    checkpoint_filename=config.tactile_checkpoint_filename,
                    background_path=config.tactile_background_path,
                    load_pretrained=config.tactile_pretrained_load_on_init,
                    backbone_trainable=config.tune_tactile_encoder,
                )
                # A saved GR00T checkpoint contains the complete Sparsh state.
                # Persisting this flag prevents an unnecessary Hub download on reload.
                if isinstance(self.tactile_encoder, SparshDinoTactileEncoder):
                    config.tactile_pretrained_load_on_init = False

        if config.sidecar_variant == "multilevel_local" and self.point_encoder is not None:
            self.point_encoder.validate_stage_mapping(config.sidecar_point_stage_names)

        self.temporal_fusion = None
        self._temporal_ring = None
        if config.temporal_window_size:
            if not isinstance(self.model, SidecarConditionedDiT):
                raise ValueError("Temporal Fusion requires SidecarConditionedDiT")
            from gr00t.model.extension.temporal_fusion import LatentRingBuffer, TemporalFusion

            self.temporal_fusion = TemporalFusion(
                self.model.inner_dim, config.max_state_dim, config.temporal_window_size
            )
            self._temporal_ring = LatentRingBuffer(config.temporal_window_size)

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        # Pin the time-sampling Beta to CPU/fp32 explicitly. The action head can
        # be instantiated under a meta / no_init_weights default-device context
        # (e.g. nested from_pretrained). A Beta built from bare Python floats
        # would then place its concentration tensors on the meta device (or in
        # the active default dtype, e.g. bf16). With validate_args enabled that
        # already fails here in __init__ (Beta's internal .item() check cannot
        # run on meta); even with validation off, sample_time would later raise
        # or return garbage. Explicit device/dtype here makes the sampler depend
        # only on the config, not on the construction-time device/dtype context,
        # so the noise schedule is identical across SDPA/FA2/FA4 and meta vs.
        # real-device loads. config is the canonical source for these values.
        self.beta_dist = Beta(
            torch.tensor(float(config.noise_beta_alpha), dtype=torch.float32, device="cpu"),
            torch.tensor(float(config.noise_beta_beta), dtype=torch.float32, device="cpu"),
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            tune_vlln=config.tune_vlln,
            tune_point_encoder=config.tune_point_encoder,
            tune_tactile_encoder=config.tune_tactile_encoder,
            tune_multimodal_adapter=config.tune_multimodal_adapter,
        )

    def bootstrap_deferred_sensor_backbones(self) -> None:
        """Load official sensor backbones after the base GR00T checkpoint has finished."""
        config = self.config
        if config.point_encoder_deferred_bootstrap:
            if self.point_encoder is not None:
                raise RuntimeError("Deferred Concerto bootstrap found an existing point encoder.")
            self.point_encoder = build_point_encoder(
                config.point_encoder_cfg,
                input_dim=config.point_input_dim,
                point_dim=self.model.inner_dim,
                checkpoint_path=config.point_encoder_checkpoint_path,
                repo_id=config.point_encoder_repo_id,
                download_root=config.point_encoder_download_root,
                grid_size=config.point_encoder_grid_size,
                enable_flash=config.point_encoder_enable_flash,
                load_pretrained=True,
                memory_map_checkpoint=config.optimize_pretrained_loading,
            )
            if not hasattr(self.point_encoder, "architecture_config"):
                raise RuntimeError("Deferred point bootstrap did not create a Concerto encoder.")
            config.point_encoder_model_config = dict(self.point_encoder.architecture_config)
            config.point_encoder_pretrained_load_on_init = False
            config.point_encoder_deferred_bootstrap = False
            if config.sidecar_variant == "multilevel_local":
                self.point_encoder.validate_stage_mapping(config.sidecar_point_stage_names)

        if config.tactile_encoder_deferred_bootstrap:
            if self.tactile_encoder is not None:
                raise RuntimeError("Deferred Sparsh bootstrap found an existing tactile encoder.")
            self.tactile_encoder = build_tactile_encoder(
                config.tactile_encoder_cfg,
                input_channels=config.tactile_input_channels,
                output_dim=self.model.inner_dim,
                pretrained_model=config.tactile_pretrained_model,
                checkpoint_filename=config.tactile_checkpoint_filename,
                background_path=config.tactile_background_path,
                load_pretrained=True,
                backbone_trainable=config.tune_tactile_encoder,
                memory_map_checkpoint=config.optimize_pretrained_loading,
            )
            config.tactile_pretrained_load_on_init = False
            config.tactile_encoder_deferred_bootstrap = False

        self.set_trainable_parameters(
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
            tune_vlln=config.tune_vlln,
            tune_point_encoder=config.tune_point_encoder,
            tune_tactile_encoder=config.tune_tactile_encoder,
            tune_multimodal_adapter=config.tune_multimodal_adapter,
        )

    def validate_embedded_sensor_backbones(self) -> None:
        """Validate sensor weights restored from a multimodal GR00T checkpoint."""
        if self.point_encoder is not None and hasattr(
            self.point_encoder, "validate_pretrained_backbone"
        ):
            self.point_encoder.validate_pretrained_backbone("embedded GR00T checkpoint")
            self.config.point_encoder_pretrained_load_on_init = False
            self.config.point_encoder_model_config = dict(self.point_encoder.architecture_config)
        if isinstance(self.tactile_encoder, SparshDinoTactileEncoder):
            self.tactile_encoder.validate_pretrained_backbone("embedded GR00T checkpoint")
            self.config.tactile_pretrained_load_on_init = False

    def set_trainable_parameters(
        self,
        tune_projector: bool,
        tune_diffusion_model: bool,
        tune_vlln: bool,
        tune_point_encoder: bool | None = None,
        tune_tactile_encoder: bool | None = None,
        tune_multimodal_adapter: bool | None = None,
    ) -> None:
        tune_point_encoder = (
            self.config.tune_point_encoder if tune_point_encoder is None else tune_point_encoder
        )
        tune_tactile_encoder = (
            self.config.tune_tactile_encoder
            if tune_tactile_encoder is None
            else tune_tactile_encoder
        )
        tune_multimodal_adapter = (
            self.config.tune_multimodal_adapter
            if tune_multimodal_adapter is None
            else tune_multimodal_adapter
        )
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        self.tune_point_encoder = tune_point_encoder
        self.tune_tactile_encoder = tune_tactile_encoder
        self.tune_multimodal_adapter = tune_multimodal_adapter
        if isinstance(self.model, SidecarConditionedDiT):
            mode = self.config.dit_training_mode
            if self.config.freeze_pretrained_dit != (mode != "full"):
                raise ValueError("freeze_pretrained_dit conflicts with dit_training_mode")
            if tune_diffusion_model != (mode == "full"):
                raise ValueError("tune_diffusion_model conflicts with dit_training_mode")
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model or (
            isinstance(self.model, SidecarConditionedDiT) and self.config.freeze_pretrained_dit
        ):
            self.model.requires_grad_(False)
        if isinstance(self.model, (MultiModalConditionedDiT, SidecarConditionedDiT)):
            # The base DiT can remain frozen while the new residual adapters train.
            self.model.set_multimodal_adapter_trainable(tune_multimodal_adapter)
        if isinstance(self.model, SidecarConditionedDiT):
            from gr00t.model.extension.dit_lora import original_dit_parameters

            for _, parameter in original_dit_parameters(self.model):
                parameter.requires_grad_(self.config.dit_training_mode == "full")
            for name, parameter in self.model.named_parameters():
                if ".lora_" in name:
                    parameter.requires_grad_(self.config.dit_training_mode == "lora")
        if self.temporal_fusion is not None:
            self.temporal_fusion.requires_grad_(tune_multimodal_adapter)
        if self.point_encoder is not None:
            if isinstance(self.point_encoder, ConcertoPointEncoder):
                self.point_encoder.set_trainable(
                    backbone=tune_point_encoder,
                    projection=tune_multimodal_adapter and self.config.sidecar_variant == "legacy",
                )
            else:
                # Preserve historical all-or-nothing freezing for PointNet++
                # and other existing encoder implementations.
                self.point_encoder.requires_grad_(tune_point_encoder)
        if self.tactile_encoder is not None:
            if isinstance(self.tactile_encoder, SparshDinoTactileEncoder):
                self.tactile_encoder.set_trainable(
                    backbone=tune_tactile_encoder,
                    projection=tune_multimodal_adapter,
                )
            else:
                self.tactile_encoder.requires_grad_(tune_tactile_encoder)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        if isinstance(self.model, SidecarConditionedDiT) and self.config.freeze_contact_teacher:
            if self.tactile_encoder is not None:
                self.tactile_encoder.requires_grad_(False)
        if isinstance(self.model, SidecarConditionedDiT) and self.config.contact_predictor_warmup:
            if (
                not self.config.use_tactile_conditioning
                or self.config.tactile_fusion_mode == "none"
            ):
                raise ValueError("Contact warmup requires tactile observations")
            self.requires_grad_(False)
            for module in (
                self.model.contact_adapter.scene_pool,
                self.model.contact_adapter.state_projection,
                self.model.contact_adapter.predictor,
            ):
                module.requires_grad_(True)
            self.model.contact_injection_enabled = False
        if isinstance(self.model, SidecarConditionedDiT):
            for name, module in (
                ("sidecar", self.model.sidecar_stages),
                ("point_pool", self.model.sidecar_pool),
                ("contact", self.model.contact_adapter),
                ("contact_injection", self.model.contact_injections),
            ):
                total = sum(getattr(p, "ds_numel", p.numel()) for p in module.parameters())
                trainable = sum(
                    getattr(p, "ds_numel", p.numel())
                    for p in module.parameters()
                    if p.requires_grad
                )
                logger.info(
                    "%s parameters: total=%s trainable=%s (action-head optimizer ownership)",
                    name,
                    total,
                    trainable,
                )
        logger.debug(f"Tune point encoder: {self.tune_point_encoder}")
        logger.debug(f"Tune tactile encoder: {self.tune_tactile_encoder}")
        logger.debug(f"Tune multimodal adapter: {self.tune_multimodal_adapter}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self._temporal_ring is not None:
                self._temporal_ring.reset()
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
                if isinstance(self.model, (MultiModalConditionedDiT, SidecarConditionedDiT)):
                    self.model.set_multimodal_adapter_mode(self.tune_multimodal_adapter)
            elif (
                isinstance(self.model, (MultiModalConditionedDiT, SidecarConditionedDiT))
                and not self.tune_multimodal_adapter
            ):
                self.model.set_multimodal_adapter_mode(False)
            if isinstance(self.point_encoder, ConcertoPointEncoder):
                if not self.tune_point_encoder:
                    self.point_encoder.backbone.eval()
                if not self.tune_multimodal_adapter:
                    self.point_encoder.projection.eval()
            elif self.point_encoder is not None and not self.tune_point_encoder:
                self.point_encoder.eval()
            if self.tactile_encoder is not None and (
                not self.tune_tactile_encoder
                or (
                    isinstance(self.model, SidecarConditionedDiT)
                    and self.config.freeze_contact_teacher
                )
            ):
                self.tactile_encoder.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        # Qwen has its own CUDA BF16 compute boundary. The action head may be
        # FP32 at inference or BF16 under ZeRO; join at the DiT's actual weight
        # dtype rather than assuming the VLM and head are stored identically.
        head_dtype = (
            self.vlln.weight.dtype
            if isinstance(self.vlln, nn.LayerNorm)
            else self.state_encoder.layer1.W.dtype
        )
        backbone_features = backbone_features.to(dtype=head_dtype)
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _encode_multimodal_inputs(
        self, action_input: BatchFeature
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Encode static sensor observations once per action-head invocation."""
        if self.temporal_fusion is not None:
            if (
                "points" not in action_input
                or "tactile" not in action_input
                or "temporal_state" not in action_input
            ):
                raise KeyError(
                    "Temporal Fusion requires synchronized point, tactile and state history"
                )
            points = action_input.points
            tactile = action_input.tactile
            state = action_input.temporal_state
            window = self.config.temporal_window_size
            if (
                points.ndim != 4
                or points.shape[1] != window
                or tactile.ndim != 5
                or tactile.shape[1] != window + 1
            ):
                raise ValueError("Temporal observations have wrong history length")
            valid = action_input.get("temporal_valid")
            if valid is None:
                raise ValueError("Temporal validity mask is required")
            valid = valid.bool()
            environment_ids = action_input.get("temporal_environment_ids")
            use_ring = environment_ids is not None and not torch.is_grad_enabled()
            if use_ring:
                if len(environment_ids) != points.shape[0] or len(set(environment_ids)) != len(
                    environment_ids
                ):
                    raise ValueError("Temporal environment IDs must be unique within the batch")
                resets = action_input.get("temporal_reset")
                if resets is None:
                    resets = [False] * len(environment_ids)
                for environment_id, reset in zip(environment_ids, resets):
                    if reset:
                        self._temporal_ring.reset(environment_id)
                need_seed = any(not self._temporal_ring.has(eid) for eid in environment_ids)
                indices = range(window) if need_seed else (window - 1,)
            else:
                indices = range(window)
            point_frames, point_masks, tactile_frames = [], [], []
            for index in indices:
                encoded, mask = self.point_encoder(points[:, index], return_mask=True)
                point_frames.append(encoded)
                point_masks.append(mask)
                tactile_frames.append(self.tactile_encoder(tactile[:, index : index + 2]))
            if use_ring:
                from gr00t.model.extension.temporal_fusion import (
                    masked_feature_norm_mean,
                    masked_point_mean,
                )

                packed_frames = []
                for local_index, index in enumerate(indices):
                    pooled_point = masked_point_mean(
                        point_frames[local_index], point_masks[local_index]
                    )
                    pooled_tactile = tactile_frames[local_index].mean(1)
                    projected_state = self.temporal_fusion.state_projection(
                        state[:, index].to(self.temporal_fusion.state_projection.weight.dtype)
                    )
                    packed_frames.append(
                        torch.stack((pooled_point, pooled_tactile, projected_state), 1)
                    )
                for batch_index, environment_id in enumerate(environment_ids):
                    if not self._temporal_ring.has(environment_id):
                        for local_index, index in enumerate(indices):
                            if valid[batch_index, index]:
                                self._temporal_ring.push(
                                    environment_id, packed_frames[local_index][batch_index]
                                )
                    else:
                        self._temporal_ring.push(environment_id, packed_frames[-1][batch_index])
                windows = [
                    self._temporal_ring.window(eid, packed_frames[-1][i])
                    for i, eid in enumerate(environment_ids)
                ]
                packed = torch.stack([item[0] for item in windows])
                fused_mask = torch.stack([item[1] for item in windows])
                fused, fused_mask = self.temporal_fusion.forward_pooled(
                    packed[:, :, 0], packed[:, :, 1], packed[:, :, 2], fused_mask
                )
                self._temporal_feature_diagnostics = {
                    "features/point_norm": masked_feature_norm_mean(packed[:, :, 0], fused_mask),
                    "features/tactile_norm": masked_feature_norm_mean(packed[:, :, 1], fused_mask),
                    "features/fused_norm": masked_feature_norm_mean(fused, fused_mask),
                    "temporal/valid_frame_ratio": fused_mask.detach().float().mean(),
                }
            else:
                from gr00t.model.extension.temporal_fusion import (
                    masked_feature_norm_mean,
                    stack_padded_point_frames,
                )

                point_history, point_mask_history = stack_padded_point_frames(
                    point_frames, point_masks
                )
                tactile_history = torch.stack(tactile_frames, 1)
                fused, fused_mask = self.temporal_fusion(
                    point_history, tactile_history, state, valid, point_mask_history
                )
                self._temporal_feature_diagnostics = {
                    "features/point_norm": masked_feature_norm_mean(
                        point_history, point_mask_history & valid[..., None]
                    ),
                    "features/tactile_norm": masked_feature_norm_mean(tactile_history, valid),
                    "features/fused_norm": masked_feature_norm_mean(fused, fused_mask),
                    "temporal/valid_frame_ratio": valid.detach().float().mean(),
                }
            current_points = point_frames[-1]
            current_mask = point_masks[-1]
            return (
                torch.cat((current_points, fused.to(current_points.dtype)), 1),
                torch.cat((current_mask, fused_mask), 1),
                None,
            )
        point_tokens = None
        point_attention_mask = None
        if self.point_encoder is not None:
            if "points" not in action_input:
                raise KeyError("Point conditioning is enabled but action_input.points is missing")
            if self.config.sidecar_variant == "multilevel_local":
                point_tokens = self.point_encoder.forward_multilevel(
                    action_input.points, stage_names=self.config.sidecar_point_stage_names
                )
            else:
                point_tokens, point_attention_mask = self.point_encoder(
                    action_input.points,
                    return_mask=True,
                )

        tactile_tokens = None
        if self.tactile_encoder is not None and not (
            isinstance(self.model, SidecarConditionedDiT)
            and self.config.tactile_fusion_mode == "none"
            and self.config.contact_input_mode == "none"
        ):
            if "tactile" not in action_input:
                raise KeyError(
                    "Tactile conditioning is enabled but action_input.tactile is missing"
                )
            tactile_tokens = self.tactile_encoder(action_input.tactile)

        return point_tokens, point_attention_mask, tactile_tokens

    def forward(
        self, backbone_output: dict | BatchFeature, action_input: dict | BatchFeature
    ) -> dict:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]
                - points: optional [B, N, point_input_dim]
                - tactile: optional [B, T_tactile, C, H, W]

        Returns:
            Dict containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        # Keep the public BatchFeature convenience API inside this forward,
        # while exposing only plain dicts to DeepSpeed's module hooks.
        backbone_output = BatchFeature(data=backbone_output)
        action_input = BatchFeature(data=action_input)

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        contact_state_features = state_features
        prepared_multimodal = None
        contact_representation = None
        if (
            isinstance(self.model, SidecarConditionedDiT)
            and self.model.input_contact_adapter is not None
        ):
            prepared_multimodal = self._encode_multimodal_inputs(action_input)
            contact_representation = self.model.input_contact_adapter.prepare_contact(
                prepared_multimodal[2], action_input.state
            )

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        action_mask = _expand_action_mask(action_input.action_mask, actions)
        if action_mask is None:
            raise ValueError("action_mask is required during training")
        actions = actions * action_mask
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype) * action_mask
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = ((1 - t) * noise + t * actions) * action_mask
        velocity = (actions - noise) * action_mask
        action_token_mask = action_mask.any(dim=-1)

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs
        if contact_representation is not None:
            action_features = self.model.input_contact_adapter(
                action_features, contact_representation, action_input.state, action_token_mask
            )
        action_features = action_features * action_token_mask.unsqueeze(-1).to(
            dtype=action_features.dtype
        )

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        hidden_attention_mask = torch.cat(
            (
                torch.ones(state_features.shape[:2], device=device, dtype=torch.bool),
                action_token_mask,
            ),
            dim=1,
        )
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.dit_type == "alternate_vl_dit":
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                hidden_attention_mask=hidden_attention_mask,
            )
        elif self.config.dit_type in {"multimodal_conditioned_dit", "sidecar_conditioned_dit"}:
            point_tokens, point_attention_mask, tactile_tokens = (
                prepared_multimodal
                if prepared_multimodal is not None
                else self._encode_multimodal_inputs(action_input)
            )

            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                point_tokens=point_tokens,
                point_attention_mask=point_attention_mask,
                tactile_tokens=tactile_tokens,
                **(
                    {"robot_state": contact_state_features}
                    if isinstance(self.model, SidecarConditionedDiT)
                    else {}
                ),
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
                hidden_attention_mask=hidden_attention_mask,
                return_all_hidden_states=True,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                hidden_attention_mask=hidden_attention_mask,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :] * action_mask

        # Slice out only the action portion of pred and target.
        # Keep the DiT in its model dtype; accumulate the flow-matching error
        # and mask denominator in FP32 to avoid BF16 reduction/rounding loss.
        action_loss = F.mse_loss(pred_actions.float(), velocity.float(), reduction="none")
        action_loss = action_loss * action_mask.float()
        loss = action_loss.sum() / (action_mask.float().sum() + 1e-6)
        diagnostic_action_loss = loss.detach()
        if isinstance(self.model, SidecarConditionedDiT):
            if self.config.contact_predictor_warmup:
                loss = (
                    self.config.contact_prediction_loss_weight * self.model.contact_prediction_loss
                )
            else:
                loss = (
                    loss
                    + self.config.contact_prediction_loss_weight
                    * self.model.contact_prediction_loss
                )

        from gr00t.experiment.action_diagnostics import action_diagnostics

        is_sidecar = isinstance(self.model, SidecarConditionedDiT)
        diagnostics = action_diagnostics(
            pred_actions,
            velocity,
            action_mask,
            loss,
            diagnostic_action_loss,
            self.model.contact_prediction_loss if is_sidecar else loss.new_zeros(()),
            self.config.contact_prediction_loss_weight if is_sidecar else 0.0,
            is_sidecar and self.config.contact_predictor_warmup,
        )
        if is_sidecar and self.config.contact_prediction_loss_weight == 0:
            diagnostics.pop("loss/contact_raw", None)
            diagnostics.pop("loss/contact_weighted", None)
        if self.temporal_fusion is not None:
            diagnostics.update(self._temporal_feature_diagnostics)
        elif (
            isinstance(self.model, SidecarConditionedDiT)
            and self.model.input_contact_adapter is not None
        ):
            diagnostics["features/point_norm"] = point_tokens.detach().float().norm(dim=-1).mean()
            diagnostics["features/tactile_norm"] = (
                tactile_tokens.detach().float().norm(dim=-1).mean()
            )
            diagnostics["features/fused_norm"] = (
                action_features.detach().float().norm(dim=-1).mean()
            )
            metrics = self.model.input_contact_adapter.last_diagnostics
            for source, target in (
                ("contact_router_noop_probability", "moe/noop_probability"),
                ("contact_router_direct_probability", "moe/direct_probability"),
                ("contact_router_entropy", "moe/router_entropy"),
            ):
                if source in metrics:
                    diagnostics[target] = metrics[source]
        if not self._dtype_diagnostics_logged and os.environ.get(
            "GR00T_DTYPE_DEBUG", ""
        ).lower() in {"1", "true", "yes"}:
            logger.info(
                "GR00T action dtype: vlm=%s state=%s action=%s point=%s tactile=%s "
                "dit_input=%s predicted=%s target=%s loss=%s",
                vl_embeds.dtype,
                state_features.dtype,
                actions.dtype,
                getattr(point_tokens, "dtype", None)
                if self.config.dit_type in {"multimodal_conditioned_dit", "sidecar_conditioned_dit"}
                else None,
                getattr(tactile_tokens, "dtype", None)
                if self.config.dit_type in {"multimodal_conditioned_dit", "sidecar_conditioned_dit"}
                else None,
                sa_embs.dtype,
                pred_actions.dtype,
                velocity.dtype,
                loss.dtype,
            )
            self._dtype_diagnostics_logged = True

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "diagnostics": diagnostics,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)
        action_mask = _expand_action_mask(action_input.get("action_mask"), actions)
        action_token_mask = None
        hidden_attention_mask = None
        if action_mask is not None:
            actions = actions * action_mask
            vel_strength = vel_strength * action_mask
            action_token_mask = action_mask.any(dim=-1)
            hidden_attention_mask = torch.cat(
                (
                    torch.ones(state_features.shape[:2], device=device, dtype=torch.bool),
                    action_token_mask,
                ),
                dim=1,
            )

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        point_tokens = None
        point_attention_mask = None
        tactile_tokens = None
        if self.config.dit_type in {"multimodal_conditioned_dit", "sidecar_conditioned_dit"}:
            point_tokens, point_attention_mask, tactile_tokens = self._encode_multimodal_inputs(
                action_input
            )
        contact_representation = None
        if (
            isinstance(self.model, SidecarConditionedDiT)
            and self.model.input_contact_adapter is not None
        ):
            contact_representation = self.model.input_contact_adapter.prepare_contact(
                tactile_tokens, action_input.state
            )

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            if action_mask is not None:
                actions = actions * action_mask
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            if contact_representation is not None:
                action_features = self.model.input_contact_adapter(
                    action_features, contact_representation, action_input.state, action_token_mask
                )
            if action_token_mask is not None:
                action_features = action_features * action_token_mask.unsqueeze(-1).to(
                    dtype=action_features.dtype
                )

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.dit_type == "alternate_vl_dit":
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                    hidden_attention_mask=hidden_attention_mask,
                )
            elif self.config.dit_type in {"multimodal_conditioned_dit", "sidecar_conditioned_dit"}:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    point_tokens=point_tokens,
                    point_attention_mask=point_attention_mask,
                    tactile_tokens=tactile_tokens,
                    **(
                        {"robot_state": state_features}
                        if isinstance(self.model, SidecarConditionedDiT)
                        else {}
                    ),
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                    hidden_attention_mask=hidden_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    hidden_attention_mask=hidden_attention_mask,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]
            if action_mask is not None:
                pred_velocity = pred_velocity * action_mask

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength
            if action_mask is not None:
                actions = actions * action_mask

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        supplied = kwargs.get("config")
        if supplied is not None or "sidecar_variant" in kwargs:
            from copy import deepcopy

            from gr00t.model.extension.multilevel_local_sidecar import validate_checkpoint_variant

            loading = {
                k: kwargs[k]
                for k in ("cache_dir", "revision", "local_files_only", "token", "subfolder")
                if k in kwargs
            }
            original = Gr00tN1d7Config.from_pretrained(pretrained_model_name_or_path, **loading)
            requested = deepcopy(supplied if supplied is not None else original)
            for key in (
                "sidecar_variant",
                "sidecar_point_stage_names",
                "sidecar_local_patch_size",
                "sidecar_window_chunk_size",
            ):
                if key in kwargs:
                    setattr(requested, key, kwargs[key])
            validate_checkpoint_variant(original, requested)
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
        checkpoint_backbone_shapes: dict | None = None,
        checkpoint_bf16_shapes: dict | None = None,
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config
        self._dtype_diagnostics_logged = False
        log_model_load_memory("Gr00tN1d7 construction start")

        backbone_cls = get_backbone_cls(config)
        backbone_loading_kwargs = {}
        if checkpoint_backbone_shapes is not None:
            backbone_loading_kwargs["checkpoint_backbone_shapes"] = checkpoint_backbone_shapes
        self.backbone = backbone_cls(
            **backbone_loading_kwargs,
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            use_lora=config.use_lora,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_bias=config.lora_bias,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        log_model_load_memory("after GR00T VLM backbone construction")

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        log_model_load_memory("after GR00T action-head construction")
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        if checkpoint_bf16_shapes:
            from .pretrained_loading import prepare_bf16_checkpoint_parameters

            prepare_bf16_checkpoint_parameters(self, checkpoint_bf16_shapes)
        log_model_load_memory("Gr00tN1d7 construction complete")

    def enable_lora(
        self,
        *,
        r: int,
        alpha: int,
        dropout: float,
        bias: str,
    ) -> list[str]:
        """Attach backbone-only LoRA after a non-LoRA GR00T checkpoint is loaded."""
        lora_names = self.backbone.enable_lora(
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=bias,
        )
        self.config.use_lora = True
        self.config.lora_r = r
        self.config.lora_alpha = alpha
        self.config.lora_dropout = dropout
        self.config.lora_bias = bias
        return lora_names

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move raw fields without applying one global parameter's dtype to all
        # modalities. Each consumer below owns the precision of its inputs.
        def to_model_device(x):
            return x.to(self.device)

        backbone_inputs = tree.map_structure(to_model_device, backbone_inputs)
        dense_head_dtype = self.action_head.state_encoder.layer1.W.dtype
        point_encoder_cfg = self.config.point_encoder_cfg.lower().replace("-", "_")
        tactile_encoder_cfg = self.config.tactile_encoder_cfg.lower().replace("-", "_")
        field_dtypes = {
            "state": dense_head_dtype,
            "action": self.action_head.action_encoder.W1.W.dtype,
            "points": (
                torch.float32
                if point_encoder_cfg in {"concerto_small", "concerto_base"}
                else dense_head_dtype
            ),
            "tactile": (
                torch.float32 if tactile_encoder_cfg == "sparsh_dino_base" else dense_head_dtype
            ),
        }

        def move_action_field(key, value):
            def move(x):
                if torch.is_floating_point(x) and key in field_dtypes:
                    return x.to(self.device, dtype=field_dtypes[key])
                return x.to(self.device)

            return tree.map_structure(move, value)

        action_inputs = BatchFeature(
            data={key: move_action_field(key, value) for key, value in action_inputs.items()}
        )

        if not self._dtype_diagnostics_logged and os.environ.get(
            "GR00T_DTYPE_DEBUG", ""
        ).lower() in {"1", "true", "yes"}:
            logger.info(
                "GR00T input dtype: model=%s qwen_pixels=%s state=%s action=%s "
                "points=%s tactile=%s cuda_amp=%s cuda_amp_dtype=%s",
                self.dtype,
                getattr(backbone_inputs.get("pixel_values"), "dtype", None),
                getattr(action_inputs.get("state"), "dtype", None),
                getattr(action_inputs.get("action"), "dtype", None),
                getattr(action_inputs.get("points"), "dtype", None),
                getattr(action_inputs.get("tactile"), "dtype", None),
                torch.is_autocast_enabled("cuda"),
                torch.get_autocast_dtype("cuda"),
            )
            self._dtype_diagnostics_logged = True

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> dict:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            Dict containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(dict(backbone_inputs))
        action_outputs = self.action_head(dict(backbone_outputs), dict(action_inputs))

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(dict(backbone_inputs))
        action_outputs = self.action_head.get_action(
            BatchFeature(data=backbone_outputs), action_inputs, options
        )

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
