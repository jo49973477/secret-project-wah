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

from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path

import torch
from transformers import PretrainedConfig

from . import register_model_config


@dataclass
class Gr00tN1d7Config(PretrainedConfig):
    """Unified configuration for Gr00tN1d7 model with backbone and action head.

    Gr00tN1d7 uses the Cosmos-Reason2-2B (Qwen3-VL architecture) VLM backbone,
    replacing the Eagle backbone used in Gr00tN1d6.
    """

    # Model identification
    model_type: str = "Gr00tN1d7"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    optimize_pretrained_loading: bool = True

    # Backbone configuration
    model_name: str = "nvidia/Cosmos-Reason2-2B"
    backbone_model_type: str = "qwen"
    model_revision: str | None = None
    tune_top_llm_layers: int = 0  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 2048  # project_to_dim; must match Cosmos-Reason2-2B hidden size
    tune_llm: bool = False
    tune_visual: bool = False
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    select_layer: int = 12
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = False  # Enable BF16 loading
    backbone_trainable_params_fp32: bool = True

    ### Processing parameters
    image_crop_size: tuple[int, int] | None = (230, 230)
    image_target_size: tuple[int, int] | None = (256, 256)

    shortest_image_edge: int | None = None
    crop_fraction: float | None = None

    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_albumentations_transforms: bool = True
    letter_box_transform: bool = False
    # Extra augmentation config (mask-based and others).
    extra_augmentation_config: dict | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    use_percentiles: bool = True
    use_relative_action: bool = False

    # Action head configuration parameters
    max_state_dim: int = 132  # Default from state_shape
    max_action_dim: int = 132  # Default from action_shape
    action_horizon: int = 40
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # State history: number of consecutive state timesteps fed to the state encoder
    state_history_length: int = 1

    # Global parameters
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    vl_self_attention_cfg: dict | None = None
    action_head_dropout_override: float | None = None
    vl_self_attention_dropout_override: float | None = None
    use_vlln: bool = True
    max_seq_len: int = 1024
    use_alternate_vl_dit: bool = True  # Legacy selector retained for checkpoint compatibility.
    attend_text_every_n_blocks: int = 2

    # Optional action-head sensor conditioning.
    dit_type: str = (
        "alternate_vl_dit"  # candidates : ["alternate_vl_dit", "multimodal_conditioned_dit", "dit"]
    )
    sidecar_variant: str = "legacy"
    sidecar_point_stage_names: list[str] | None = None
    sidecar_local_patch_size: int = 64
    sidecar_window_chunk_size: int = 16
    sidecar_num_stages: int = 4
    sidecar_stage_indices: list[int] | None = None
    sidecar_hidden_dim: int = 256
    sidecar_num_heads: int = 4
    sidecar_num_point_tokens: int = 32
    tactile_fusion_mode: str = "surprise"
    contact_num_tokens: int = 4
    contact_hidden_dim: int = 256
    contact_prediction_loss_weight: float = 0.1
    freeze_pretrained_dit: bool = True
    freeze_contact_teacher: bool = True
    contact_injection_enabled: bool = True
    contact_predictor_warmup: bool = False
    dit_training_mode: str = "frozen"
    dit_lora_rank: int = 16
    dit_lora_alpha: float = 32.0
    dit_lora_dropout: float = 0.0
    temporal_window_size: int = 0
    contact_input_mode: str = "none"
    use_point_conditioning: bool = True
    use_tactile_conditioning: bool = True
    point_input_dim: int = 3
    tactile_input_channels: int = 3
    tactile_encoder_cfg: str = "sparsh_dino_base"
    tactile_pretrained_model: str = "facebook/sparsh-dino-base"
    tactile_checkpoint_filename: str = "dino_vitbase.safetensors"
    tactile_background_path: str | None = None
    tactile_pretrained_load_on_init: bool = True
    tactile_encoder_deferred_bootstrap: bool = False
    tactile_temporal_delta_indices: list[int] = field(default_factory=lambda: [-1, 0])
    point_encoder_cfg: str = "pointnet2"  # candidates: ["pointnet2", "pointnet", "point_transformer", "pointnet++", "transformer"]
    point_encoder_pretrained_load_on_init: bool = True
    point_encoder_deferred_bootstrap: bool = False
    point_encoder_model_config: dict | None = None
    point_encoder_checkpoint_path: str | None = None
    point_encoder_repo_id: str = "Pointcept/Concerto"
    point_encoder_download_root: str | None = None
    point_encoder_grid_size: float = 0.02
    point_encoder_enable_flash: bool | None = None

    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 16,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True
    tune_point_encoder: bool = True
    tune_tactile_encoder: bool = True
    tune_multimodal_adapter: bool = True

    # State augmentation parameters
    state_dropout_prob: float = 0.8  # State dropout probability
    exclude_state: bool = False  # Zero out all state inputs (ablation)
    use_mean_std: bool = False  # Use mean/std normalization instead of min/max

    # Multi-embodiment parameters
    max_num_embodiments: int = 32

    def __init__(self, **kwargs):
        # Honor the legacy selector when loading checkpoints without the explicit DiT type.
        if "dit_type" not in kwargs:
            kwargs["dit_type"] = (
                "alternate_vl_dit" if kwargs.get("use_alternate_vl_dit", True) else "dit"
            )
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict:
        """Return a dictionary representation of this config, optionally excluding augmentation keys."""
        if is_dataclass(self):
            cfg = asdict(self)
        else:
            cfg = dict(self.__dict__)

        if exclude_augment:
            exclude_keys = {
                "random_rotation_angle",
                "color_jitter_params",
                "use_albumentations_transforms",
                "formalize_language",
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
            }
            cfg = {k: v for k, v in cfg.items() if k not in exclude_keys}

        return cfg

    def to_filtered_json(self, exclude_augment: bool = True, **kwargs) -> str:
        """Return a JSON string of this config, optionally excluding augmentation keys."""

        def default(o):
            if isinstance(o, (Path, torch.dtype, torch.device)):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            return str(o)

        return json.dumps(
            self.to_filtered_dict(exclude_augment),
            indent=2,
            default=default,
            **kwargs,
        )


register_model_config("Gr00tN1d7", Gr00tN1d7Config)
