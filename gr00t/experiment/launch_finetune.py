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

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path
import sys

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    # ``bool | str`` lets callers select either the latest checkpoint or an
    # explicit path. Preserve the legacy bare boolean flag by supplying the
    # value Tyro requires for union-typed options.
    cli_args = list(sys.argv[1:])
    for index, arg in enumerate(cli_args):
        if arg in {"--resume-from-checkpoint", "--resume_from_checkpoint"} and (
            index + 1 == len(cli_args) or cli_args[index + 1].startswith("--")
        ):
            cli_args.insert(index + 1, "True")
            break
    ft_config = tyro.cli(FinetuneConfig, description=__doc__, args=cli_args)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    tactile_modality = MODALITY_CONFIGS.get(embodiment_tag, {}).get("tactile")
    tactile_delta_indices = (
        list(tactile_modality.delta_indices) if tactile_modality is not None else [-1, 0]
    )
    if (
        ft_config.dit_type in {"multimodal_conditioned_dit", "sidecar_conditioned_dit"}
        and ft_config.use_tactile_conditioning
        and ft_config.tactile_encoder_cfg == "sparsh_dino_base"
        and (
            len(tactile_delta_indices) != 2
            or tactile_delta_indices[-1] != 0
            or tactile_delta_indices[0] >= 0
        )
    ):
        raise ValueError(
            "Sparsh-DINO requires two chronological tactile frames [previous,current]; "
            "configure tactile delta_indices=[negative_offset,0], got "
            f"{tactile_delta_indices}."
        )

    dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None
    config.data.seed = ft_config.seed

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.use_lora = ft_config.use_lora
    config.model.lora_r = ft_config.lora_r
    config.model.lora_alpha = ft_config.lora_alpha
    config.model.lora_dropout = ft_config.lora_dropout
    config.model.lora_bias = ft_config.lora_bias
    config.model.dit_type = ft_config.dit_type
    config.model.use_point_conditioning = ft_config.use_point_conditioning
    config.model.use_tactile_conditioning = ft_config.use_tactile_conditioning
    config.model.point_input_dim = ft_config.point_input_dim
    config.model.tactile_input_channels = ft_config.tactile_input_channels
    config.model.tactile_encoder_cfg = ft_config.tactile_encoder_cfg
    config.model.tactile_pretrained_model = ft_config.tactile_pretrained_model
    config.model.tactile_checkpoint_filename = ft_config.tactile_checkpoint_filename
    config.model.tactile_background_path = ft_config.tactile_background_path
    config.model.tactile_pretrained_load_on_init = True
    config.model.tactile_temporal_delta_indices = tactile_delta_indices
    config.model.point_encoder_cfg = ft_config.point_encoder_cfg
    config.model.point_encoder_checkpoint_path = ft_config.point_encoder_checkpoint_path
    config.model.point_encoder_repo_id = ft_config.point_encoder_repo_id
    config.model.point_encoder_download_root = ft_config.point_encoder_download_root
    config.model.point_encoder_grid_size = ft_config.point_encoder_grid_size
    config.model.point_encoder_enable_flash = ft_config.point_encoder_enable_flash
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.tune_vlln = ft_config.tune_vlln
    config.model.tune_point_encoder = ft_config.tune_point_encoder
    config.model.tune_tactile_encoder = ft_config.tune_tactile_encoder
    config.model.sidecar_variant = ft_config.sidecar_variant
    config.model.sidecar_point_stage_names = ft_config.sidecar_point_stage_names
    config.model.sidecar_local_patch_size = ft_config.sidecar_local_patch_size
    config.model.sidecar_window_chunk_size = ft_config.sidecar_window_chunk_size
    config.model.sidecar_num_stages = ft_config.sidecar_num_stages
    config.model.sidecar_stage_indices = ft_config.sidecar_stage_indices
    config.model.sidecar_hidden_dim = ft_config.sidecar_hidden_dim
    config.model.sidecar_num_heads = ft_config.sidecar_num_heads
    config.model.sidecar_num_point_tokens = ft_config.sidecar_num_point_tokens
    config.model.sidecar_point_gate_init = ft_config.sidecar_point_gate_init
    config.model.sidecar_action_gate_init = ft_config.sidecar_action_gate_init
    config.model.sidecar_gate_init_seed = ft_config.sidecar_gate_init_seed
    config.model.sidecar_gate_mode = ft_config.sidecar_gate_mode
    config.model.sidecar_adaptive_gate_hidden_dim = ft_config.sidecar_adaptive_gate_hidden_dim
    config.model.sidecar_adaptive_gate_init_action = ft_config.sidecar_adaptive_gate_init_action
    config.model.sidecar_adaptive_gate_init_point = ft_config.sidecar_adaptive_gate_init_point
    config.model.tactile_fusion_mode = ft_config.tactile_fusion_mode
    config.model.contact_num_tokens = ft_config.contact_num_tokens
    config.model.contact_hidden_dim = ft_config.contact_hidden_dim
    config.model.contact_prediction_loss_weight = ft_config.contact_prediction_loss_weight
    config.model.freeze_pretrained_dit = ft_config.freeze_pretrained_dit
    config.model.freeze_contact_teacher = ft_config.freeze_contact_teacher
    config.model.contact_injection_enabled = ft_config.contact_injection_enabled
    config.model.contact_predictor_warmup = ft_config.contact_predictor_warmup
    config.model.tune_multimodal_adapter = ft_config.tune_multimodal_adapter
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = (
        None if ft_config.disable_color_jitter else ft_config.color_jitter_params
    )
    config.model.use_percentiles = ft_config.use_percentiles
    if (ft_config.shortest_image_edge is None) != (ft_config.crop_fraction is None):
        raise ValueError("shortest_image_edge and crop_fraction must be set together")
    if ft_config.shortest_image_edge is not None:
        config.model.shortest_image_edge = ft_config.shortest_image_edge
        config.model.crop_fraction = ft_config.crop_fraction
        config.model.image_crop_size = None
        config.model.image_target_size = None
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    if ft_config.action_head_dropout is not None:
        config.model.action_head_dropout_override = ft_config.action_head_dropout
    if ft_config.vl_self_attention_dropout is not None:
        config.model.vl_self_attention_dropout_override = ft_config.vl_self_attention_dropout

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch_fused"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.vlm_learning_rate = ft_config.vlm_learning_rate
    config.training.action_head_learning_rate = ft_config.action_head_learning_rate
    config.training.point_encoder_learning_rate = ft_config.point_encoder_learning_rate
    config.training.tactile_encoder_learning_rate = ft_config.tactile_encoder_learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.deepspeed_stage = ft_config.deepspeed_stage
    config.training.zero3_offload_optimizer = ft_config.zero3_offload_optimizer
    config.training.zero3_offload_param = ft_config.zero3_offload_param
    config.training.zero3_sub_group_size = ft_config.zero3_sub_group_size
    config.training.gradient_checkpointing = ft_config.gradient_checkpointing
    config.training.use_wandb = ft_config.use_wandb
    config.training.telegram_on = ft_config.telegram_on
    config.training.telegram_chat_id = ft_config.telegram_chat_id
    config.training.telegram_notify_start = ft_config.telegram_notify_start
    config.training.telegram_notify_save = ft_config.telegram_notify_save
    config.training.telegram_notify_finish = ft_config.telegram_notify_finish
    config.training.telegram_notify_error = ft_config.telegram_notify_error
    config.training.discord_on = ft_config.discord_on
    config.training.discord_notify_start = ft_config.discord_notify_start
    config.training.discord_notify_save = ft_config.discord_notify_save
    config.training.discord_notify_finish = ft_config.discord_notify_finish
    config.training.discord_notify_error = ft_config.discord_notify_error
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.allow_padding = ft_config.allow_padding
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.ds_weights_alpha = ft_config.ds_weights_alpha

    config.training.save_only_model = ft_config.save_only_model
    config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
    config.training.skip_weight_loading = ft_config.skip_weight_loading
    config.training.enable_training_dashboard = ft_config.enable_training_dashboard
    config.training.dashboard_refresh_seconds = ft_config.dashboard_refresh_seconds
    config.training.enable_multimodal_diagnostics = ft_config.enable_multimodal_diagnostics
    config.training.enable_gate_gradient_diagnostics = ft_config.enable_gate_gradient_diagnostics
    config.training.multimodal_diagnostics_interval = ft_config.multimodal_diagnostics_interval
    config.training.multimodal_diagnostics_level = ft_config.multimodal_diagnostics_level

    print(f"DeepSpeed stage : {config.training.deepspeed_stage}")
    run(config)
