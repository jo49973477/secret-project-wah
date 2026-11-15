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

# Finetune config used for single node post-training.
from dataclasses import dataclass, field
import warnings


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: str
    """Path to one dataset root, or an os.pathsep-separated list of dataset roots."""

    embodiment_tag: str
    """Embodiment tag (name or value, case-insensitive). See EmbodimentTag for known tags."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment. 
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`. 
    """

    seed: int = 42
    """Seed used for model initialization, data sampling, and worker RNGs."""

    dit_type: str = "alternate_vl_dit"
    """Action-head DiT implementation; use multimodal_conditioned_dit for sensor inputs."""

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
    use_point_conditioning: bool = True
    """Enable point-cloud conditioning when using multimodal_conditioned_dit."""

    use_tactile_conditioning: bool = True
    """Enable tactile conditioning when using multimodal_conditioned_dit."""

    point_input_dim: int = 6
    """Number of values per input point (3 for XYZ, 6 for XYZRGB)."""

    tactile_input_channels: int = 3
    """Number of tactile image input channels."""

    tactile_encoder_cfg: str = "sparsh_dino_base"
    """Tactile encoder backend: pretrained sparsh_dino_base (default) or resnet18."""

    tactile_pretrained_model: str = "facebook/sparsh-dino-base"
    """Hugging Face model ID, local checkpoint file, or local model directory."""

    tactile_checkpoint_filename: str = "dino_vitbase.safetensors"
    """Exact checkpoint filename within the Hugging Face repository/local directory."""

    tactile_background_path: str | None = None
    """Optional no-contact RGB background used by official Sparsh subtraction."""

    point_encoder_cfg: str = "concerto_small"
    """Point encoder backend used by multimodal_conditioned_dit."""

    point_encoder_checkpoint_path: str | None = None
    """Optional local official Concerto .pth file; otherwise use Pointcept/Concerto."""

    point_encoder_repo_id: str = "Pointcept/Concerto"
    """Hugging Face repository used by the official Concerto loader."""

    point_encoder_download_root: str | None = None
    """Optional Concerto checkpoint cache directory."""

    point_encoder_grid_size: float = 0.02
    """Concerto serialization grid size in metres."""

    point_encoder_enable_flash: bool | None = None
    """Override Concerto FlashAttention use; None preserves checkpoint configuration."""

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    use_lora: bool = False
    """If True, freeze the Qwen3-VL base weights and train LoRA adapters on its linear layers."""

    lora_r: int = 16
    """Rank of the Qwen3-VL LoRA adapters."""

    lora_alpha: int = 32
    """LoRA scaling alpha."""

    lora_dropout: float = 0.0
    """Dropout probability applied on LoRA adapter inputs."""

    lora_bias: str = "none"
    """PEFT LoRA bias mode. Must remain ``none`` to keep all backbone base weights frozen."""

    tune_vlln: bool = True
    """If True, fine-tune the VLM feature normalization/self-attention projection."""

    tune_point_encoder: bool = True
    """If True, fine-tune the optional point-cloud encoder."""

    tune_tactile_encoder: bool = True
    """If True, fine-tune the optional tactile image encoder."""

    tune_multimodal_adapter: bool = True
    """If True, fine-tune point/tactile cross-attention branches and residual gates."""

    state_dropout_prob: float = 0.2
    """
    Per-sample state-feature dropout applied once after the action-head state
    encoder during training. Zero disables it. Older processor configs may
    contain this field, but preprocessing no longer applies a second dropout.
    """

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = field(
        default_factory=lambda: {
            "brightness": 0.3,
            "contrast": 0.4,
            "saturation": 0.5,
            "hue": 0.08,
        }
    )
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, color jitter is disabled. The default preserves the normal fine-tuning augmentation.
    """

    disable_color_jitter: bool = False
    """Explicitly disable color jitter, including a value saved in the pretrained processor."""

    use_percentiles: bool = True
    """
    If True, use q01/q99 percentile statistics for state/action min-max normalization.
    If False, use full min/max statistics.
    """

    shortest_image_edge: int | None = 256
    """
    Resize images so the shortest edge has this size before fractional cropping.
    If set, crop_fraction must also be set and legacy image_crop_size/image_target_size
    preprocessing is disabled.
    """

    crop_fraction: float | None = 0.95
    """
    Fraction of the resized image retained by the random/center crop.
    If set, shortest_image_edge must also be set and legacy image_crop_size/image_target_size
    preprocessing is disabled.
    """

    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total batch summed across all GPUs in one forward/backward, BEFORE
    gradient accumulation."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Default optimizer learning rate, used by either split LR when it is omitted."""

    vlm_learning_rate: float | None = None
    """Optional LR for trainable VLM parameters, including LoRA. Falls back to learning_rate."""

    action_head_learning_rate: float | None = None
    """Optional LR for GR00T action-head parameters. Falls back to learning_rate."""

    point_encoder_learning_rate: float | None = 1e-5
    """LR for a trainable pretrained Concerto backbone; projection/adapters use action-head LR."""

    tactile_encoder_learning_rate: float | None = 1e-5
    """LR for a trainable pretrained Sparsh backbone; projection/adapters use action-head LR."""

    action_head_dropout: float | None = None
    """Optional DiT/action-head dropout override. None preserves the normal model setting."""

    vl_self_attention_dropout: float | None = None
    """Optional VL self-attention dropout override. None preserves the normal model setting."""

    gradient_accumulation_steps: int = 1
    """Forward passes per optimizer step. Multiplies ``global_batch_size`` to
    produce the post-accumulation per-optimizer-step batch."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    experiment_name: str | None = None
    """Optional experiment name used as the W&B run name. Defaults to the output directory basename."""

    wandb_project: str = "finetune-gr00t-n1d7"
    """W&B project name to log runs to."""

    save_steps: int = 100
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    deepspeed_stage: int = 3
    """DeepSpeed ZeRO stage: supports 2 and 3 (default: 3).
    ZeRO-2 replicates parameters and shards gradients/optimizer states.
    ZeRO-3 also shards parameters.
    """

    zero3_offload_optimizer: bool = False
    """Offload ZeRO-3 optimizer states and AdamW computation to CPU."""

    zero3_offload_param: bool = False
    """Offload ZeRO-3 parameter partitions to CPU independently of optimizer offload."""

    zero3_sub_group_size: int | None = None
    """Optional ZeRO-3 processing tile size. None preserves DeepSpeed's existing default."""

    gradient_checkpointing: bool = False
    """Enable supported activation checkpointing paths; disabled by default."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d7`.
    You need to login to wandb to view the logs.
    """

    telegram_on: bool = False
    """Enable best-effort Telegram training notifications."""

    telegram_chat_id: str | None = None
    """Optional chat ID override. Defaults to the TELEGRAM_CHAT_ID environment variable."""

    telegram_notify_start: bool = True
    """Send a notification when Hugging Face Trainer begins training."""

    telegram_notify_save: bool = True
    """Send a notification after each regular Hugging Face checkpoint save."""

    telegram_notify_finish: bool = True
    """Send a notification after training and the final model save succeed."""

    telegram_notify_error: bool = True
    """Send a notification for catchable training/finalization exceptions."""

    discord_on: bool = False
    """Enable best-effort Discord webhook training notifications."""

    discord_notify_start: bool = True
    """Send a Discord notification when Hugging Face Trainer begins training."""

    discord_notify_save: bool = True
    """Send a Discord notification after each regular Hugging Face checkpoint save."""

    discord_notify_finish: bool = True
    """Send a Discord notification after training and the final model save succeed."""

    discord_notify_error: bool = True
    """Send a Discord notification for catchable training/finalization exceptions."""

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    ds_weights_alpha: float | None = None
    """Power-law exponent for dataset soup weighting. When set, each dataset's
    sampling weight is len(dataset)^alpha and per-dataset mix_ratio values are ignored."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    allow_padding: bool = False
    """Clamp out-of-range delta indices to the current episode boundary."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    save_only_model: bool = False
    """If True, save only model weights (skip optimizer/scheduler/RNG states). Cannot resume training from these checkpoints."""

    resume_from_checkpoint: bool | str = False
    """If True, resume from the latest ``checkpoint-*`` in ``output_dir``. A
    string selects an explicit checkpoint directory. Default False so a rerun
    against an existing ``output_dir`` starts fresh instead of silently merging
    with a previous experiment. Incompatible with ``save_only_model=True``
    (enforced by ``experiment.run``)."""

    skip_weight_loading: bool = False
    """If True, skip loading model weights from base_model_path (architecture only).
    The processor (tokenizer/config) is still loaded from base_model_path.
    Useful for CI/testing to skip the slow checkpoint shard loading."""

    enable_training_dashboard: bool | None = None
    dashboard_refresh_seconds: float = 1.0
    enable_multimodal_diagnostics: bool = False
    multimodal_diagnostics_interval: int = 50
    multimodal_diagnostics_level: str = "basic"

    @property
    def resolved_vlm_learning_rate(self) -> float:
        return self.learning_rate if self.vlm_learning_rate is None else self.vlm_learning_rate

    @property
    def resolved_action_head_learning_rate(self) -> float:
        return (
            self.learning_rate
            if self.action_head_learning_rate is None
            else self.action_head_learning_rate
        )

    @property
    def resolved_point_encoder_learning_rate(self) -> float:
        return (
            self.resolved_action_head_learning_rate
            if self.point_encoder_learning_rate is None
            else self.point_encoder_learning_rate
        )

    @property
    def resolved_tactile_encoder_learning_rate(self) -> float:
        return (
            self.resolved_action_head_learning_rate
            if self.tactile_encoder_learning_rate is None
            else self.tactile_encoder_learning_rate
        )

    def __post_init__(self) -> None:
        from gr00t.experiment.discord_notifier import validate_discord_configuration
        from gr00t.experiment.telegram_notifier import validate_telegram_configuration

        validate_telegram_configuration(
            enabled=self.telegram_on,
            chat_id=self.telegram_chat_id,
        )
        validate_discord_configuration(enabled=self.discord_on)
        if self.dashboard_refresh_seconds <= 0:
            raise ValueError("dashboard_refresh_seconds must be positive")
        if self.multimodal_diagnostics_interval < 1:
            raise ValueError("multimodal_diagnostics_interval must be positive")
        if self.multimodal_diagnostics_level not in {"off", "basic", "per_layer"}:
            raise ValueError("multimodal_diagnostics_level must be off, basic, or per_layer")
        if self.deepspeed_stage not in (2, 3):
            raise ValueError(f"deepspeed_stage must be 2 or 3, got {self.deepspeed_stage}")
        if self.zero3_sub_group_size is not None and self.zero3_sub_group_size <= 0:
            raise ValueError(
                "zero3_sub_group_size must be positive when provided, got "
                f"{self.zero3_sub_group_size}"
            )
        if self.deepspeed_stage != 3 and (
            self.zero3_offload_optimizer
            or self.zero3_offload_param
            or self.zero3_sub_group_size is not None
        ):
            raise ValueError("ZeRO-3 memory options require deepspeed_stage=3")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.resolved_vlm_learning_rate <= 0:
            raise ValueError(
                "resolved vlm_learning_rate must be positive, got "
                f"{self.resolved_vlm_learning_rate}"
            )
        if self.resolved_action_head_learning_rate <= 0:
            raise ValueError(
                "resolved action_head_learning_rate must be positive, got "
                f"{self.resolved_action_head_learning_rate}"
            )
        if self.resolved_point_encoder_learning_rate <= 0:
            raise ValueError(
                "resolved point_encoder_learning_rate must be positive, got "
                f"{self.resolved_point_encoder_learning_rate}"
            )
        if self.resolved_tactile_encoder_learning_rate <= 0:
            raise ValueError(
                "resolved tactile_encoder_learning_rate must be positive, got "
                f"{self.resolved_tactile_encoder_learning_rate}"
            )
        for name, value in (
            ("action_head_dropout", self.action_head_dropout),
            ("vl_self_attention_dropout", self.vl_self_attention_dropout),
        ):
            if value is not None and not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")
        if self.use_lora and (self.tune_llm or self.tune_visual):
            raise ValueError("use_lora cannot be combined with tune_llm or tune_visual")
        if self.use_lora and (
            not self.tune_projector or not self.tune_diffusion_model or not self.tune_vlln
        ):
            raise ValueError(
                "use_lora requires tune_projector, tune_diffusion_model, and tune_vlln so "
                "the complete GR00T action head remains trainable"
            )
        if self.lora_r < 1:
            raise ValueError(f"lora_r must be >= 1, got {self.lora_r}")
        if self.lora_alpha < 1:
            raise ValueError(f"lora_alpha must be >= 1, got {self.lora_alpha}")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")
        if self.lora_bias != "none":
            raise ValueError(
                "lora_bias must be 'none': training backbone biases would violate the "
                "frozen-backbone LoRA strategy"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )
        if self.gradient_accumulation_steps > 1:
            accumulated_batch_size = self.global_batch_size * self.gradient_accumulation_steps
            warnings.warn(
                f"global_batch_size={self.global_batch_size} is pre-accumulation; "
                f"accumulated_batch_size={accumulated_batch_size} "
                f"(× gradient_accumulation_steps={self.gradient_accumulation_steps}).",
                stacklevel=2,
            )
