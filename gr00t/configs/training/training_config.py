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

from dataclasses import dataclass, field
from typing import Optional
import warnings


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Output
    output_dir: str = "./outputs"
    experiment_name: Optional[str] = None

    # Basic training
    max_steps: int = 30000  # this will override num_epochs

    global_batch_size: int = 1024
    """Total batch summed across all GPUs in one forward/backward, BEFORE
    gradient accumulation. See :attr:`accumulated_batch_size` for the
    post-accumulation per-optimizer-step value."""

    per_gpu_batch_size: Optional[int] = None
    """Deprecated. When set, overrides ``global_batch_size`` as the per-GPU batch."""

    gradient_accumulation_steps: int = 1
    """Forward passes per optimizer step. Passed through to HuggingFace
    ``TrainingArguments``; see :attr:`accumulated_batch_size`."""

    # Optimization
    learning_rate: float = 1e-4
    vlm_learning_rate: float | None = None
    action_head_learning_rate: float | None = None
    dit_learning_rate: float | None = None
    point_encoder_learning_rate: float | None = None
    tactile_encoder_learning_rate: float | None = None
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    warmup_steps: int = 0  # this will override warmup_ratio
    max_grad_norm: float = 1.0

    # Optimizer choice (huggingface TrainingArguments.optim)
    # Options include: 'adamw_torch', 'adamw_torch_fused', 'paged_adamw_32bit',
    # 'paged_adamw_8bit' (requires bitsandbytes), 'adafactor', etc.
    optim: str = "adamw_torch_fused"

    start_from_checkpoint: Optional[str] = None
    skip_weight_loading: bool = False  # skip loading checkpoint weights (architecture only)

    # Mixed precision
    tf32: bool = True
    fp16: bool = False
    bf16: bool = True
    eval_bf16: bool = True

    # Logging and saving
    logging_steps: int = 10
    enable_training_dashboard: bool | None = None
    """Render an in-place terminal dashboard; None enables it only on a TTY."""
    dashboard_refresh_seconds: float = 1.0
    enable_multimodal_diagnostics: bool = False
    enable_gate_gradient_diagnostics: bool = False
    multimodal_diagnostics_interval: int = 50
    multimodal_diagnostics_level: str = "basic"
    save_steps: int = 100
    save_total_limit: int = 5

    # Model saving
    save_vl_model: bool = False  # Control whether to save VL model and processor in callbacks
    save_only_model: bool = False  # Skip optimizer/scheduler/RNG states — cannot resume training

    # True selects the latest checkpoint; a string selects an explicit path.
    resume_from_checkpoint: bool | str = False

    # Checkpoint uploading
    upload_checkpoints: bool = False
    upload_every: int = 1000
    upload_last_n_checkpoints: int = 5
    max_concurrent_uploads: int = 2

    # Evaluation
    eval_strategy: str = "no"  # no, steps, epoch
    eval_steps: int = 500
    eval_set_split_ratio: float = 0.1
    eval_batch_size: int = 2
    save_best_eval_metric_name: str = ""
    save_best_eval_metric_greater_is_better: bool = True

    # DeepSpeed (default)
    deepspeed_stage: int = 3  # ZeRO stage (2 or 3)
    zero3_offload_optimizer: bool = False
    zero3_offload_param: bool = False
    zero3_sub_group_size: int | None = None
    gradient_checkpointing: bool = False

    # Transformers loading parameters
    transformers_trust_remote_code: bool = True
    transformers_local_files_only: bool = False
    transformers_cache_dir: str | None = None
    transformers_access_token: str | None = None  # Access token for HuggingFace Hub

    # DDP
    use_ddp: bool = False
    ddp_bucket_cap_mb: int = 100

    # Hardware
    num_gpus: int = 1
    dataloader_num_workers: int = 2

    # Data handling
    remove_unused_columns: bool = False

    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "finetune-gr00t-n1d7"
    wandb_group: str | None = None
    wandb_name: str | None = None

    # Deprecated legacy notifications (non-Hydra entrypoints only).
    # Hydra uses the top-level trainctl configuration and ignores these flags.
    telegram_on: bool = False
    telegram_chat_id: str | None = None
    telegram_notify_start: bool = True
    telegram_notify_save: bool = True
    telegram_notify_finish: bool = True
    telegram_notify_error: bool = True

    # Discord notifications. The webhook URL is intentionally never part of config.
    discord_on: bool = False
    discord_notify_start: bool = True
    discord_notify_save: bool = True
    discord_notify_finish: bool = True
    discord_notify_error: bool = True

    # Profiling
    enable_profiling: bool = False

    # Max number of retries in training for fault tolerance
    max_retries: int = 3

    # For testing.
    assert_loss_less_than: float | None = None

    # RL
    add_rl_callback: bool = False

    # Open-loop evaluation
    enable_open_loop_eval: bool = False
    """Enable open-loop evaluation on saved checkpoints."""

    open_loop_eval_traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    open_loop_eval_steps_per_traj: int = 100
    """Number of steps to evaluate per trajectory."""

    open_loop_eval_plot_indices: Optional[list[int]] = None
    """List of action indices to plot. If None, plots all indices."""

    @property
    def accumulated_batch_size(self) -> int:
        """Total samples per optimizer step, after gradient accumulation."""
        if self.per_gpu_batch_size is not None:
            global_batch = self.per_gpu_batch_size * self.num_gpus
        else:
            global_batch = self.global_batch_size
        return global_batch * self.gradient_accumulation_steps

    @property
    def resolved_vlm_learning_rate(self) -> float:
        """Effective VLM LR after falling back to the global learning rate."""
        return self.learning_rate if self.vlm_learning_rate is None else self.vlm_learning_rate

    @property
    def resolved_action_head_learning_rate(self) -> float:
        """Effective action-head LR after falling back to the global learning rate."""
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
    def resolved_dit_learning_rate(self) -> float:
        return (
            self.resolved_action_head_learning_rate
            if self.dit_learning_rate is None
            else self.dit_learning_rate
        )

    @property
    def resolved_tactile_encoder_learning_rate(self) -> float:
        return (
            self.resolved_action_head_learning_rate
            if self.tactile_encoder_learning_rate is None
            else self.tactile_encoder_learning_rate
        )

    def __post_init__(self) -> None:
        if self.dashboard_refresh_seconds <= 0:
            raise ValueError("dashboard_refresh_seconds must be positive")
        if self.multimodal_diagnostics_interval < 1:
            raise ValueError("multimodal_diagnostics_interval must be positive")
        if self.multimodal_diagnostics_level not in {"off", "basic", "per_layer"}:
            raise ValueError("multimodal_diagnostics_level must be off, basic, or per_layer")
        if self.enable_gate_gradient_diagnostics and (
            not self.enable_multimodal_diagnostics or self.multimodal_diagnostics_level == "off"
        ):
            raise ValueError(
                "enable_gate_gradient_diagnostics requires enabled multimodal diagnostics"
            )
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
        if self.resolved_dit_learning_rate <= 0:
            raise ValueError("resolved dit_learning_rate must be positive")
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
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )
        if self.gradient_accumulation_steps > 1 and self.per_gpu_batch_size is None:
            warnings.warn(
                f"global_batch_size={self.global_batch_size} is pre-accumulation; "
                f"accumulated_batch_size={self.accumulated_batch_size} "
                f"(× gradient_accumulation_steps={self.gradient_accumulation_steps}).",
                stacklevel=2,
            )


def check_resume_compatibility(training: TrainingConfig) -> None:
    """Reject ``save_only_model=True`` with any requested resume.

    HF Trainer would otherwise restore ``global_step`` from
    ``trainer_state.json`` and silently re-init optimizer / LR schedule.
    """
    if training.save_only_model and training.resume_from_checkpoint:
        raise ValueError(
            "save_only_model=True is incompatible with resume_from_checkpoint=True: "
            "the checkpoint lacks optimizer/scheduler/RNG state, so resuming "
            "silently re-initializes the optimizer and LR schedule, degrading "
            "training quality. Either disable save_only_model, or start fresh "
            "(set resume_from_checkpoint=False or use a different output_dir)."
        )
    if training.resume_from_checkpoint:
        warnings.warn(
            "Resuming restores optimizer and scheduler state from the checkpoint. "
            "Requested learning rates (VLM="
            f"{training.resolved_vlm_learning_rate:g}, action head="
            f"{training.resolved_action_head_learning_rate:g}) are used to construct the optimizer, "
            "but checkpoint state may restore different current group LRs and scheduler progress.",
            stacklevel=2,
        )
