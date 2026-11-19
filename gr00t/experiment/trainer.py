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

"""Custom Trainer with simple profiling utilities.

This subclass of HuggingFace's ``Trainer`` measures:
1. Data loading latency (time between the end of the previous ``training_step`` and
   the start of the current ``training_step``).
2. Forward-pass latency (time spent inside the base ``training_step`` implementation,
   which essentially wraps the model's forward / loss computation).

The statistics are logged via ``self.log`` every ``profile_log_interval`` steps and
also sent to the standard ``logging`` logger.  This is *not* meant to be a fully
fledged profiler – it is a quick, lightweight way to confirm whether the training
pipeline is bottlenecked by data loading or by the model's computation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import queue
import sys
import threading
from typing import Any, Optional

import torch
from transformers.trainer import TRAINER_STATE_NAME, Trainer, TrainerState, get_last_checkpoint
from transformers.trainer_callback import TrainerCallback

from gr00t.experiment.checkpoint_memory import monitor_checkpoint_memory
from gr00t.experiment.sample_exposure import SampleExposure
from gr00t.experiment.training_memory import log_deepspeed_engine_state, log_memory_snapshot


MEMORY_LOGGER = logging.getLogger("gr00t.first_step_memory")
MEMORY_LOGGER.setLevel(logging.INFO)


class FirstStepMemoryCallback(TrainerCallback):
    """Capture rank-local memory around DeepSpeed init and the first optimizer step."""

    def __init__(
        self,
        deepspeed_enabled: bool,
        *,
        zero3_offload_optimizer: bool = False,
        zero3_offload_param: bool = False,
        zero3_sub_group_size: int | None = None,
    ):
        self.deepspeed_enabled = deepspeed_enabled
        self.zero3_offload_optimizer = zero3_offload_optimizer
        self.zero3_offload_param = zero3_offload_param
        self.zero3_sub_group_size = zero3_sub_group_size
        self._before_optimizer_step = False
        self._after_optimizer_step = False

    def on_train_begin(self, args, state, control, **kwargs):
        phase = (
            "after_deepspeed_initialization"
            if self.deepspeed_enabled
            else "after_trainer_initialization"
        )
        log_memory_snapshot(phase)
        if self.deepspeed_enabled:
            log_deepspeed_engine_state(
                kwargs.get("optimizer"),
                expected_optimizer_offload=self.zero3_offload_optimizer,
                expected_param_offload=self.zero3_offload_param,
                expected_sub_group_size=self.zero3_sub_group_size,
            )
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if not self.deepspeed_enabled and not self._before_optimizer_step:
            log_memory_snapshot("before_first_optimizer_step")
            self._before_optimizer_step = True
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        if not self.deepspeed_enabled and not self._after_optimizer_step:
            log_memory_snapshot("after_first_optimizer_step")
            self._after_optimizer_step = True
        return control


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


class _BatchIterator:
    """Lightweight iterator that yields pre-collated batches."""

    def __init__(self, buffer, bs, collator, total_steps):
        self._buffer = buffer
        self._bs = bs
        self._collate = collator
        self._total_steps = total_steps
        self._produced = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self._total_steps

    def __next__(self):
        if self._produced >= self._total_steps:
            raise StopIteration

        # Fast path – single lock acquisition inside ``sample_batch``.
        batch_samples = self._buffer.sample_batch(self._bs)  # type: ignore[attr-defined]
        self._produced += 1
        return self._collate(batch_samples)


class _PrefetchIterator:
    def __init__(self, buffer, bs, collate_fn, total_steps):
        self.buffer = buffer
        self.bs = bs
        self.collate = collate_fn
        self.total = total_steps
        self.produced = 0

        self._q = queue.Queue(maxsize=4)
        self._stop = False

        # Start background worker
        self._worker = threading.Thread(target=self._fill)
        self._worker.daemon = True
        self._worker.start()

    def _fill(self):
        while not self._stop:
            if self.produced + self._q.qsize() >= self.total:
                break
            # block if queue is full
            samples = self.buffer.sample_batch(self.bs)
            batch = self.collate(samples)
            self._q.put(batch)

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def __next__(self):
        if self.produced >= self.total:
            self._stop = True
            # in case worker is blocked on put()
            raise StopIteration
        batch = self._q.get()  # this will block until the next batch is ready
        self.produced += 1
        return batch


def _batch_accuracy(
    preds: torch.Tensor, labels: torch.Tensor, action_offset: Optional[int] = None
) -> torch.Tensor:  # noqa: D401
    """Compute token-level accuracy, ignoring ``-100`` label positions.

    Args:
        preds: Predicted token ids of shape ``(batch, seq_len)``.
        labels: Ground-truth label ids with the same shape as ``preds``.

    Returns:
        Scalar tensor with the fraction of correctly predicted labels in the
        current batch.
    """
    # casual prediction
    # Shift so that tokens < n predict n
    # https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L60
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = (preds == labels) & mask

    # Avoid division by zero for empty masks (should not happen in practice)
    denom = mask.sum().clamp(min=1)
    accuracy = correct.sum().float() / denom.float()
    return accuracy


class Gr00tTrainer(Trainer):
    """Trainer that bypasses torch dataloader and makes data collator async."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:  # noqa: D401 – simple description above
        """Initialize the trainer.

        Args:
            *args: Positional arguments forwarded to ``Trainer``.
        """
        self.action_offset = kwargs.pop("action_offset", None)
        self.multiprocessing_context = kwargs.pop("multiprocessing_context", "fork")
        self.vlm_learning_rate = kwargs.pop("vlm_learning_rate", None)
        self.action_head_learning_rate = kwargs.pop("action_head_learning_rate", None)
        self.dit_learning_rate = kwargs.pop("dit_learning_rate", None)
        self.point_encoder_learning_rate = kwargs.pop("point_encoder_learning_rate", None)
        self.tactile_encoder_learning_rate = kwargs.pop("tactile_encoder_learning_rate", None)
        dashboard_enabled = kwargs.pop("enable_training_dashboard", None)
        dashboard_refresh_seconds = kwargs.pop("dashboard_refresh_seconds", 1.0)
        self.enable_multimodal_diagnostics = kwargs.pop("enable_multimodal_diagnostics", False)
        self.enable_gate_gradient_diagnostics = kwargs.pop(
            "enable_gate_gradient_diagnostics", False
        )
        diagnostics_level = kwargs.pop("multimodal_diagnostics_level", "basic")
        self.multimodal_diagnostics_interval = kwargs.pop("multimodal_diagnostics_interval", 50)
        super().__init__(*args, **kwargs)
        self.zero3_gradient_audit = None
        if os.environ.get("GR00T_ZERO3_DEBUG", "").lower() in {"1", "true", "yes"}:
            from gr00t.experiment.zero3_debug import Zero3GradientAudit

            self.zero3_gradient_audit = Zero3GradientAudit(self.model, self.args.output_dir)
        self.sample_exposure = SampleExposure(self.train_dataset, self.args)
        if self.is_deepspeed_enabled:
            # Transformers leaves Accelerator's plugin at its default 1 while its
            # TrainingArguments and the finalized DeepSpeed config use the requested
            # accumulation count. Align the plugin before Accelerator.prepare().
            self.accelerator.gradient_accumulation_steps = self.args.gradient_accumulation_steps
        # HF's PrinterCallback prints a raw dict for every Trainer.log call. Replace
        # both default renderers so rank zero owns one coordinated display.
        from transformers.trainer_callback import PrinterCallback, ProgressCallback

        from gr00t.experiment.training_dashboard import (
            QuietProgressCallback,
            TrainingDashboardCallback,
        )

        self.callback_handler.remove_callback(PrinterCallback)
        self.callback_handler.remove_callback(ProgressCallback)
        if dashboard_enabled is False and sys.stdout.isatty():
            self.add_callback(QuietProgressCallback())
        from gr00t.experiment.dashboard_telemetry import TELEMETRY_ENV, DashboardTelemetryCallback

        if os.environ.get(TELEMETRY_ENV):
            self.add_callback(
                DashboardTelemetryCallback(os.environ[TELEMETRY_ENV], dashboard_refresh_seconds)
            )
        else:
            self.add_callback(
                TrainingDashboardCallback(
                    dashboard_enabled,
                    dashboard_refresh_seconds,
                    diagnostics_enabled=self.enable_multimodal_diagnostics
                    and diagnostics_level != "off",
                )
            )
        self._forward_diagnostics: dict[str, torch.Tensor] = {}
        self._forward_diagnostics_count = 0
        self._forward_diagnostics_step = None
        self._gradient_diagnostics: dict[str, torch.Tensor] = {}
        self._clip_observations = []
        self.multimodal_monitor = None
        if self.enable_multimodal_diagnostics and diagnostics_level != "off":
            from gr00t.experiment.multimodal_diagnostics import MultimodalDiagnostics

            self.multimodal_monitor = MultimodalDiagnostics(
                self,
                self.multimodal_diagnostics_interval,
                diagnostics_level,
                enable_gate_gradient_diagnostics=self.enable_gate_gradient_diagnostics,
            )
            self.add_callback(self.multimodal_monitor)

    def _save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Capture memory across the complete native Trainer checkpoint operation."""
        step = getattr(self.state, "global_step", "unknown")
        with monitor_checkpoint_memory(f"checkpoint-{step}.complete"):
            return super()._save_checkpoint(*args, **kwargs)

    def save_model(self, output_dir: str | None = None, *args: Any, **kwargs: Any) -> None:
        """Measure model saving separately from optimizer, scheduler, and RNG state."""
        destination = output_dir or self.args.output_dir
        internal = kwargs.get("_internal_call", False)
        if internal and self._is_zero3_enabled():
            # Transformers' ZeRO-3 save_model fallback calls engine.save_checkpoint
            # when gathering is disabled. _save_optimizer_and_scheduler below
            # calls it again, where DeepSpeed writes model + optimizer + scheduler
            # shards together. Skip this first call to avoid duplicate full saves.
            with monitor_checkpoint_memory(
                f"model-save.{Path(destination).name}.internal-zero3-skipped"
            ):
                logging.info(
                    "Skipping HF model consolidation for native ZeRO-3 checkpoint %s",
                    destination,
                )
            return
        with monitor_checkpoint_memory(
            f"model-save.{Path(destination).name}.internal-{int(internal)}"
        ):
            return super().save_model(output_dir, *args, **kwargs)

    def _is_zero3_enabled(self) -> bool:
        """Read the active DeepSpeed stage from the plugin or Trainer config."""
        if not self.is_deepspeed_enabled:
            return False
        plugin = getattr(getattr(self.accelerator, "state", None), "deepspeed_plugin", None)
        if getattr(plugin, "zero_stage", None) == 3:
            return True
        ds_config = getattr(self.args, "deepspeed", None)
        if isinstance(ds_config, str):
            try:
                with open(ds_config) as config_file:
                    ds_config = json.load(config_file)
            except (OSError, json.JSONDecodeError):
                ds_config = None
        if isinstance(ds_config, dict):
            return ds_config.get("zero_optimization", {}).get("stage") == 3
        hf_config = getattr(self.args, "hf_deepspeed_config", None)
        config = getattr(hf_config, "config", {})
        return config.get("zero_optimization", {}).get("stage") == 3

    def _save_optimizer_and_scheduler(self, *args: Any, **kwargs: Any) -> None:
        with monitor_checkpoint_memory(f"optimizer-scheduler-save.step-{self.state.global_step}"):
            return super()._save_optimizer_and_scheduler(*args, **kwargs)

    def _save_rng_state(self, *args: Any, **kwargs: Any) -> None:
        with monitor_checkpoint_memory(f"rng-save.step-{self.state.global_step}"):
            return super()._save_rng_state(*args, **kwargs)

    def create_optimizer(self):
        """Create identity-safe VLM/action-head optimizer parameter groups.

        Direct ``Gr00tTrainer`` users that do not provide the split learning rates retain
        HuggingFace Trainer's standard optimizer behavior. The GR00T experiment path always
        provides both rates.
        """
        if self.vlm_learning_rate is None and self.action_head_learning_rate is None:
            return super().create_optimizer()
        if self.vlm_learning_rate is None or self.action_head_learning_rate is None:
            raise ValueError(
                "vlm_learning_rate and action_head_learning_rate must be provided together"
            )
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        optimizer_grouped_parameters = self._create_split_optimizer_groups(opt_model)

        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
        else:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )
        optimizer_kwargs = dict(optimizer_kwargs)
        unsupported_overrides = {
            key for key in ("params", "model", "optimizer_dict") if key in optimizer_kwargs
        }
        if unsupported_overrides:
            raise ValueError(
                "The selected optimizer supplies its own parameter grouping and cannot preserve "
                "the VLM/action-head learning-rate split: "
                f"{sorted(unsupported_overrides)}"
            )

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        self._configure_bitsandbytes_embedding_overrides(opt_model, optimizer_cls, optimizer_kwargs)
        return self.optimizer

    def _create_split_optimizer_groups(self, model: torch.nn.Module) -> list[dict[str, Any]]:
        """Classify every trainable tensor by module identity, LR, and HF decay semantics."""
        backbone = getattr(model, "backbone", None)
        action_head = getattr(model, "action_head", None)
        if not isinstance(backbone, torch.nn.Module) or not isinstance(
            action_head, torch.nn.Module
        ):
            raise RuntimeError(
                "Split learning rates require model.backbone and model.action_head modules"
            )

        named_parameters = list(model.named_parameters())
        trainable_parameter_ids = {
            id(parameter) for _name, parameter in named_parameters if parameter.requires_grad
        }
        vlm_parameter_ids = {
            id(parameter) for parameter in backbone.parameters() if parameter.requires_grad
        }
        action_head_parameter_ids = {
            id(parameter) for parameter in action_head.parameters() if parameter.requires_grad
        }
        point_encoder = getattr(action_head, "point_encoder", None)
        point_backbone_parameter_ids: set[int] = set()
        if self.point_encoder_learning_rate is not None and callable(
            getattr(point_encoder, "backbone_parameters", None)
        ):
            point_backbone_parameter_ids = {
                id(parameter)
                for parameter in point_encoder.backbone_parameters()
                if parameter.requires_grad
            }
            if not point_backbone_parameter_ids:
                logging.info(
                    "Point encoder backbone is frozen; no point-encoder LR optimizer group created"
                )
        action_head_standard_parameter_ids = (
            action_head_parameter_ids - point_backbone_parameter_ids
        )
        tactile_encoder = getattr(action_head, "tactile_encoder", None)
        tactile_backbone_parameter_ids: set[int] = set()
        if self.tactile_encoder_learning_rate is not None and callable(
            getattr(tactile_encoder, "backbone_parameters", None)
        ):
            tactile_backbone_parameter_ids = {
                id(parameter)
                for parameter in tactile_encoder.backbone_parameters()
                if parameter.requires_grad
            }
            if not tactile_backbone_parameter_ids:
                logging.info(
                    "Tactile encoder backbone is frozen; no tactile-backbone LR group created"
                )
        action_head_standard_parameter_ids -= tactile_backbone_parameter_ids
        dit_parameter_ids: set[int] = set()
        if self.dit_learning_rate is not None and hasattr(action_head, "model"):
            dit = action_head.model
            if hasattr(dit, "dit_training_mode"):
                from gr00t.model.extension.dit_lora import original_dit_parameters

                dit_parameter_ids = {
                    id(parameter)
                    for _, parameter in original_dit_parameters(dit)
                    if parameter.requires_grad
                }
                action_head_standard_parameter_ids -= dit_parameter_ids

        duplicate_parameter_ids = vlm_parameter_ids & action_head_parameter_ids
        classified_parameter_ids = vlm_parameter_ids | action_head_parameter_ids
        unclassified_parameter_ids = trainable_parameter_ids - classified_parameter_ids
        missing_trainable_parameter_ids = classified_parameter_ids - trainable_parameter_ids

        names_by_id = {id(parameter): name for name, parameter in named_parameters}

        def parameter_names(parameter_ids: set[int]) -> list[str]:
            return [
                names_by_id.get(parameter_id, f"<unknown:{parameter_id}>")
                for parameter_id in parameter_ids
            ]

        if duplicate_parameter_ids or unclassified_parameter_ids or missing_trainable_parameter_ids:
            raise RuntimeError(
                "Invalid optimizer parameter ownership: "
                f"total_trainable_tensors={len(trainable_parameter_ids)}, "
                f"vlm_tensors={len(vlm_parameter_ids)}, "
                f"action_head_tensors={len(action_head_parameter_ids)}, "
                f"duplicate={parameter_names(duplicate_parameter_ids)[:10]}, "
                f"unclassified={parameter_names(unclassified_parameter_ids)[:10]}, "
                f"non_trainable_classified={parameter_names(missing_trainable_parameter_ids)[:10]}"
            )

        if bool(getattr(getattr(model, "config", None), "use_lora", False)):
            from gr00t.model.modules.qwen3_backbone import _is_lora_parameter_name

            trainable_backbone_names = [
                name for name, parameter in backbone.named_parameters() if parameter.requires_grad
            ]
            if not trainable_backbone_names:
                raise RuntimeError("LoRA mode has no trainable VLM adapter parameters")
            non_lora_names = [
                name for name in trainable_backbone_names if not _is_lora_parameter_name(name)
            ]
            if non_lora_names:
                raise RuntimeError(
                    "LoRA mode found trainable non-LoRA backbone parameters; frozen Qwen3-VL "
                    f"base weights must not enter the optimizer: {non_lora_names[:10]}"
                )

        decay_parameter_names = set(self.get_decay_parameter_names(model))
        decay_parameter_ids = {
            id(parameter) for name, parameter in named_parameters if name in decay_parameter_names
        }

        group_specs = (
            ("VLM decay", vlm_parameter_ids, True, self.vlm_learning_rate),
            ("VLM no_decay", vlm_parameter_ids, False, self.vlm_learning_rate),
            (
                "Action Head decay",
                action_head_standard_parameter_ids,
                True,
                self.action_head_learning_rate,
            ),
            (
                "Action Head no_decay",
                action_head_standard_parameter_ids,
                False,
                self.action_head_learning_rate,
            ),
            ("Pretrained DiT decay", dit_parameter_ids, True, self.dit_learning_rate),
            ("Pretrained DiT no_decay", dit_parameter_ids, False, self.dit_learning_rate),
            (
                "Point Encoder Backbone decay",
                point_backbone_parameter_ids,
                True,
                self.point_encoder_learning_rate,
            ),
            (
                "Point Encoder Backbone no_decay",
                point_backbone_parameter_ids,
                False,
                self.point_encoder_learning_rate,
            ),
            (
                "Tactile Encoder Backbone decay",
                tactile_backbone_parameter_ids,
                True,
                self.tactile_encoder_learning_rate,
            ),
            (
                "Tactile Encoder Backbone no_decay",
                tactile_backbone_parameter_ids,
                False,
                self.tactile_encoder_learning_rate,
            ),
        )
        optimizer_groups: list[dict[str, Any]] = []
        for group_name, owner_ids, use_decay, learning_rate in group_specs:
            parameters = [
                parameter
                for _name, parameter in named_parameters
                if id(parameter) in owner_ids
                and (id(parameter) in decay_parameter_ids) == use_decay
            ]
            if parameters:
                optimizer_groups.append(
                    {
                        "name": group_name,
                        "params": parameters,
                        "lr": learning_rate,
                        "weight_decay": self.args.weight_decay if use_decay else 0.0,
                    }
                )

        optimizer_parameter_ids = [
            id(parameter) for group in optimizer_groups for parameter in group["params"]
        ]
        unique_optimizer_parameter_ids = set(optimizer_parameter_ids)
        if len(optimizer_parameter_ids) != len(unique_optimizer_parameter_ids):
            raise RuntimeError("A trainable parameter was assigned to multiple optimizer groups")
        if unique_optimizer_parameter_ids != trainable_parameter_ids:
            missing = trainable_parameter_ids - unique_optimizer_parameter_ids
            unexpected = unique_optimizer_parameter_ids - trainable_parameter_ids
            raise RuntimeError(
                "Optimizer parameter coverage mismatch: "
                f"missing={parameter_names(missing)[:10]}, "
                f"unexpected={parameter_names(unexpected)[:10]}"
            )
        if not optimizer_groups:
            raise RuntimeError(
                "Cannot create an optimizer because the model has no trainable parameters"
            )

        vlm_parameter_count = sum(
            parameter.numel()
            for _name, parameter in named_parameters
            if id(parameter) in vlm_parameter_ids
        )
        action_head_parameter_count = sum(
            parameter.numel()
            for _name, parameter in named_parameters
            if id(parameter) in action_head_parameter_ids
        )
        MEMORY_LOGGER.info(
            "Optimizer ownership: VLM=%s parameters/%d tensors, Action Head=%s parameters/%d tensors",
            f"{vlm_parameter_count:,}",
            len(vlm_parameter_ids),
            f"{action_head_parameter_count:,}",
            len(action_head_parameter_ids),
        )
        for group in optimizer_groups:
            MEMORY_LOGGER.info(
                "Optimizer group rank=%s name=%r: lr=%g, weight_decay=%g, tensors=%d, "
                "trainable_parameters=%s",
                os.environ.get("RANK", "0"),
                group["name"],
                group["lr"],
                group["weight_decay"],
                len(group["params"]),
                f"{sum(parameter.numel() for parameter in group['params']):,}",
            )
        from gr00t.experiment.optimizer_artifacts import describe_groups

        self.optimizer_group_manifest = describe_groups(optimizer_groups)
        return optimizer_groups

    @staticmethod
    def _configure_bitsandbytes_embedding_overrides(
        model: torch.nn.Module, optimizer_cls: type, optimizer_kwargs: dict[str, Any]
    ) -> None:
        """Preserve HuggingFace Trainer's 8-bit embedding override behavior."""
        if "bitsandbytes" not in str(optimizer_cls) or optimizer_kwargs.get("optim_bits") != 8:
            return

        import bitsandbytes

        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
        skipped = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                parameter_sizes = {
                    parameter.data_ptr(): parameter.numel() for parameter in module.parameters()
                }
                skipped += sum(parameter_sizes.values())
                manager.register_module_override(module, "weight", {"optim_bits": 32})
        logging.info("bitsandbytes optimizer keeps %s embedding parameters in fp32", f"{skipped:,}")

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, *args, **kwargs):
        # Trainer calls this at optimizer boundaries and epoch end; count each step once.
        if getattr(self, "_clip_step", None) != self.state.global_step and grad_norm is not None:
            import math

            norm = float(grad_norm)
            if math.isfinite(norm) and self.args.max_grad_norm > 0:
                self._clip_observations.append(float(norm > self.args.max_grad_norm))
            self._clip_step = self.state.global_step
        return super()._maybe_log_save_evaluate(tr_loss, grad_norm, *args, **kwargs)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        logs = dict(logs)
        if self.is_in_train:
            # Plain keys are namespaced once by HF's existing W&B callback.
            # TrainerState.global_step is restored before resumed logging.
            logs.update(self.sample_exposure.metrics(self.state.global_step))
        if "loss" in logs and self._forward_diagnostics_count:
            from gr00t.experiment.action_diagnostics import loss_ratios

            names = sorted(self._forward_diagnostics)
            # Sum microbatch observations and counts together; all ranks use the same schema.
            values = torch.stack([self._forward_diagnostics[n].detach().float() for n in names])
            packed = torch.cat((values, values.new_tensor([self._forward_diagnostics_count])))
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(packed)
            logs.update(zip(names, (packed[:-1] / packed[-1]).cpu().tolist()))
            loss_ratios(logs)
            self._forward_diagnostics = {}
            self._forward_diagnostics_count = 0
        for source, target in (
            ("loss/action", "train/action_loss"),
            ("loss/total", "train/total_loss"),
            ("loss/zero_prediction_mse", "train/zero_prediction_mse"),
            ("loss/action_to_zero_ratio", "train/action_to_zero_ratio"),
            ("grad_norm/dit", "train/grad_norm_dit"),
            ("grad_norm/lora", "train/grad_norm_lora"),
            ("grad_norm/sidecar", "train/grad_norm_sidecar"),
            ("grad_norm/temporal", "train/grad_norm_temporal"),
            ("grad_norm/moe", "train/grad_norm_moe"),
        ):
            if source in logs and logs[source] is not None:
                logs[target] = logs[source]
        if logs.get("grad_norm") is not None:
            logs["train/grad_norm_global"] = logs["grad_norm"]
        import time

        now = time.monotonic()
        last_time = getattr(self, "_last_system_log_time", None)
        last_step = getattr(self, "_last_system_log_step", None)
        if last_time is not None and last_step is not None and self.state.global_step > last_step:
            seconds = now - last_time
            if seconds > 0:
                logs["system/step_time"] = seconds / (self.state.global_step - last_step)
                logs["system/steps_per_second"] = (self.state.global_step - last_step) / seconds
        self._last_system_log_time = now
        self._last_system_log_step = self.state.global_step
        if torch.cuda.is_available():
            logs["system/gpu_memory_allocated"] = torch.cuda.memory_allocated() / (1024**3)
        if logs.get("grad_norm") is not None:
            import math

            norm = float(logs["grad_norm"])
            logs["grad_norm/pre_clip"] = norm
            logs["grad_norm/is_finite"] = float(math.isfinite(norm))
            # This is the configured coefficient, not an observed post-clip norm.
            if math.isfinite(norm) and self.args.max_grad_norm > 0:
                coefficient = min(1.0, self.args.max_grad_norm / (norm + 1e-6))
                logs["grad_norm/clip_coefficient"] = coefficient
                observations = self._clip_observations
                logs["grad_norm/clip_fraction"] = (
                    sum(observations) / len(observations)
                    if observations
                    else float(coefficient < 1)
                )
                self._clip_observations = []
        # Hide epoch from logged metrics as it's misleading for Iterable datasets.
        epoch = self.state.epoch
        self.state.epoch = None
        try:
            super().log(logs, start_time=start_time)
            # HF rewrites every custom name as train/<name>. Retain those and expose
            # the canonical diagnostic namespaces in the same W&B optimizer-step axis.
            if self.is_world_process_zero() and "wandb" in self.args.report_to:
                import wandb

                scalars = {
                    k: v for k, v in logs.items() if "/" in k and isinstance(v, (int, float))
                }
                if wandb.run is not None and scalars:
                    wandb.log(
                        {**scalars, "train/global_step": self.state.global_step}, commit=False
                    )
        finally:
            self.state.epoch = epoch

    def get_train_dataloader(self):  # noqa: D401
        """Return a iterable dataloader without skipping the data during resume, but reseed the dataset instead."""

        # Fall back to default behaviour if not using the custom buffer.
        # During resume, don't skip the data
        self.args.ignore_data_skip = True
        curr_global_step = self.state.global_step
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0:
            # ``new_seed`` MUST be the same on every rank: ``ShardedMixtureDataset``
            # builds its shard schedule from this seed and partitions disjointly
            # by index, so a per-rank delta here would cause sample duplication
            # / loss across ranks. Both inputs are rank-symmetric (the dataset's
            # own seed was set rank-symmetrically at __init__, and global_step
            # is read from TrainerState which is broadcast via rendezvous).
            new_seed = self.train_dataset.seed + curr_global_step
            self.train_dataset.reset_seed(new_seed)
            print(
                f"Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible."
            )

        print("Creating custom train dataloader")
        # Handle the case where the dataset is an IterableDataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )
        # Use persistent workers for sharded dataset if num_workers is greater than 0
        persistent_workers = self.args.dataloader_num_workers > 0

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
        }

        # multiprocessing_context can only be used with num_workers > 0
        if self.args.dataloader_num_workers > 0:
            dataloader_params["multiprocessing_context"] = self.multiprocessing_context

        return torch.utils.data.DataLoader(self.train_dataset, **dataloader_params)

    def train(
        self,
        resume_from_checkpoint=None,
        **kwargs,
    ):
        """Pre-load TrainerState before super().train() so get_train_dataloader
        can read self.state.global_step (stateful samplers rely on this).
        ``resume_from_checkpoint=True`` with no checkpoint raises rather than
        silently starting fresh.
        """
        if resume_from_checkpoint is True:
            latest_checkpoint = get_last_checkpoint(self.args.output_dir)
            if latest_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )
        elif resume_from_checkpoint in (False, None):
            latest_checkpoint = None
        else:
            latest_checkpoint = resume_from_checkpoint  # caller passed an explicit path

        if latest_checkpoint is not None:
            self._validate_resumable_checkpoint(latest_checkpoint)
            logging.info(f"Resuming from checkpoint {latest_checkpoint}")
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(latest_checkpoint, TRAINER_STATE_NAME)
            )

        self.sample_exposure.prepare(
            self.args.output_dir,
            latest_checkpoint,
            rank_zero=self.is_world_process_zero(),
        )

        try:
            return super().train(resume_from_checkpoint=latest_checkpoint, **kwargs)
        finally:
            if self.multimodal_monitor is not None:
                self.multimodal_monitor.close()
            self._close_training_display()

    def _close_training_display(self):
        from gr00t.experiment.training_dashboard import (
            QuietProgressCallback,
            TrainingDashboardCallback,
        )

        for callback in self.callback_handler.callbacks:
            if isinstance(callback, TrainingDashboardCallback):
                callback._stop_live()
            elif isinstance(callback, QuietProgressCallback):
                callback.close()

    def evaluate(self, *args, **kwargs):
        try:
            return super().evaluate(*args, **kwargs)
        except BaseException:
            self._close_training_display()
            raise

    def predict(self, *args, **kwargs):
        try:
            return super().predict(*args, **kwargs)
        except BaseException:
            self._close_training_display()
            raise

    @staticmethod
    def _validate_resumable_checkpoint(checkpoint: str | os.PathLike[str]) -> None:
        """Reject partial/save-only checkpoints before HF silently reinitializes state."""
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_dir():
            raise ValueError(f"Checkpoint directory does not exist: {checkpoint_path}")

        required_root_files = {
            "trainer state": checkpoint_path / TRAINER_STATE_NAME,
        }
        missing = [name for name, path in required_root_files.items() if not path.is_file()]

        has_optimizer = (checkpoint_path / "optimizer.pt").is_file() or any(
            checkpoint_path.rglob("*optim_states.pt")
        )
        has_scheduler = (checkpoint_path / "scheduler.pt").is_file() or any(
            checkpoint_path.rglob("*model_states.pt")
        )
        has_rng = any(checkpoint_path.glob("rng_state*.pth"))

        if not has_optimizer:
            missing.append("optimizer state")
        if not has_scheduler:
            missing.append("scheduler state")
        if not has_rng:
            missing.append("RNG state")

        if missing:
            raise ValueError(
                f"Checkpoint {checkpoint_path} is not resumable; missing "
                f"{', '.join(missing)}. Save checkpoints with save_only_model=False."
            )

    # ------------------------------------------------------------------
    # Loss / accuracy computation override
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):  # type: ignore[override]
        """Compute loss *and* log token-level accuracy every training step.

        We delegate the heavy-lifting (including label smoothing, custom loss
        functions, etc.) to the parent ``Trainer.compute_loss`` implementation
        by calling it with ``return_outputs=True``.  After obtaining the loss
        *and* model outputs, we calculate accuracy and push it to the logger.
        """

        # Use parent implementation to preserve built-in functionality.
        monitor = getattr(self, "multimodal_monitor", None)
        if monitor is not None and monitor.dit is not None:
            monitor.dit._diagnostic_observer = (
                monitor.observe if model.training and monitor.sampled() else None
            )
            monitor.dit._diagnostic_gradient_observer = (
                monitor.observe_gate_gradients
                if model.training and monitor.sampled() and monitor.enable_gate_gradient_diagnostics
                else None
            )
        try:
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        finally:
            if monitor is not None and monitor.dit is not None:
                monitor.dit._diagnostic_observer = None
                monitor.dit._diagnostic_gradient_observer = None
        output_dict = outputs if isinstance(outputs, dict) else getattr(outputs, "__dict__", {})
        diagnostics = output_dict.get("diagnostics", {}) if model.training else {}
        for name, value in diagnostics.items():
            if torch.is_tensor(value):
                detached = value.detach()
                self._forward_diagnostics[name] = (
                    detached
                    if name not in self._forward_diagnostics
                    else self._forward_diagnostics[name] + detached
                )
        if diagnostics:
            self._forward_diagnostics_count += 1
        # import ipdb; ipdb.set_trace()
        # # save the model's embedding for the first step
        # input_embeddings = model.get_input_embeddings().weight.data.cpu()
        # output_embeddings = model.get_output_embeddings().weight.data.cpu()
        # torch.save(input_embeddings, f"input_embeddings_{self.state.global_step}.pt")
        # torch.save(output_embeddings, f"output_embeddings_{self.state.global_step}.pt")

        # Record last loss for testing purposes.
        self.loss = loss

        # --------------------------------------------------------------
        # Accuracy calculation
        # --------------------------------------------------------------
        if (
            self.state.global_step % self.args.logging_steps == 0
            and model.training
            and "labels" in inputs
        ):
            if self.action_offset is not None:
                preds = outputs.logits.detach()[:, :, self.action_offset :].argmax(dim=-1).cpu()
            else:
                preds = outputs.logits.detach().argmax(dim=-1).cpu()
            with torch.no_grad():
                acc_local = _batch_accuracy(
                    preds, inputs["labels"].to(device=preds.device), self.action_offset
                )
            acc_tensor = torch.tensor(acc_local.item(), device=loss.device)
            acc_mean = self._nested_gather(acc_tensor).mean().item()

            self.log({"train_accuracy": acc_mean})
            if self.args.local_rank in (-1, 0):
                # Log a sample of ground-truth vs predicted action tokens from
                # the first batch element so users can verify the model is
                # learning the right behaviors.
                shifted_labels = inputs["labels"][:1, 1:].cpu()
                shifted_preds = preds[:1, :-1]
                mask_0 = shifted_labels[0] != -100
                gt_tokens = shifted_labels[0][mask_0][:20]
                if self.action_offset is not None:
                    gt_tokens = gt_tokens - self.action_offset
                gt_sample = gt_tokens.tolist()
                pred_sample = shifted_preds[0][mask_0[: shifted_preds.shape[1]]][:20].tolist()
                logging.info(
                    "Step %d — GT vs Pred (first 20 action tokens, batch[0]):\n"
                    "  GT:   %s\n  Pred: %s",
                    self.state.global_step,
                    gt_sample,
                    pred_sample,
                )

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not getattr(self, "_first_forward_memory_logged", False):
            log_memory_snapshot("before_first_forward")
            self._first_forward_memory_logged = True
        if self.multimodal_monitor is not None and self.is_deepspeed_enabled:
            self.multimodal_monitor.install_deepspeed(model)
        if self.is_deepspeed_enabled:
            if self.zero3_gradient_audit is not None:
                self.zero3_gradient_audit.install(model)
            # The one-shot memory wrapper restores its captured step method.
            # Install it last so it restores the persistent gradient audit.
            self._install_first_deepspeed_optimizer_step_diagnostics(model)
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.is_deepspeed_enabled and self.zero3_gradient_audit is not None:
            self.zero3_gradient_audit.record_loss(loss)
        if not self.is_deepspeed_enabled and not getattr(
            self, "_first_backward_memory_logged", False
        ):
            log_memory_snapshot("after_first_backward")
            self._first_backward_memory_logged = True
        return loss

    def _install_first_deepspeed_optimizer_step_diagnostics(self, model) -> None:
        """Wrap exactly one real ZeRO optimizer step with rank-local snapshots.

        Accelerate invokes ``DeepSpeedEngine.step()`` from inside
        ``accelerator.backward``. Consequently, Trainer's nominal
        ``on_pre_optimizer_step`` event occurs *after* the real ZeRO step. A
        one-shot wrapper on the initialized ZeRO optimizer is the only point
        that accurately brackets the first optimizer step and still disappears
        before subsequent steps.
        """
        if getattr(self, "_first_deepspeed_step_diagnostics_installed", False):
            return
        zero_optimizer = getattr(model, "optimizer", None)
        original_step = getattr(zero_optimizer, "step", None)
        if not callable(original_step):
            raise RuntimeError(
                "DeepSpeed is enabled but the initialized engine has no callable optimizer.step"
            )

        self._first_deepspeed_step_diagnostics_installed = True

        def first_step(*args, **kwargs):
            # Restore the original bound method before entering it so an
            # exception cannot leave diagnostics on later retries.
            zero_optimizer.step = original_step
            log_memory_snapshot("after_first_backward")
            self._first_backward_memory_logged = True
            log_memory_snapshot("before_first_optimizer_step")
            result = original_step(*args, **kwargs)
            log_memory_snapshot("after_first_optimizer_step")
            return result

        zero_optimizer.step = first_step
