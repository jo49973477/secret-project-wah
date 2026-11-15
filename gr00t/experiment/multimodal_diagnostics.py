"""Opt-in optimizer-boundary diagnostics. All distributed ranks participate."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
from transformers.trainer_callback import TrainerCallback

from gr00t.model.extension.dit_lora import original_dit_parameters


def residual_ratio(hidden, update):
    """Detached FP32 norm of the actual injected residual, with no graph retention."""
    with torch.no_grad():
        return update.detach().float().norm() / hidden.detach().float().norm().clamp_min(1e-12)


def full_value(parameter, gradient=False):
    """Safe full access, used only for the tiny gate vectors (on every rank)."""
    if gradient and (
        not parameter.requires_grad
        or (hasattr(parameter, "ds_id") and not hasattr(parameter, "_z3_optimizer"))
    ):
        return None
    if not gradient and hasattr(parameter, "ds_id"):
        from deepspeed import zero

        # Read the actual forward parameter, including BF16 rounding, rather
        # than an FP32 optimizer master that may change below one BF16 ULP.
        with zero.GatheredParameters([parameter], modifier_rank=None):
            return parameter.detach().float().cpu().clone()
    if gradient and (hasattr(parameter, "ds_id") or hasattr(parameter, "_hp_mapping")):
        from deepspeed.utils import safe_get_full_grad

        value = safe_get_full_grad(parameter)
    else:
        value = parameter.grad if gradient else parameter
    return None if value is None else value.detach().float().cpu().clone()


def collective_device(parameter, group=None):
    if torch.distributed.is_initialized() and torch.distributed.get_backend(group) == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return parameter.device


def gradient_stats(parameters, scale=1.0):
    """Stream local ZeRO-3 gradient shards and reduce only scalar statistics.

    DeepSpeed allocates zero-filled gradient buffers even for unused parameters.
    Buffer availability therefore does not prove an autograd gradient exists.
    Expose that distinction instead of labelling an unused branch as present.
    """
    parameters = [p for p in parameters if p.requires_grad]
    sharded = [p for p in parameters if hasattr(p, "ds_id")]
    ordinary = [p for p in parameters if not hasattr(p, "ds_id")]
    squares = 0.0
    present = 0
    finite = True
    for p in ordinary:
        if hasattr(p, "_hp_mapping"):
            from deepspeed.utils import safe_get_full_grad

            grad = safe_get_full_grad(p)
        else:
            grad = p.grad
        if grad is not None:
            present += 1
            value = grad.detach().double() / scale
            finite = finite and bool(torch.isfinite(value).all())
            squares += float(value.square().sum().cpu())
    if sharded:
        from deepspeed.utils import safe_get_local_grad

        # A scalar vector per module replaces full-gradient allgathers.
        group = getattr(sharded[0], "ds_process_group", None)
        device = collective_device(sharded[0], group)
        totals = torch.zeros(3, dtype=torch.float64, device=device)
        rank = torch.distributed.get_rank(group) if torch.distributed.is_initialized() else 0
        for p in sharded:
            if not hasattr(p, "_z3_optimizer"):
                continue
            grad = safe_get_local_grad(p)
            if grad is None:
                continue
            # Drop partition padding; it is not part of the parameter.
            valid = min(grad.numel(), max(0, p.ds_numel - rank * grad.numel()))
            value = grad.detach().flatten()[:valid].double() / scale
            totals[0] += value.square().sum()
            totals[1] += int(valid > 0)
            totals[2] += (~torch.isfinite(value)).sum()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals, group=group)
        total_square, available, invalid = totals.cpu().tolist()
        squares += total_square
        present += int(available > 0)
        finite = finite and invalid == 0
    available = bool(present)
    status = (
        "missing"
        if not available
        else "invalid"
        if not finite
        else "zero"
        if squares == 0
        else "nonzero"
    )
    return {
        "grad_present": None if sharded else available,
        "grad_buffer_present": available,
        "grad_finite": finite if available else None,
        "grad_norm": math.sqrt(squares) if available and finite else None,
        "grad_status": (status + "_buffer")
        if sharded and status in {"zero", "nonzero"}
        else status,
        "grad_presence_note": "ZeRO-3 buffers do not establish autograd presence"
        if sharded
        else "autograd",
    }


class MultimodalDiagnostics(TrainerCallback):
    """Observe real optimizer steps, including steps executed inside DS backward.

    Full gradients from DeepSpeed safe accessors are already global. Do not sum
    them again across ranks. Parameter snapshots are tiny gate vectors only.
    """

    def __init__(self, trainer, interval=50, level="basic"):
        if interval < 1 or level not in {"basic", "per_layer"}:
            raise ValueError("Invalid multimodal diagnostic interval or level")
        self.trainer, self.interval, self.level = trainer, interval, level
        self.before = {}
        self.record = None
        self.ratios = {}
        self.last_step = -1
        self.dit = getattr(getattr(trainer.model, "action_head", None), "model", None)
        self.previous = {}
        self._wrapped = None
        self._zero_optimizer = None

    def sampled(self):
        return (self.trainer.state.global_step + 1) % self.interval == 0

    def observe(self, modality, layer, hidden, update):
        value = residual_ratio(hidden, update)
        key = (modality, layer)
        total, count = self.ratios.get(key, (0, 0))
        self.ratios[key] = (total + value, count + 1)

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_step = -1
        self.record = None
        self.previous = {}
        if not hasattr(self.dit, "sidecar_stages") and not any(
            hasattr(self.dit, f"{modality}_gates") for modality in ("point", "tactile")
        ):
            logging.warning(
                "Multimodal diagnostics enabled, but this model has no multimodal gates"
            )

    def on_step_begin(self, args, state, control, **kwargs):
        self.ratios = {}

    def _sidecar_pre_step(self):
        if not self.sampled() or self.last_step == self.trainer.state.global_step + 1:
            return
        self.last_step = self.trainer.state.global_step + 1
        self.record = {"global_step": self.last_step, "grad_status/sample_step": self.last_step}
        optimizer = self._zero_optimizer or self.trainer.optimizer
        scale = float(getattr(optimizer, "loss_scale", 1.0))
        for label, attribute in (("point", "point_gate"), ("action", "action_gate")):
            gates = [
                getattr(s, attribute) for s in self.dit.sidecar_stages if hasattr(s, attribute)
            ]
            if not gates:
                self.record[f"sidecar_{label}_gate_mean_abs"] = None
                continue
            values = [full_value(g) for g in gates]
            self.record[f"sidecar_{label}_gate_mean_abs"] = sum(
                float(v.abs().mean()) for v in values if v is not None
            ) / len(values)
            if self.level == "per_layer":
                for index, (gate, value) in enumerate(zip(gates, values)):
                    gradient = full_value(gate, gradient=True)
                    self.record[f"sidecar_{label}_gate_stage_{index}"] = (
                        None if value is None else float(value)
                    )
                    self.record[f"sidecar_{label}_gate_grad_stage_{index}"] = (
                        None if gradient is None else float(gradient) / scale
                    )
            self.record[f"sidecar_{label}_gate_grad_norm"] = gradient_stats(gates, scale)[
                "grad_norm"
            ]
        if getattr(self.dit, "sidecar_variant", "legacy") == "multilevel_local":
            self.record["sidecar_stage_mapping"] = [
                dict(stage=i, point_stage=name, dit_block=block, **module.last_diagnostics)
                for i, (name, block, module) in enumerate(
                    zip(
                        self.dit.sidecar_point_stage_names,
                        self.dit.stage_indices,
                        self.dit.sidecar_stages,
                    )
                )
            ]
            for i, module in enumerate(self.dit.sidecar_stages):
                for key, value in module.last_diagnostics.items():
                    if isinstance(value, (float, int)):
                        self.record[f"sidecar_local/{key}_stage_{i}"] = value
                    elif key in {"point_counts", "window_counts"}:
                        self.record[f"sidecar_local/{key}_stage_{i}"] = sum(value)

        head = self.trainer.model.action_head
        groups = {
            "sidecar": list(self.dit.sidecar_stages.parameters())
            + list(self.dit.sidecar_pool.parameters()),
            "point_encoder": list(head.point_encoder.parameters())
            if head.point_encoder is not None
            else [],
            "tactile_encoder": list(head.tactile_encoder.parameters())
            if getattr(head, "tactile_encoder", None) is not None
            else [],
            "contact_teacher": list(self.dit.contact_adapter.projector.parameters()),
            "contact_predictor": list(self.dit.contact_adapter.predictor.parameters())
            + list(self.dit.contact_adapter.scene_pool.parameters())
            + list(self.dit.contact_adapter.state_projection.parameters()),
            "contact_injection": list(self.dit.contact_injections.parameters())
            + list(self.dit.contact_adapter.gate.parameters()),
            "gates": [
                g
                for stage in self.dit.sidecar_stages
                for g in (getattr(stage, "point_gate", None), stage.action_gate)
                if g is not None
            ]
            + [module.scale for module in self.dit.contact_injections],
            "dit": [p for _, p in original_dit_parameters(self.dit)],
            "lora": [p for n, p in self.dit.named_parameters() if ".lora_" in n],
            "temporal": list(head.temporal_fusion.parameters())
            if getattr(head, "temporal_fusion", None) is not None
            else [],
            "moe": list(self.dit.input_contact_adapter.parameters())
            if getattr(self.dit, "input_contact_adapter", None) is not None
            else [],
        }
        groups.update(
            {
                f"sidecar_stage_{i}": list(stage.parameters())
                for i, stage in enumerate(self.dit.sidecar_stages)
            }
        )
        self.record["gradient_timing"] = (
            "before DeepSpeed optimizer step/unscale/clip"
            if self._zero_optimizer
            else "after Trainer clipping, before optimizer step"
        )
        for name, parameters in groups.items():
            stats = gradient_stats(parameters, scale)
            self.record[f"grad_norm/{name}"] = stats["grad_norm"]
            self.record[f"{name}_grad_norm"] = stats["grad_norm"]
            if stats["grad_finite"] is False:
                logging.warning(
                    "Non-finite %s gradients at optimizer step %s", name, self.last_step
                )
            # Non-DS callback runs after clipping: this is a real measured global post norm.
            for field in ("grad_present", "grad_buffer_present", "grad_finite"):
                self.record[f"grad_status/{name}/{field}"] = stats[field]
            self.record[f"grad_status/{name}/trainable_numel"] = sum(
                getattr(p, "ds_numel", p.numel()) for p in parameters if p.requires_grad
            )
            self.record[f"grad_status/{name}/frozen"] = not any(p.requires_grad for p in parameters)
        if not self._zero_optimizer and hasattr(self.trainer.model, "parameters"):
            stats = gradient_stats(list(self.trainer.model.parameters()), scale)
            self.record["grad_norm/post_clip"] = stats["grad_norm"]
        aliases = {
            "observed_contact_variance": "contact/observed_variance",
            "observed_contact_norm": "contact/observed_norm",
            "predicted_contact_variance": "contact/predicted_variance",
            "contact_surprise_norm": "contact/surprise_norm",
            "contact_gate_mean": "contact/gate_mean",
            "contact_prediction_loss": "contact/prediction_loss",
        }
        for key, alias in aliases.items():
            value = getattr(self.dit, "contact_diagnostics", {}).get(key)
            pair = torch.zeros(2, device=next(self.dit.parameters()).device)
            if value is not None:
                pair[0], pair[1] = value.detach(), 1
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(pair)
            self.record[alias] = float(pair[0] / pair[1]) if pair[1] else None
            self.record[key] = self.record[alias]
        for label in ("action", "point"):
            self.record[f"sidecar/{label}_gate_mean_abs"] = self.record[
                f"sidecar_{label}_gate_mean_abs"
            ]
            pair = torch.zeros(2, device=next(self.dit.parameters()).device)
            for stage in range(len(self.dit.sidecar_stages)):
                total, count = self.ratios.get((f"sidecar_{label}", stage), (0, 0))
                stage_pair = torch.zeros_like(pair)
                stage_pair[0], stage_pair[1] = total, count
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(stage_pair)
                self.record[f"sidecar_{label}_residual_ratio_stage_{stage}"] = (
                    float(stage_pair[0] / stage_pair[1]) if stage_pair[1] else None
                )
                pair += stage_pair
            self.record[f"sidecar/{label}_residual_ratio"] = (
                float(pair[0] / pair[1]) if pair[1] else None
            )

    def pre_step(self):
        if hasattr(self.dit, "sidecar_stages"):
            self._sidecar_pre_step()
            return
        if not self.sampled() or not any(
            hasattr(self.dit, f"{modality}_gates") for modality in ("point", "tactile")
        ):
            return
        step = self.trainer.state.global_step + 1
        if step == self.last_step:
            return
        self.last_step = step
        self.before = {}
        self.record = {"global_step": step, "layers": []}
        for modality in ("point", "tactile"):
            for name in (
                "gate_mean_abs",
                "gate_max_abs",
                "gate_grad_norm",
                "gate_update_norm",
                "gate_sample_delta_norm",
                "gate_near_zero_fraction",
                "encoder_grad_norm",
                "cross_attention_grad_norm",
                "residual_ratio_mean",
                "residual_ratio_max",
            ):
                self.record[f"{modality}_{name}"] = None
        if self.level == "per_layer":
            count = max(
                (
                    getattr(gate, "ds_numel", gate.numel())
                    for modality in ("point", "tactile")
                    if (gate := getattr(self.dit, f"{modality}_gates", None)) is not None
                ),
                default=0,
            )
            self.record["layers"] = [
                {
                    "layer": i,
                    **{
                        f"{modality}_{field}": None
                        for modality in ("point", "tactile")
                        for field in ("gate", "gate_grad", "gate_delta", "residual_ratio")
                    },
                }
                for i in range(count)
            ]
        optimizer = self._zero_optimizer or self.trainer.optimizer
        original_groups = getattr(optimizer, "trainable_param_groups", None)
        if original_groups is not None:
            owned = {id(p) for group in original_groups for p in group["params"]}
        elif hasattr(optimizer, "bit16_groups"):
            owned = {id(p) for group in optimizer.bit16_groups for p in group}
        else:
            owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
        scale = float(getattr(optimizer, "loss_scale", 1.0))
        self.record["gradient_scale"] = scale
        self.record["gate_value_source"] = (
            "model parameter (gathered under ZeRO-3), not optimizer master"
        )
        self.record["gradient_timing"] = (
            "before DeepSpeed step/unscale/clip"
            if self._zero_optimizer
            else "after Trainer unscale/clip, before optimizer step"
        )
        if self._zero_optimizer is not None:
            logging.getLogger(__name__).info(
                "ZeRO diagnostic norms use gradient buffers; unused and zero gradients may be indistinguishable."
            )
        for modality in ("point", "tactile"):
            gate = getattr(self.dit, f"{modality}_gates", None)
            if gate is None:
                continue
            values, grads = full_value(gate), full_value(gate, gradient=True)
            self.before[modality] = values
            stats = gradient_stats([gate], scale)
            if grads is not None:
                grads /= scale
            if values is None or (gate.requires_grad and grads is None):
                logging.warning(
                    "%s gate parameter/gradient unavailable at optimizer boundary", modality
                )
            self.record.update({f"{modality}_gate_{k}": v for k, v in stats.items()})
            self.record[f"{modality}_gate_trainable"] = gate.requires_grad
            self.record[f"{modality}_gate_in_optimizer"] = id(gate) in owned
            if values is None:
                logging.warning(
                    "%s gate values unavailable through safe parameter access", modality
                )
                continue
            finite = bool(torch.isfinite(values).all())
            if not finite or stats["grad_finite"] is False:
                logging.warning(
                    "Non-finite %s gate parameter or gradient at optimizer step %s", modality, step
                )
            self.record.update(
                {
                    f"{modality}_gate_mean_abs": float(values.abs().mean()) if finite else None,
                    f"{modality}_gate_max_abs": float(values.abs().max()) if finite else None,
                    f"{modality}_gate_near_zero_fraction": float(
                        (values.abs() <= 1e-8).float().mean()
                    )
                    if finite
                    else None,
                    f"{modality}_gate_finite": finite,
                }
            )
            ratios = []
            # All ranks reduce every layer, even when a rank has no sensor.
            pairs = torch.zeros(values.numel(), 2, device=collective_device(gate))
            for layer in range(values.numel()):
                total, count = self.ratios.get((modality, layer), (None, 0))
                if count:
                    pairs[layer, 0] = total.detach()
                    pairs[layer, 1] = count
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(pairs)
            for layer, (value, pair) in enumerate(zip(values.flatten(), pairs.cpu())):
                ratio = float(pair[0] / pair[1]) if pair[1] else None
                if ratio is not None:
                    ratios.append(ratio)
                if self.level == "per_layer":
                    while len(self.record["layers"]) <= layer:
                        self.record["layers"].append({"layer": len(self.record["layers"])})
                    self.record["layers"][layer].update(
                        {
                            f"{modality}_gate": float(value),
                            f"{modality}_gate_grad": None
                            if grads is None
                            else float(grads.flatten()[layer]),
                            f"{modality}_residual_ratio": ratio,
                            f"{modality}_residual_finite": None
                            if ratio is None
                            else math.isfinite(ratio),
                            f"{modality}_gate_trainable": gate.requires_grad,
                            f"{modality}_gate_finite": bool(torch.isfinite(value)),
                            f"{modality}_gate_grad_finite": None
                            if grads is None
                            else bool(torch.isfinite(grads.flatten()[layer])),
                            f"{modality}_branch_trainable": any(
                                p.requires_grad
                                for p in getattr(
                                    self.dit, f"{modality}_cross_attention"
                                ).parameters()
                            ),
                        }
                    )
            self.record[f"{modality}_residual_ratio_mean"] = (
                sum(ratios) / len(ratios) if ratios else None
            )
            self.record[f"{modality}_residual_ratio_max"] = max(ratios) if ratios else None
            head = getattr(self.trainer.model, "action_head", None)
            for suffix, module in (
                ("encoder", getattr(head, f"{modality}_encoder", None)),
                ("cross_attention", getattr(self.dit, f"{modality}_cross_attention", None)),
            ):
                params = list(module.parameters()) if module is not None else []
                self.record.update(
                    {
                        f"{modality}_{suffix}_{k}": v
                        for k, v in gradient_stats(params, scale).items()
                    }
                )
                if self.record[f"{modality}_{suffix}_grad_finite"] is False:
                    logging.warning(
                        "Non-finite %s %s gradients at optimizer step %s", modality, suffix, step
                    )
                if params and self.record[f"{modality}_{suffix}_grad_norm"] is None:
                    logging.warning(
                        "%s %s gradient norm unavailable at sampled optimizer boundary",
                        modality,
                        suffix,
                    )
                if suffix == "cross_attention":
                    for row in self.record["layers"]:
                        row[f"{modality}_branch_grad_finite"] = self.record[
                            f"{modality}_{suffix}_grad_finite"
                        ]
                self.record[f"{modality}_{suffix}_trainable"] = any(p.requires_grad for p in params)
                self.record[f"{modality}_{suffix}_in_optimizer"] = any(
                    p.requires_grad for p in params
                ) and all(id(p) in owned for p in params if p.requires_grad)

    def post_step(self):
        if self.record is None:
            return
        for modality, before in self.before.items():
            after = full_value(getattr(self.dit, f"{modality}_gates"))
            if before is None or after is None:
                continue
            delta = after - before
            previous = self.previous.get(modality)
            self.record[f"{modality}_gate_update_norm"] = float(delta.norm())
            self.record[f"{modality}_gate_sample_delta_norm"] = (
                None if previous is None else float((after - previous).norm())
            )
            self.previous[modality] = after
            for layer in self.record["layers"]:
                layer[f"{modality}_gate_delta"] = float(delta.flatten()[layer["layer"]])
                layer[f"{modality}_gate_sample_delta"] = (
                    None
                    if previous is None
                    else float((after - previous).flatten()[layer["layer"]])
                )

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if not self.trainer.is_deepspeed_enabled:
            self.pre_step()

    def on_optimizer_step(self, args, state, control, **kwargs):
        if not self.trainer.is_deepspeed_enabled:
            self.post_step()

    def install_deepspeed(self, engine):
        if self._wrapped is not None:
            return
        optimizer = engine.optimizer
        self._zero_optimizer = optimizer
        if any(hasattr(p, "ds_id") for p in self.trainer.model.parameters()):
            logging.warning(
                "ZeRO-3 diagnostics: gradient buffers do not prove autograd participation; grad_present is unavailable, see grad_buffer_present and grad_status."
            )
        original = optimizer.step

        def step(*args, **kwargs):
            self.pre_step()
            result = original(*args, **kwargs)
            self.post_step()
            return result

        self._wrapped = optimizer, original
        optimizer.step = step

    def close(self):
        if self._wrapped is not None:
            optimizer, original = self._wrapped
            optimizer.step = original
            self._wrapped = None
        if self.dit is not None:
            self.dit._diagnostic_observer = None

    def on_step_end(self, args, state, control, **kwargs):
        if self.record is None:
            return
        record, self.record = self.record, None

        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clean(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value):
                logging.warning("Non-finite multimodal diagnostic at step %s", state.global_step)
                return None
            return value

        record = clean(record)
        if self.trainer.is_world_process_zero():
            path = Path(args.output_dir) / "multimodal_diagnostics.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.trainer.log(
            {
                k: v
                for k, v in record.items()
                if (isinstance(v, (float, int)) or v is None) and k != "global_step"
            }
        )
