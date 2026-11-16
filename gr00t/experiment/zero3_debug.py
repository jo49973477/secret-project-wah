"""Opt-in autograd participation and local FP32 optimizer-update evidence."""

import json
import logging
from pathlib import Path

import torch


class Zero3GradientAudit:
    """Observe real autograd hooks; ZeRO's allocated gradient buffers are insufficient."""

    def __init__(self, model, output_dir):
        self.parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.seen = set()
        self.invalid = set()
        self.nonzero = set()
        self.step = 0
        self.output_dir = Path(output_dir)
        self.handles = []
        self.inactive = {}
        head = getattr(model, "action_head", None)
        if head is not None:
            from gr00t.model.extension.sidecar_dit import SidecarConditionedDiT

            if (
                isinstance(getattr(head, "model", None), SidecarConditionedDiT)
                and head.config.tactile_fusion_mode == "none"
                and head.config.contact_input_mode == "none"
            ):
                # Match _encode_multimodal_inputs' explicit disabled tactile path.
                # Observe/report these parameters without changing requires_grad.
                self.inactive = {
                    n: "tactile_fusion_mode=none and contact_input_mode=none"
                    for n in self.parameters
                    if n.startswith("action_head.tactile_encoder.")
                }

    def _register_hooks(self):
        for name, parameter in self.parameters.items():

            def observe(parameter, name=name):
                self.seen.add(name)
                gradient = parameter.grad
                if gradient is not None:
                    if not torch.isfinite(gradient).all():
                        self.invalid.add(name)
                    if torch.count_nonzero(gradient):
                        self.nonzero.add(name)

            self.handles.append(parameter.register_post_accumulate_grad_hook(observe))

    def install(self, engine):
        if self.step or getattr(self, "installed", False):
            return
        self.installed = True
        current = {n: p for n, p in engine.module.named_parameters() if p.requires_grad}
        replaced = [n for n, p in current.items() if self.parameters.get(n) is not p]
        logging.warning("zero3-gradient-audit initialization replaced_parameters=%s", replaced)
        self.parameters = current
        self._register_hooks()
        checkpoint_modules = [
            n
            for n, module in engine.module.named_modules()
            if getattr(module, "gradient_checkpointing", False)
        ]
        logging.warning(
            "zero3-gradient-audit stage=%d bf16=%s checkpoint_modules=%s inactive_parameters=%s",
            engine.zero_optimization_stage(),
            engine.bfloat16_enabled(),
            checkpoint_modules,
            self.inactive,
        )
        optimizer = engine.optimizer
        original_step = optimizer.step

        def samples():
            from deepspeed.utils import safe_get_local_fp32_param

            result = {}
            for name, parameter in self.parameters.items():
                value = safe_get_local_fp32_param(parameter).detach().flatten()
                # Bound debug memory; include samples throughout category-specific weights.
                stride = max(1, value.numel() // 1024)
                result[name] = value[::stride][:1024].clone()
            return result

        def audited_step(*args, **kwargs):
            from deepspeed.utils import safe_get_local_grad

            before = samples()
            missing = sorted(self.parameters.keys() - self.seen)
            missing_expected = sorted(set(missing) - self.inactive.keys())
            # Reduction can clear param.grad before our post-accumulate observer.
            # These buffers establish numerical validity, while the hooks above
            # independently establish genuine autograd participation.
            for name, parameter in self.parameters.items():
                gradient = safe_get_local_grad(parameter)
                if gradient is not None:
                    if not torch.isfinite(gradient).all():
                        self.invalid.add(name)
                    if torch.count_nonzero(gradient):
                        self.nonzero.add(name)
            invalid = sorted(self.invalid)
            result = original_step(*args, **kwargs)
            after = samples()
            changed = [name for name in before if not torch.equal(before[name], after[name])]
            self.step += 1
            rank = torch.distributed.get_rank()
            record = dict(
                rank=rank,
                optimizer_step=self.step,
                trainable_parameters=len(self.parameters),
                autograd_seen=len(self.seen),
                nonzero_gradients=len(self.nonzero),
                missing_gradients=missing,
                missing_expected_gradients=missing_expected,
                inactive_parameters=self.inactive,
                nonfinite_gradients=invalid,
                sampled_updated_parameters=changed,
                fp32_samples_finite=all(bool(torch.isfinite(v).all()) for v in after.values()),
            )
            with (self.output_dir / f"zero3_gradient_audit_rank{rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            logging.warning(
                "zero3-gradient-audit rank=%d step=%d seen=%d/%d nonzero=%d missing=%s invalid=%s sampled_updates=%d finite=%s",
                rank,
                self.step,
                len(self.seen),
                len(self.parameters),
                len(self.nonzero),
                missing_expected,
                invalid,
                len(changed),
                record["fp32_samples_finite"],
            )
            self.seen.clear()
            self.invalid.clear()
            self.nonzero.clear()
            if missing_expected or invalid or not record["fp32_samples_finite"]:
                raise RuntimeError("ZeRO-3 gradient audit failed; see rank-local JSONL")
            return result

        optimizer.step = audited_step

    def record_loss(self, loss):
        value = float(loss.detach().float())
        if not torch.isfinite(loss.detach()).all():
            raise RuntimeError("Non-finite training loss")
        rank = torch.distributed.get_rank()
        with (self.output_dir / f"zero3_loss_audit_rank{rank}.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(rank=rank, optimizer_step=self.step, loss=value)) + "\n")
