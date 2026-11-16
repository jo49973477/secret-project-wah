"""Opt-in evolving sidecar. Original DiT modules retain their names and implementation."""

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from gr00t.model.extension.contact_input_moe import ContactInputAdapter
from gr00t.model.extension.contact_surprise import ContactInjection, ContactSurprise
from gr00t.model.extension.dit_lora import insert_dit_lora
from gr00t.model.extension.multimodal_dit import MultiModalConditionedDiT
from gr00t.model.extension.point_action_sidecar import PointActionStage, TokenPool, stage_indices
from gr00t.model.modules.dit import (
    AlternateVLDiT,
    _canonical_hidden_attention_mask,
    _mask_hidden_states,
)


class SidecarConditionedDiT(AlternateVLDiT):
    def __init__(
        self,
        *,
        sidecar_variant="legacy",
        sidecar_point_stage_names=None,
        sidecar_point_stage_dims=None,
        sidecar_local_patch_size=64,
        sidecar_window_chunk_size=16,
        sidecar_num_stages=4,
        sidecar_stage_indices=None,
        sidecar_hidden_dim=256,
        sidecar_num_heads=4,
        sidecar_num_point_tokens=32,
        tactile_fusion_mode="surprise",
        contact_num_tokens=4,
        contact_hidden_dim=256,
        freeze_pretrained_dit=True,
        freeze_contact_teacher=True,
        contact_injection_enabled=True,
        dit_training_mode="frozen",
        dit_lora_rank=16,
        dit_lora_alpha=32,
        dit_lora_dropout=0.0,
        contact_input_mode="none",
        contact_state_dim=132,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if tactile_fusion_mode not in {"none", "direct", "contact_gate", "surprise", "hybrid"}:
            raise ValueError("Unknown tactile fusion mode")
        if freeze_pretrained_dit:
            self.requires_grad_(False)
        self.stage_indices = stage_indices(
            len(self.transformer_blocks), sidecar_num_stages, sidecar_stage_indices
        )
        if sidecar_variant not in {"legacy", "multilevel_local"}:
            raise ValueError(f"Unknown sidecar_variant: {sidecar_variant}")
        self.sidecar_variant = sidecar_variant
        self.sidecar_point_stage_names = list(sidecar_point_stage_names or [])
        if sidecar_variant == "legacy":
            self.sidecar_pool = TokenPool(
                self.inner_dim, sidecar_hidden_dim, sidecar_num_point_tokens, sidecar_num_heads
            )
            self.sidecar_stages = nn.ModuleList(
                PointActionStage(self.inner_dim, sidecar_hidden_dim, sidecar_num_heads)
                for _ in self.stage_indices
            )
        else:
            from gr00t.model.extension.multilevel_local_sidecar import LocalJointActionStage

            if tactile_fusion_mode != "none" or contact_input_mode != "none":
                raise ValueError(
                    "multilevel_local requires tactile_fusion_mode=none and contact_input_mode=none"
                )
            if (
                len(self.sidecar_point_stage_names) != len(self.stage_indices)
                or len(set(self.sidecar_point_stage_names)) != len(self.stage_indices)
                or sidecar_point_stage_dims is None
                or len(sidecar_point_stage_dims) != len(self.stage_indices)
            ):
                raise ValueError(
                    "multilevel_local requires validated distinct point stage names and dimensions"
                )
            self.sidecar_pool = nn.Identity()  # no pooling parameters or pooled tokens
            self.sidecar_stages = nn.ModuleList(
                LocalJointActionStage(
                    self.inner_dim,
                    point_dim,
                    sidecar_hidden_dim,
                    sidecar_num_heads,
                    sidecar_local_patch_size,
                    sidecar_window_chunk_size,
                )
                for point_dim in sidecar_point_stage_dims
            )
        self.contact_adapter = ContactSurprise(
            self.inner_dim,
            contact_hidden_dim,
            contact_num_tokens,
            sidecar_num_heads,
            freeze_contact_teacher,
        )
        self.contact_injections = nn.ModuleList(
            ContactInjection(self.inner_dim, contact_hidden_dim, sidecar_num_heads)
            for _ in self.stage_indices
        )
        self.tactile_fusion_mode = tactile_fusion_mode
        self.contact_injection_enabled = contact_injection_enabled
        self.freeze_contact_teacher = freeze_contact_teacher
        self.freeze_pretrained_dit = freeze_pretrained_dit
        self.dit_training_mode = dit_training_mode
        if dit_training_mode not in {"frozen", "lora", "full"}:
            raise ValueError(f"Invalid DiT training mode: {dit_training_mode}")
        if dit_training_mode == "lora":
            insert_dit_lora(self, dit_lora_rank, dit_lora_alpha, dit_lora_dropout)
        self.input_contact_adapter = (
            ContactInputAdapter(
                self.inner_dim,
                contact_state_dim,
                mode=contact_input_mode,
                contact_dim=contact_hidden_dim,
                num_tokens=contact_num_tokens,
                use_surprise=False,
                diagnostics=True,
            )
            if contact_input_mode != "none"
            else None
        )

    _run_original_gr00t_block = MultiModalConditionedDiT._run_original_gr00t_block

    def set_multimodal_adapter_trainable(self, trainable):
        for module in (
            self.sidecar_pool,
            self.sidecar_stages,
            self.contact_adapter,
            self.contact_injections,
            self.input_contact_adapter,
        ):
            if module is not None:
                module.requires_grad_(trainable)
        self.contact_adapter.projector.requires_grad_(trainable and not self.freeze_contact_teacher)
        if self.tactile_fusion_mode == "none":
            self.contact_adapter.requires_grad_(False)
            self.contact_injections.requires_grad_(False)
        elif self.tactile_fusion_mode == "direct":
            self.contact_adapter.gate.requires_grad_(False)
        if not self.contact_injection_enabled:
            self.contact_injections.requires_grad_(False)
            self.contact_adapter.gate.requires_grad_(False)

    def set_multimodal_adapter_mode(self, training):
        for module in (
            self.sidecar_pool,
            self.sidecar_stages,
            self.contact_adapter,
            self.contact_injections,
            self.input_contact_adapter,
        ):
            if module is not None:
                module.train(training)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
        *,
        point_tokens: Optional[torch.Tensor] = None,
        tactile_tokens: Optional[torch.Tensor] = None,
        point_attention_mask: Optional[torch.Tensor] = None,
        tactile_attention_mask: Optional[torch.Tensor] = None,
        num_state_tokens: int = 1,
        robot_state: Optional[torch.Tensor] = None,
        hidden_attention_mask: Optional[torch.Tensor] = None,
    ):
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [B, N_state + N_action, D_dit], "
                f"got {tuple(hidden_states.shape)}."
            )
        batch_size = hidden_states.shape[0]
        if hidden_states.shape[-1] != self.inner_dim:
            raise ValueError(
                f"Expected hidden_states width {self.inner_dim}, got {hidden_states.shape[-1]}."
            )
        if encoder_hidden_states.ndim != 3:
            raise ValueError(
                "encoder_hidden_states must have shape [B, N_vlm, D_vlm], "
                f"got {tuple(encoder_hidden_states.shape)}."
            )
        if encoder_hidden_states.shape[0] != batch_size:
            raise ValueError("hidden_states and encoder_hidden_states must share a batch size.")
        if not 0 <= num_state_tokens < hidden_states.shape[1]:
            raise ValueError(
                "num_state_tokens must leave at least one action token. "
                f"Got {num_state_tokens} for a sequence of length {hidden_states.shape[1]}."
            )
        if image_mask is None or backbone_attention_mask is None:
            raise ValueError("image_mask and backbone_attention_mask are required")
        if point_tokens is None and point_attention_mask is not None:
            raise ValueError("point_attention_mask was provided without point_tokens.")
        if tactile_tokens is None and tactile_attention_mask is not None:
            raise ValueError("tactile_attention_mask was provided without tactile_tokens.")

        # Preserve AlternateVLDiT's image/non-image attention masks and schedule.
        valid_vlm_tokens = backbone_attention_mask.to(dtype=torch.bool)
        image_tokens = image_mask.to(dtype=torch.bool)
        image_attention_mask = image_tokens & valid_vlm_tokens
        non_image_attention_mask = (~image_tokens) & valid_vlm_tokens
        if not self.config.interleave_self_attention:
            raise ValueError(
                "MultiModalConditionedDiT preserves AlternateVLDiT and requires "
                "interleave_self_attention=True"
            )

        if self.sidecar_variant == "multilevel_local" and point_tokens is not None:
            if not isinstance(point_tokens, dict) or set(point_tokens) != set(
                self.sidecar_point_stage_names
            ):
                raise ValueError(
                    "multilevel_local point tokens must contain the configured stage mapping"
                )
            if point_attention_mask is not None:
                raise ValueError("Packed multilevel stages use offsets, not point_attention_mask")
            for name in self.sidecar_point_stage_names:
                if len(point_tokens[name]["offset"]) != batch_size:
                    raise ValueError("Multilevel point batch size does not match actions")
        if (
            self.sidecar_variant == "legacy"
            and point_tokens is not None
            and point_tokens.shape[0] != batch_size
        ):
            raise ValueError("hidden_states and point_tokens must share a batch size.")
        if tactile_tokens is not None and tactile_tokens.shape[0] != batch_size:
            raise ValueError("hidden_states and tactile_tokens must share a batch size.")
        # Point and tactile encoders have independent compute islands. Join the
        # DiT activation stream explicitly before LayerNorm and MHA Q/K/V.
        if self.sidecar_variant == "legacy" and point_tokens is not None:
            point_tokens = point_tokens.to(dtype=hidden_states.dtype)
        if tactile_tokens is not None:
            tactile_tokens = tactile_tokens.to(dtype=hidden_states.dtype)
        for token_name, tokens in (
            ("point_tokens", point_tokens),
            ("tactile_tokens", tactile_tokens),
        ):
            if (
                tokens is not None
                and not isinstance(tokens, dict)
                and tokens.shape[-1] != self.inner_dim
            ):
                raise ValueError(
                    f"{token_name} width must be {self.inner_dim}, got {tokens.shape[-1]}."
                )

        timestep_embedding = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        hidden_attention_mask = _canonical_hidden_attention_mask(
            hidden_attention_mask, hidden_states
        )
        hidden_states = _mask_hidden_states(hidden_states, hidden_attention_mask)
        vlm_hidden_states = encoder_hidden_states.contiguous()
        all_hidden_states = [hidden_states]

        # All evolving values are local: never cache action-dependent point tokens.
        points = point_mask = None
        if point_tokens is not None:
            if self.sidecar_variant == "legacy":
                points, point_mask = self.sidecar_pool(point_tokens, point_attention_mask)
            else:
                points = point_tokens
        contact_tokens = contact_mask = None
        self.contact_diagnostics = {}
        self.contact_prediction_loss = hidden_states.new_zeros((), dtype=torch.float32)
        if self.tactile_fusion_mode != "none" and tactile_tokens is not None:
            if point_tokens is None or robot_state is None:
                raise ValueError("Contact conditioning requires scene points and observation state")
            observed, predicted, surprise, gate, contact_mask, loss = self.contact_adapter(
                point_tokens,
                robot_state.to(hidden_states.dtype),
                tactile_tokens,
                point_attention_mask,
                tactile_attention_mask,
            )
            self.contact_prediction_loss = loss
            if getattr(self, "_diagnostic_observer", None) is not None:
                self.contact_diagnostics = {
                    "contact_prediction_loss": loss.detach(),
                    "observed_contact_norm": observed.detach().float().norm(dim=-1).mean(),
                    "observed_contact_variance": observed.detach()
                    .float()
                    .var(dim=(0, 1), unbiased=False)
                    .mean(),
                    "predicted_contact_variance": predicted.detach()
                    .float()
                    .var(dim=(0, 1), unbiased=False)
                    .mean(),
                    "contact_surprise_norm": surprise.detach().float().norm(dim=-1).mean(),
                    "contact_gate_mean": gate.detach().float().mean(),
                }
            mode = self.tactile_fusion_mode
            contact_tokens = (
                observed
                if mode == "direct"
                else observed * gate
                if mode == "contact_gate"
                else surprise * gate
            )
            if mode == "hybrid":
                contact_tokens = torch.cat((observed, surprise * gate), 1)
                contact_mask = torch.cat((contact_mask, contact_mask), 1)
        for block_index in range(len(self.transformer_blocks)):
            args = (
                block_index,
                hidden_states,
                vlm_hidden_states,
                timestep_embedding,
                image_attention_mask,
                non_image_attention_mask,
                hidden_attention_mask,
            )
            # Frozen backbone eval mode must not disable activation recomputation.
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    self._run_original_gr00t_block, *args
                )
            else:
                hidden_states = self._run_original_gr00t_block(*args)
            hidden_states = _mask_hidden_states(hidden_states, hidden_attention_mask)
            if block_index in self.stage_indices:
                stage = self.stage_indices.index(block_index)
                action = hidden_states[:, num_state_tokens:]
                if points is not None:
                    action_mask = (
                        None
                        if hidden_attention_mask is None
                        else hidden_attention_mask[:, num_state_tokens:]
                    )
                    if self.sidecar_variant == "legacy":
                        action, points = self.sidecar_stages[stage](
                            action,
                            points,
                            point_mask,
                            action_mask,
                            getattr(self, "_diagnostic_observer", None),
                            stage,
                        )
                    else:
                        name = self.sidecar_point_stage_names[stage]
                        local_args = (
                            action,
                            points[name],
                            None,
                            action_mask,
                            getattr(self, "_diagnostic_observer", None),
                            stage,
                        )
                        action, _ = self.sidecar_stages[stage](
                            *local_args,
                            checkpoint_activations=self.gradient_checkpointing,
                        )
                if contact_tokens is not None and self.contact_injection_enabled:
                    before_contact = action
                    action = self.contact_injections[stage](action, contact_tokens, contact_mask)
                    observer = getattr(self, "_diagnostic_observer", None)
                    if observer is not None:
                        observer(
                            "contact_injection", stage, before_contact, action - before_contact
                        )
                hidden_states = torch.cat((hidden_states[:, :num_state_tokens], action), 1)
                hidden_states = _mask_hidden_states(hidden_states, hidden_attention_mask)
            all_hidden_states.append(hidden_states)

        # Preserve the original DiT adaptive output normalization and projection.
        shift, scale = self.proj_out_1(F.silu(timestep_embedding)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale[:, None]) + shift[:, None]
        hidden_states = _mask_hidden_states(hidden_states, hidden_attention_mask)
        output = self.proj_out_2(hidden_states)
        output = _mask_hidden_states(output, hidden_attention_mask)

        if return_all_hidden_states:
            return output, all_hidden_states
        return output
