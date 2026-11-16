"""Stage-aligned serialized local joint attention; no global pooling or residual subtraction.

Inspired by PointACT's action replication into serialized point patches. This
implementation uses PyTorch SDPA with explicit padding and batch boundaries.
"""

import torch
from torch import nn
import torch.nn.functional as F

from gr00t.configs.sidecar_config import (
    default_action_gate_init,
    resolve_point_stages as resolve_point_stages,
    validate_checkpoint_variant as validate_checkpoint_variant,
    validate_variant as validate_variant,
)
from gr00t.model.extension.point_action_sidecar import gate_parameter


def snapshot_level(point, expected_dim):
    """Snapshot tensors, not mutable PTv3 Point objects; clone retains autograd."""
    level = {key: point[key].clone() for key in ("feat", "coord", "offset")}
    for key in ("grid_coord", "serialized_order"):
        if key in point:
            level[key] = point[key].clone()
    validate_level(level, expected_dim)
    return level


def validate_level(level, expected_dim, batch_size=None):
    feat, coord, offset = (level[k] for k in ("feat", "coord", "offset"))
    if feat.ndim != 2 or feat.shape[1] != expected_dim or coord.shape != (len(feat), 3):
        raise ValueError("Concerto stage feat/coord dimensions do not match the configured stage")
    if offset.ndim != 1 or not len(offset) or offset.dtype not in (torch.int32, torch.int64):
        raise ValueError("Concerto offset must be a nonempty integer vector")
    if batch_size is not None and len(offset) != batch_size:
        raise ValueError("Concerto offsets and action batch size disagree")
    counts = torch.diff(torch.cat((offset.new_zeros(1), offset)))
    if (counts <= 0).any() or int(offset[-1]) != len(feat):
        raise ValueError("Concerto offsets must be strictly increasing and cover all stage points")
    if not torch.isfinite(coord).all() or not torch.isfinite(feat).all():
        raise ValueError("Nonfinite Concerto stage features or coordinates")
    if any(value.device != feat.device for value in (coord, offset)):
        raise ValueError("Concerto stage tensors must share a device")
    order = level.get("serialized_order")
    if order is None or order.ndim != 2 or order.shape[1] != len(feat):
        raise ValueError("multilevel_local requires PTv3 serialized_order [orders, N]")
    first = order[0].long()
    if not torch.equal(first.sort().values, torch.arange(len(feat), device=feat.device)):
        raise ValueError("Concerto serialization must be a permutation of stage points")
    # PTv3 serialization encodes batch in the high bits; never allow a window
    # to read another environment even if an upstream implementation changes.
    start = 0
    for end in offset.tolist():
        if ((first[start:end] < start) | (first[start:end] >= end)).any():
            raise ValueError("Concerto serialized order crosses a batch boundary")
        start = end


class LocalJointActionStage(nn.Module):
    def __init__(
        self,
        action_dim,
        point_dim,
        dim,
        heads,
        patch_size=64,
        window_chunk_size=16,
        *,
        action_gate_init=None,
        gate_init_seed=42,
        stage_index=0,
    ):
        super().__init__()
        if min(patch_size, window_chunk_size, heads, dim) < 1 or dim % heads:
            raise ValueError("Invalid local attention dimensions or window sizes")
        self.point_dim, self.dim, self.heads = point_dim, dim, heads
        self.patch_size, self.window_chunk_size = patch_size, window_chunk_size
        self.action_projection = nn.Linear(action_dim, dim)
        self.point_projection = nn.Linear(point_dim, dim)
        self.position_projection = nn.Linear(3, dim)
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attention_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, action_dim)
        self.action_gate, self.initial_action_gate, self.action_gate_init = gate_parameter(
            action_gate_init,
            default=default_action_gate_init,
            seed=gate_init_seed,
            stage=stage_index,
            kind="action",
        )
        self.last_diagnostics = {}
        # Window counts vary across ranks: gather once per stage, not per chunk.
        self._z3_leaf = True

    def _joint(self, actions, points, valid):
        """Return attention branch outputs, before any action residual addition."""
        horizon = actions.shape[1]
        tokens = self.norm(torch.cat((actions, points), dim=1))
        q, k, v = (
            self.qkv(tokens)
            .reshape(tokens.shape[0], tokens.shape[1], 3, self.heads, self.dim // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=valid[:, None, None, :], dropout_p=0.0
        )
        # All queries (including point queries) participate in joint self-attention.
        attended = attended.transpose(1, 2).reshape(tokens.shape)
        return self.attention_projection(attended)[:, :horizon]

    def forward(
        self,
        action,
        level,
        point_mask=None,
        action_mask=None,
        observer=None,
        stage=0,
        *,
        checkpoint_activations=False,
    ):
        # The ZeRO leaf module's gather/release and backward hooks must surround
        # checkpointing, rather than being replayed by it. Replaying __call__
        # releases gathered weights at the end of recomputation while the
        # original autograd graph still needs them (e.g. output_projection).
        if checkpoint_activations and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(
                self._forward,
                action,
                level,
                point_mask,
                action_mask,
                observer,
                stage,
                use_reentrant=False,
            )
        return self._forward(action, level, point_mask, action_mask, observer, stage)

    def _forward(self, action, level, point_mask=None, action_mask=None, observer=None, stage=0):
        if point_mask is not None:
            raise ValueError(
                "Local stages use packed valid points and offsets, not a padded point mask"
            )
        feat, coord, offset = (level[k] for k in ("feat", "coord", "offset"))
        if len(offset) != action.shape[0] or feat.shape[-1] != self.point_dim:
            raise ValueError("Point stage batch/dimension mismatch")
        if action_mask is not None and action_mask.shape != action.shape[:2]:
            raise ValueError("action_mask must be [batch, horizon], True for valid tokens")
        # Coordinates have the exact same per-scene CenterShift as the backbone.
        # Keep absolute centered coordinates (metres), not window recentering.
        point = self.point_projection(feat.to(device=action.device, dtype=action.dtype))
        point = point + self.position_projection(coord.to(device=action.device, dtype=action.dtype))
        projected_action = self.action_projection(action)
        order = level["serialized_order"][0].to(device=action.device, dtype=torch.long)
        counts = torch.diff(torch.cat((offset.new_zeros(1), offset)))
        if (counts <= 0).any() or int(offset[-1]) != len(feat) or order.numel() != len(feat):
            raise ValueError(
                "Local attention requires positive point counts and offsets covering all points"
            )
        outputs, counts_by_batch = [], []
        start = 0
        for batch, end in enumerate(offset.tolist()):
            indices = order[start:end]
            if ((indices < start) | (indices >= end)).any():
                raise ValueError("Local window serialization crosses a batch boundary")
            count = end - start
            windows = (count + self.patch_size - 1) // self.patch_size
            counts = [min(self.patch_size, count - w * self.patch_size) for w in range(windows)]
            counts_by_batch.append(counts)
            total = None
            for chunk_start in range(0, windows, self.window_chunk_size):
                chunk_end = min(windows, chunk_start + self.window_chunk_size)
                selected = indices[chunk_start * self.patch_size : chunk_end * self.patch_size]
                values = point[selected]
                padded_size = (chunk_end - chunk_start) * self.patch_size
                values = F.pad(values, (0, 0, 0, padded_size - len(values))).reshape(
                    -1, self.patch_size, self.dim
                )
                point_valid = torch.arange(padded_size, device=action.device).reshape(
                    -1, self.patch_size
                ) < len(selected)
                actions = projected_action[batch : batch + 1].expand(len(values), -1, -1)
                am = (
                    torch.ones(actions.shape[:2], dtype=torch.bool, device=action.device)
                    if action_mask is None
                    else action_mask[batch : batch + 1].bool().expand(len(values), -1)
                )
                valid = torch.cat((am, point_valid), dim=1)
                update = self._joint(actions, values, valid)
                part = update.float().sum(0)
                total = part if total is None else total + part
            outputs.append((total / windows).to(action.dtype))
            start = end
        delta = self.output_projection(torch.stack(outputs))
        if action_mask is not None:
            delta = delta * action_mask[..., None].to(delta.dtype)
        update = self.action_gate * delta
        result = action + update
        if observer is not None:
            masked_action = (
                action if action_mask is None else action * action_mask[..., None].to(action.dtype)
            )
            actual_residual = result - action
            self.last_diagnostics = dict(
                point_counts=[sum(c) for c in counts_by_batch],
                window_counts=[len(c) for c in counts_by_batch],
                window_point_counts=counts_by_batch,
                point_feature_norm=float(feat.detach().float().norm()),
                action_feature_norm=float(masked_action.detach().float().norm()),
                action_gate=float(self.action_gate.detach()),
                action_gate_initial_value=self.initial_action_gate,
                action_pre_gate_update_norm=float(delta.detach().float().norm()),
                action_update_norm=float(delta.detach().float().norm()),
                action_gated_residual_norm=float(update.detach().float().norm()),
                action_residual_norm=float(update.detach().float().norm()),
                action_actual_residual_norm=float(actual_residual.detach().float().norm()),
                action_bf16_loss_norm=(
                    float((update.detach().float() - actual_residual.detach().float()).norm())
                    if action.dtype == torch.bfloat16
                    else None
                ),
                action_injection_ratio=float(
                    actual_residual.detach().float().norm()
                    / (masked_action.detach().float().norm() + 1e-12)
                ),
            )
            observer("sidecar_action", stage, masked_action, actual_residual)
        return result, level
