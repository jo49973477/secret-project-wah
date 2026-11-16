"""Small independent bidirectional stages; masks use True for valid tokens."""

import math

import torch
from torch import nn

from gr00t.configs.sidecar_config import (
    default_action_gate_init,
    default_point_gate_init,
    validate_gate_init,
    validate_gate_init_seed,
)
from gr00t.model.extension.multimodal_dit import ModalityCrossAttention


def gate_initial_value(settings, *, default, seed, stage, kind):
    """Sample one deterministic scalar without advancing PyTorch's global RNG."""
    settings = validate_gate_init(settings, default=default, name=f"sidecar_{kind}_gate_init")
    seed = validate_gate_init_seed(seed)
    if isinstance(stage, bool) or not isinstance(stage, int) or stage < 0:
        raise ValueError("Sidecar gate stage index must be a nonnegative integer")
    mode = settings["mode"]
    if mode == "zero":
        return 0.0, settings
    if mode == "one":
        return 1.0, settings
    if mode == "constant":
        return settings["value"], settings
    stream = 0 if kind == "point" else 1 if kind == "action" else None
    if stream is None:
        raise ValueError(f"Unknown Sidecar gate kind: {kind!r}")
    generator = torch.Generator(device="cpu")
    derived_seed = (seed + 1_000_003 * (stage + 1) + 97_409 * (stream + 1)) % 2**63
    generator.manual_seed(derived_seed)
    sample = torch.empty((), dtype=torch.float32)
    if mode == "uniform":
        sample.uniform_(settings["low"], settings["high"], generator=generator)
    else:
        sample.normal_(settings["mean"], settings["std"], generator=generator)
    return float(sample), settings


def gate_parameter(settings, *, default, seed, stage, kind):
    value, resolved = gate_initial_value(
        settings, default=default, seed=seed, stage=stage, kind=kind
    )
    parameter = nn.Parameter(torch.tensor(value, dtype=torch.float32))
    return parameter, value, resolved


def stage_indices(depth, count=4, indices=None):
    if depth < 1 or count < 1:
        raise ValueError("depth and stage count must be positive")
    result = (
        list(indices)
        if indices is not None
        else [math.ceil((i + 1) * depth / min(count, depth)) - 1 for i in range(min(count, depth))]
    )
    if not result or result != sorted(set(result)) or result[0] < 0 or result[-1] >= depth:
        raise ValueError("stage indices must be unique, increasing, zero-based block indices")
    return result


class TokenPool(nn.Module):
    def __init__(self, input_dim, dim, count, heads):
        super().__init__()
        if count < 1:
            raise ValueError("token count must be positive")
        self.queries = nn.Parameter(torch.randn(count, dim) * 0.02)
        self.projection = nn.Linear(input_dim, dim)
        self.attention = ModalityCrossAttention(dim, dim, heads)

    def forward(self, tokens, mask=None):
        tokens = self.projection(tokens)
        queries = self.queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        output = self.attention(queries, encoder_hidden_states=tokens, encoder_attention_mask=mask)
        valid = torch.ones(output.shape[:2], dtype=torch.bool, device=output.device)
        if mask is not None:
            valid = valid & mask.bool().any(-1, keepdim=True)
        return output, valid


class PointActionStage(nn.Module):
    def __init__(
        self,
        action_dim,
        dim,
        heads,
        *,
        point_gate_init=None,
        action_gate_init=None,
        gate_init_seed=42,
        stage_index=0,
    ):
        super().__init__()
        self.action_projection = nn.Linear(action_dim, dim)
        self.output_projection = nn.Linear(dim, action_dim)
        self.action_norm = nn.LayerNorm(dim)
        self.point_norm = nn.LayerNorm(dim)
        self.updated_point_norm = nn.LayerNorm(dim)
        self.point_reads_action = ModalityCrossAttention(dim, dim, heads)
        self.action_reads_point = ModalityCrossAttention(dim, dim, heads)
        self.point_gate, self.initial_point_gate, self.point_gate_init = gate_parameter(
            point_gate_init,
            default=default_point_gate_init,
            seed=gate_init_seed,
            stage=stage_index,
            kind="point",
        )
        self.action_gate, self.initial_action_gate, self.action_gate_init = gate_parameter(
            action_gate_init,
            default=default_action_gate_init,
            seed=gate_init_seed,
            stage=stage_index,
            kind="action",
        )
        self.last_diagnostics = {}

    def forward(self, action, points, point_mask, action_mask=None, observer=None, stage=0):
        query = self.action_norm(self.action_projection(action))
        delta = self.point_reads_action(
            self.point_norm(points), encoder_hidden_states=query, encoder_attention_mask=action_mask
        )
        point_update = self.point_gate * delta * point_mask[..., None].to(points.dtype)
        if observer is not None:
            observer("sidecar_point", stage, points, point_update)
        points = (points + point_update) * point_mask[..., None].to(points.dtype)
        delta = self.action_reads_point(
            query,
            encoder_hidden_states=self.updated_point_norm(points),
            encoder_attention_mask=point_mask,
        )
        pre_gate_update = self.output_projection(delta)
        valid = point_mask.any(-1)[:, None, None].to(pre_gate_update.dtype)
        pre_gate_update = pre_gate_update * valid
        update = self.action_gate * pre_gate_update
        result = action + update
        if observer is not None:
            actual_residual = result - action
            action_norm = action.detach().float().norm()
            self.last_diagnostics = dict(
                action_gate_initial_value=self.initial_action_gate,
                action_pre_gate_update_norm=float(pre_gate_update.detach().float().norm()),
                action_gated_residual_norm=float(update.detach().float().norm()),
                action_actual_residual_norm=float(actual_residual.detach().float().norm()),
                action_injection_ratio=float(
                    actual_residual.detach().float().norm() / (action_norm + 1e-12)
                ),
                action_bf16_loss_norm=(
                    float((update.detach().float() - actual_residual.detach().float()).norm())
                    if action.dtype == torch.bfloat16
                    else None
                ),
            )
            observer("sidecar_action", stage, action, actual_residual)
        return result, points
