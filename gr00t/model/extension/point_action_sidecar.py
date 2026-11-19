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


def masked_mean(tokens, mask=None):
    """Differentiable per-sample mean over valid tokens."""
    if tokens.ndim != 3:
        raise ValueError("Adaptive gate pooling expects [batch, tokens, width]")
    if mask is None:
        return tokens.mean(dim=1)
    if mask.shape != tokens.shape[:2]:
        raise ValueError("Adaptive gate token mask shape mismatch")
    weights = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


class AdaptiveGate(nn.Module):
    """Sample-dependent scalar gate with a stable sigmoid-bias initialization."""

    def __init__(self, dim, hidden_dim, initial_value, *, seed=42, stage=0, kind="action"):
        super().__init__()
        if hidden_dim < 1 or not 0 < initial_value < 1:
            raise ValueError("Adaptive gate hidden_dim/value are invalid")
        self.norm = nn.LayerNorm(2 * dim)
        # Gate construction and initialization must not advance the backbone's
        # global RNG stream. This is especially important when ranks construct
        # modules independently before ZeRO-3 partitioning.
        cpu_rng = torch.random.get_rng_state()
        try:
            self.input = nn.Linear(2 * dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, 1)
        finally:
            torch.random.set_rng_state(cpu_rng)
        self.initial_value = float(initial_value)
        # A tiny nonzero final weight makes the initial gate input-sensitive while
        # the logit bias keeps values close to the requested stable target.
        if not self.output.weight.is_meta:
            generator = torch.Generator(device="cpu")
            stream = 0 if kind == "point" else 1
            generator.manual_seed(
                (validate_gate_init_seed(seed) + 1_000_003 * (stage + 1) + 97_409 * (stream + 1))
                % 2**63
            )
            input_weight = torch.empty(self.input.weight.shape, dtype=torch.float32)
            nn.init.kaiming_uniform_(input_weight, a=math.sqrt(5), generator=generator)
            input_bias = torch.empty(self.input.bias.shape, dtype=torch.float32)
            bound = 1 / math.sqrt(self.input.in_features)
            input_bias.uniform_(-bound, bound, generator=generator)
            output_weight = torch.empty(self.output.weight.shape, dtype=torch.float32)
            output_weight.normal_(mean=0.0, std=1e-3, generator=generator)
            with torch.no_grad():
                self.input.weight.copy_(input_weight.to(self.input.weight))
                self.input.bias.copy_(input_bias.to(self.input.bias))
                self.output.weight.copy_(output_weight.to(self.output.weight))
                self.output.bias.fill_(math.log(initial_value / (1 - initial_value)))

    def forward(self, action, points, action_mask=None, point_mask=None):
        return self.forward_pooled(
            masked_mean(action, action_mask), masked_mean(points, point_mask)
        )

    def forward_pooled(self, action, points):
        pooled = torch.cat((action, points), dim=-1)
        logits = self.output(torch.nn.functional.silu(self.input(self.norm(pooled))))
        return torch.sigmoid(logits).unsqueeze(-1)


def gate_statistics(value):
    samples = value.detach().float().reshape(-1)
    return float(samples.mean()), float(samples.std(unbiased=False))


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
        gate_mode="scalar",
        adaptive_gate_hidden_dim=128,
        adaptive_gate_init_action=0.01,
        adaptive_gate_init_point=0.8,
    ):
        super().__init__()
        self.action_projection = nn.Linear(action_dim, dim)
        self.output_projection = nn.Linear(dim, action_dim)
        self.action_norm = nn.LayerNorm(dim)
        self.point_norm = nn.LayerNorm(dim)
        self.updated_point_norm = nn.LayerNorm(dim)
        self.point_reads_action = ModalityCrossAttention(dim, dim, heads)
        self.action_reads_point = ModalityCrossAttention(dim, dim, heads)
        self.gate_mode = gate_mode
        if gate_mode == "scalar":
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
        elif gate_mode == "adaptive":
            self.initial_point_gate = float(adaptive_gate_init_point)
            self.initial_action_gate = float(adaptive_gate_init_action)
            self.point_gate_init = {"mode": "adaptive", "value": self.initial_point_gate}
            self.action_gate_init = {"mode": "adaptive", "value": self.initial_action_gate}
            self.point_gate_network = AdaptiveGate(
                dim,
                adaptive_gate_hidden_dim,
                self.initial_point_gate,
                seed=gate_init_seed,
                stage=stage_index,
                kind="point",
            )
            self.action_gate_network = AdaptiveGate(
                dim,
                adaptive_gate_hidden_dim,
                self.initial_action_gate,
                seed=gate_init_seed,
                stage=stage_index,
                kind="action",
            )
        else:
            raise ValueError(f"Unknown Sidecar gate mode: {gate_mode!r}")
        self._action_injection_enabled = True
        self._point_injection_enabled = True
        self.last_gate_values = {}
        self.last_diagnostics = {}

    def gate_parameters(self, kind):
        if kind not in {"action", "point"}:
            raise ValueError(f"Unknown gate kind: {kind}")
        if self.gate_mode == "scalar":
            return [getattr(self, f"{kind}_gate")]
        return list(getattr(self, f"{kind}_gate_network").parameters())

    def set_injection_enabled(self, kind, enabled):
        if kind not in {"action", "point"}:
            raise ValueError(f"Unknown gate kind: {kind}")
        setattr(self, f"_{kind}_injection_enabled", bool(enabled))

    def _gate(self, kind, action, points, action_mask, point_mask):
        if self.gate_mode == "scalar":
            value = getattr(self, f"{kind}_gate")
        else:
            value = getattr(self, f"{kind}_gate_network")(action, points, action_mask, point_mask)
        if not getattr(self, f"_{kind}_injection_enabled"):
            value = value * 0
        self.last_gate_values[kind] = value.detach()
        return value

    def forward(
        self,
        action,
        points,
        point_mask,
        action_mask=None,
        observer=None,
        stage=0,
        gradient_observer=None,
    ):
        query = self.action_norm(self.action_projection(action))
        delta = self.point_reads_action(
            self.point_norm(points), encoder_hidden_states=query, encoder_attention_mask=action_mask
        )
        point_gate = self._gate("point", query, points, action_mask, point_mask)
        point_valid = point_mask[..., None].to(points.dtype)
        point_pre_gate_update = delta * point_valid
        # Keep the legacy scalar multiplication order bit-for-bit compatible.
        point_update = point_gate * delta * point_valid
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
        action_gate = self._gate("action", query, points, action_mask, point_mask)
        update = action_gate * pre_gate_update
        result = action + update
        if gradient_observer is not None:
            gradient_observer(stage, pre_gate_update, result, action_gate)
        if observer is not None:
            actual_residual = result - action
            action_norm = action.detach().float().norm()
            action_gate_mean, action_gate_std = gate_statistics(action_gate)
            point_gate_mean, point_gate_std = gate_statistics(point_gate)
            self.last_diagnostics = dict(
                gate_mode=self.gate_mode,
                action_gate_value_mean=action_gate_mean,
                action_gate_value_std=action_gate_std,
                point_gate_value_mean=point_gate_mean,
                point_gate_value_std=point_gate_std,
                action_gate_initial_value=self.initial_action_gate,
                point_gate_initial_value=self.initial_point_gate,
                point_update_norm=float(point_update.detach().float().norm()),
                point_pre_gate_update_norm=float(point_pre_gate_update.detach().float().norm()),
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
