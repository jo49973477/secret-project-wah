"""Dependency-free Sidecar architecture contracts for config/preflight and loaders."""

import math
from numbers import Real


GATE_INIT_MODES = frozenset({"zero", "one", "constant", "uniform", "normal"})


def default_point_gate_init():
    return {
        "mode": "one",
        "value": 1.0,
        "low": 0.8,
        "high": 1.2,
        "mean": 1.0,
        "std": 0.05,
    }


def default_action_gate_init():
    return {
        "mode": "zero",
        "value": 0.02,
        "low": 0.01,
        "high": 0.05,
        "mean": 0.02,
        "std": 0.01,
    }


def validate_gate_init(settings, *, default, name):
    """Return a validated plain mapping without importing torch or mutating input."""
    settings = default() if settings is None else dict(settings)
    allowed = {"mode", "value", "low", "high", "mean", "std"}
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"{name} has unsupported keys: {sorted(unknown)}")
    mode = settings.get("mode", default()["mode"])
    if mode not in GATE_INIT_MODES:
        raise ValueError(f"{name}.mode must be one of {sorted(GATE_INIT_MODES)}; got {mode!r}")
    resolved = {**default(), **settings, "mode": mode}
    for key in ("value", "low", "high", "mean", "std"):
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name}.{key} must be a finite real number; got {value!r}")
        resolved[key] = float(value)
    if mode == "uniform" and not resolved["low"] < resolved["high"]:
        raise ValueError(f"{name}.low must be less than {name}.high for uniform initialization")
    if mode == "normal" and resolved["std"] <= 0:
        raise ValueError(f"{name}.std must be positive for normal initialization")
    return resolved


def validate_gate_init_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("sidecar_gate_init_seed must be an integer in [0, 2**63)")
    return seed


def validate_gate_initialization(config):
    point = validate_gate_init(
        getattr(config, "sidecar_point_gate_init", None),
        default=default_point_gate_init,
        name="sidecar_point_gate_init",
    )
    action = validate_gate_init(
        getattr(config, "sidecar_action_gate_init", None),
        default=default_action_gate_init,
        name="sidecar_action_gate_init",
    )
    seed = validate_gate_init_seed(getattr(config, "sidecar_gate_init_seed", 42))
    variant = getattr(config, "sidecar_variant", "legacy")
    if variant == "multilevel_local" and point["mode"] != "one":
        raise ValueError(
            "multilevel_local has no point gate; model.sidecar_point_gate_init.mode must remain "
            "'one' (the legacy compatibility default)"
        )
    return point, action, seed


def resolve_point_stages(architecture, names, count):
    if not architecture or "enc_channels" not in architecture or "enc_depths" not in architecture:
        raise ValueError("multilevel_local requires the exact Concerto encoder architecture")
    channels, depths = architecture["enc_channels"], architecture["enc_depths"]
    if len(channels) != len(depths):
        raise ValueError("Concerto encoder channel/depth counts disagree")
    available = {f"enc{i}": int(dim) for i, dim in enumerate(channels) if depths[i] > 0}
    names = list(names) if names is not None else list(available)[-count:]
    if len(names) != count or len(set(names)) != count:
        raise ValueError(
            "Point stage mapping must contain one distinct Concerto stage per Sidecar stage"
        )
    if any(name not in available for name in names):
        raise ValueError(f"Unknown point stages {names}; available encoder stages: {available}")
    return names, [available[name] for name in names]


def validate_variant(config):
    variant = getattr(config, "sidecar_variant", "legacy")
    if variant not in {"legacy", "multilevel_local"}:
        raise ValueError(f"Unknown sidecar_variant: {variant}")
    validate_gate_initialization(config)
    if variant == "legacy":
        return
    if config.dit_type != "sidecar_conditioned_dit" or not config.use_point_conditioning:
        raise ValueError("multilevel_local requires Sidecar DiT and point conditioning")
    if config.point_encoder_cfg.replace("-", "_").lower() not in {
        "concerto_small",
        "concerto_base",
    }:
        raise ValueError("multilevel_local requires a Concerto encoder")
    if (
        config.temporal_window_size
        or config.contact_input_mode != "none"
        or config.tactile_fusion_mode != "none"
    ):
        raise ValueError(
            "multilevel_local currently requires temporal=0, contact_input_mode=none, tactile_fusion_mode=none"
        )
    if config.contact_prediction_loss_weight or config.contact_predictor_warmup:
        raise ValueError("multilevel_local does not support Contact prediction losses or warmup")
    if config.sidecar_local_patch_size < 1 or config.sidecar_window_chunk_size < 1:
        raise ValueError("Local patch size and window chunk size must be positive")
    if (
        min(config.sidecar_hidden_dim, config.sidecar_num_heads, config.sidecar_num_stages) < 1
        or config.sidecar_hidden_dim % config.sidecar_num_heads
    ):
        raise ValueError("Sidecar hidden dimension must be divisible by attention heads")
    names = config.sidecar_point_stage_names
    count = (
        len(config.sidecar_stage_indices)
        if config.sidecar_stage_indices is not None
        else min(config.sidecar_num_stages, config.diffusion_model_cfg["num_layers"])
    )
    if names is not None and (len(names) != count or len(set(names)) != count):
        raise ValueError("Point mapping must have one distinct name per actual DiT injection stage")
    architecture = getattr(config, "point_encoder_model_config", None)
    if architecture:
        resolve_point_stages(architecture, names, count)


def validate_checkpoint_variant(checkpoint, requested):
    """Original NVIDIA -> new adapter is allowed; existing WAH variants cannot mix."""
    if getattr(checkpoint, "dit_type", None) != "sidecar_conditioned_dit":
        return
    old = getattr(checkpoint, "sidecar_variant", "legacy")
    new = getattr(requested, "sidecar_variant", "legacy")
    if old != new:
        raise ValueError(
            f"Incompatible Sidecar checkpoint variant: {old} -> {new}; start from NVIDIA GR00T for a new architecture"
        )
    if old == "multilevel_local":
        for key in (
            "sidecar_point_stage_names",
            "sidecar_local_patch_size",
            "sidecar_window_chunk_size",
        ):
            expected, actual = getattr(checkpoint, key), getattr(requested, key)
            if key == "sidecar_point_stage_names" and actual is None:
                actual, _ = resolve_point_stages(
                    checkpoint.point_encoder_model_config, None, len(expected)
                )
            if expected != actual:
                raise ValueError(f"Incompatible multilevel_local checkpoint configuration: {key}")
