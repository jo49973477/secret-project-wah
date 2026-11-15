"""Dependency-free Sidecar architecture contracts for config/preflight and loaders."""


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
