"""Header-only VLM coverage and exact source-BF16 preparation before materialization."""

import json
import logging
from pathlib import Path

from safetensors import safe_open
from transformers.utils.hub import cached_file


def checkpoint_backbone_shapes(checkpoint, loading_kwargs, *, bf16_parameter_shapes=None):
    """Read actual Safetensors headers, never trusting an index alone for coverage.

    An optional BF16 shape collector reuses this header pass for action-head
    Parameters; no source dtype is guessed and no tensor data is read.
    Unsupported formats use the legacy loader. Corrupt or inaccessible files still
    raise, rather than disguising a broken checkpoint as a successful load.
    """
    kwargs = dict(loading_kwargs)
    kwargs.pop("trust_remote_code", None)
    index = cached_file(
        checkpoint,
        "model.safetensors.index.json",
        _raise_exceptions_for_missing_entries=False,
        **kwargs,
    )
    if index:
        with open(index) as stream:
            weight_map = json.load(stream)["weight_map"]
        filenames = sorted(
            {
                v
                for k, v in weight_map.items()
                if bf16_parameter_shapes is not None or k.startswith("backbone.model.")
            }
        )
    else:
        filenames = ["model.safetensors"]
    shapes = {}
    for filename in filenames:
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError(f"Unsafe checkpoint shard path: {filename}")
        path = cached_file(
            checkpoint,
            filename,
            _raise_exceptions_for_missing_entries=False,
            **kwargs,
        )
        if path is None:
            logging.warning("No Safetensors coverage available; using legacy VLM loading")
            return None
        with safe_open(path, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if bf16_parameter_shapes is not None and key.startswith("action_head."):
                    tensor = reader.get_slice(key)
                    if tensor.get_dtype() == "BF16":
                        bf16_parameter_shapes[key] = tuple(tensor.get_shape())
                if key.startswith("backbone.model."):
                    shapes[key.removeprefix("backbone.model.")] = tuple(
                        reader.get_slice(key).get_shape()
                    )
    return shapes


def backbone_is_covered(model, shapes):
    """Require every persistent tensor and its exact shape, including tied aliases."""
    if not shapes:
        return False
    return all(
        name in shapes and tuple(getattr(tensor, "ds_shape", tensor.shape)) == shapes[name]
        for name, tensor in model.state_dict(keep_vars=True).items()
    )


def finalize_pretrained_model(model):
    """Restore runtime-only VLM state after HF has replaced meta parameters."""
    backbone = model.backbone
    if hasattr(backbone, "_reset_rotary_inv_freq"):
        backbone._reset_rotary_inv_freq()
        backbone._apply_vision_patch_embed_channels_last()
    meta = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.is_meta
    ]
    if meta:
        raise RuntimeError(f"Checkpoint loading left required tensors on meta: {meta}")


def initialize_missing_sidecar(model, missing_keys):
    """Restore constructor initialization only for validated new Sidecar tensors.

    HF's generic missing-key initializer does not know about standalone gates
    and query parameters. Build only the small adapters outside its meta/no-init
    context; never reconstruct the pretrained DiT or either sensor encoder.
    """
    from contextlib import nullcontext

    import torch
    from transformers.integrations import is_deepspeed_zero3_enabled

    from gr00t.model.extension.contact_input_moe import ContactInputAdapter
    from gr00t.model.extension.contact_surprise import ContactInjection, ContactSurprise
    from gr00t.model.extension.point_action_sidecar import PointActionStage, TokenPool
    from gr00t.model.extension.temporal_fusion import TemporalFusion
    from gr00t.model.gr00t_n1d7.sensor_loading import SIDECAR_ADAPTER_PREFIXES

    keys = {
        key
        for key in missing_keys
        if key.startswith(SIDECAR_ADAPTER_PREFIXES)
        or key.startswith("action_head.temporal_fusion.")
        or (key.startswith("action_head.model.") and ".lora_" in key)
    }
    if not keys:
        return
    cfg = model.config
    dit = model.action_head.model
    factories = {
        "sidecar_pool": lambda: TokenPool(
            dit.inner_dim,
            cfg.sidecar_hidden_dim,
            cfg.sidecar_num_point_tokens,
            cfg.sidecar_num_heads,
        ),
        "sidecar_stages": lambda: torch.nn.ModuleList(
            PointActionStage(
                dit.inner_dim,
                cfg.sidecar_hidden_dim,
                cfg.sidecar_num_heads,
                point_gate_init=getattr(cfg, "sidecar_point_gate_init", None),
                action_gate_init=getattr(cfg, "sidecar_action_gate_init", None),
                gate_init_seed=getattr(cfg, "sidecar_gate_init_seed", 42),
                stage_index=stage,
                gate_mode=getattr(cfg, "sidecar_gate_mode", "scalar"),
                adaptive_gate_hidden_dim=getattr(cfg, "sidecar_adaptive_gate_hidden_dim", 128),
                adaptive_gate_init_action=getattr(cfg, "sidecar_adaptive_gate_init_action", 0.01),
                adaptive_gate_init_point=getattr(cfg, "sidecar_adaptive_gate_init_point", 0.8),
            )
            for stage, _ in enumerate(dit.sidecar_stages)
        ),
        "contact_adapter": lambda: ContactSurprise(
            dit.inner_dim,
            cfg.contact_hidden_dim,
            cfg.contact_num_tokens,
            cfg.sidecar_num_heads,
            cfg.freeze_contact_teacher,
        ),
        "contact_injections": lambda: torch.nn.ModuleList(
            ContactInjection(dit.inner_dim, cfg.contact_hidden_dim, cfg.sidecar_num_heads)
            for _ in dit.contact_injections
        ),
        "input_contact_adapter": lambda: ContactInputAdapter(
            dit.inner_dim,
            cfg.max_state_dim,
            mode=cfg.contact_input_mode,
            contact_dim=cfg.contact_hidden_dim,
            num_tokens=cfg.contact_num_tokens,
            use_surprise=False,
            diagnostics=True,
        ),
        "temporal_fusion": lambda: TemporalFusion(
            dit.inner_dim, cfg.max_state_dim, cfg.temporal_window_size
        ),
    }
    if getattr(cfg, "sidecar_variant", "legacy") == "multilevel_local":
        from gr00t.model.extension.multilevel_local_sidecar import (
            LocalJointActionStage,
            resolve_point_stages,
        )

        _, dimensions = resolve_point_stages(
            cfg.point_encoder_model_config, cfg.sidecar_point_stage_names, len(dit.sidecar_stages)
        )
        factories["sidecar_stages"] = lambda: torch.nn.ModuleList(
            LocalJointActionStage(
                dit.inner_dim,
                dim,
                cfg.sidecar_hidden_dim,
                cfg.sidecar_num_heads,
                cfg.sidecar_local_patch_size,
                cfg.sidecar_window_chunk_size,
                action_gate_init=getattr(cfg, "sidecar_action_gate_init", None),
                gate_init_seed=getattr(cfg, "sidecar_gate_init_seed", 42),
                stage_index=stage,
                gate_mode=getattr(cfg, "sidecar_gate_mode", "scalar"),
                adaptive_gate_hidden_dim=getattr(cfg, "sidecar_adaptive_gate_hidden_dim", 128),
                adaptive_gate_init_action=getattr(cfg, "sidecar_adaptive_gate_init_action", 0.01),
                temporal_context_dim=(
                    dit.inner_dim if getattr(cfg, "temporal_window_size", 0) else None
                ),
            )
            for stage, dim in enumerate(dimensions)
        )

    for name, factory in factories.items():
        prefix = (
            ("action_head." if name == "temporal_fusion" else "action_head.model.") + name + "."
        )
        selected = {key.removeprefix(prefix) for key in keys if key.startswith(prefix)}
        if not selected:
            continue
        fresh = factory().state_dict()
        target = getattr(model.action_head if name == "temporal_fusion" else dit, name).state_dict(
            keep_vars=True
        )
        for key in sorted(selected):
            parameter = target[key]
            context = nullcontext()
            if is_deepspeed_zero3_enabled():
                import deepspeed

                context = deepspeed.zero.GatheredParameters([parameter], modifier_rank=0)
            with context, torch.no_grad():
                parameter.copy_(fresh[key].to(device=parameter.device, dtype=parameter.dtype))

    import math

    for name, parameter in model.named_parameters():
        if name not in missing_keys or not name.startswith("action_head.model."):
            continue
        if ".lora_A.weight" not in name and ".lora_B.weight" not in name:
            continue
        context = nullcontext()
        if is_deepspeed_zero3_enabled():
            import deepspeed

            context = deepspeed.zero.GatheredParameters([parameter], modifier_rank=0)
        with context, torch.no_grad():
            if ".lora_A.weight" in name:
                torch.nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
            else:
                torch.nn.init.zeros_(parameter)


def prepare_bf16_checkpoint_parameters(model, shapes):
    """Preserve BF16 checkpoint storage without changing construction or its RNG.

    Only matched action-head meta Parameters are converted, before HF assigns
    checkpoint tensors and before ZeRO-3 initialization. Missing/new Sidecar
    tensors, real tensors, buffers, the VLM's explicit FP32 settings, and source
    FP32 tensors are untouched. Values are never rounded: selected source
    tensors are already BF16. No actual CPU tensor storage is allocated here.
    """
    import torch
    from transformers.integrations import is_deepspeed_zero3_enabled

    if not shapes or is_deepspeed_zero3_enabled():
        return 0
    count = 0
    for module_name, module in model.named_modules():
        if module_name != "action_head" and not module_name.startswith("action_head."):
            continue
        matched = {
            id(parameter)
            for name, parameter in module.named_parameters(recurse=False)
            if parameter.is_meta
            and parameter.is_floating_point()
            and shapes.get(f"{module_name}.{name}") == tuple(parameter.shape)
        }
        if not matched:
            continue
        # Module._apply is the same conversion mechanism used by .to(). The
        # converter is restricted to Parameters; buffers retain original dtype.
        module._apply(
            lambda tensor: tensor.to(dtype=torch.bfloat16) if id(tensor) in matched else tensor,
            recurse=False,
        )
        count += len(matched)
    logging.info("Prepared %d checkpoint-backed meta Parameters for BF16 storage", count)
    return count
