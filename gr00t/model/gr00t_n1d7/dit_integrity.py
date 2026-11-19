"""Numerical comparison helpers used by the standalone pretrained DiT audit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from gr00t.model.gr00t_n1d7.sensor_loading import (
    MULTIMODAL_ADAPTER_PREFIXES,
    SIDECAR_ADAPTER_PREFIXES,
)


def is_new_multimodal_adapter_key(name: str) -> bool:
    if name.startswith("action_head.temporal_fusion.") or (
        name.startswith("action_head.model.") and ".lora_" in name
    ):
        return True
    return any(
        name.startswith(prefix) if prefix.endswith(".") else name == prefix
        for prefix in MULTIMODAL_ADAPTER_PREFIXES + SIDECAR_ADAPTER_PREFIXES
    )


def compare_shared_tensors(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
    expected_keys=None,
) -> dict[str, Any]:
    """Compare all shared original DiT values after documented dtype casting.

    Adapter keys are excluded from the expected pretrained set. Encoder keys are
    excluded too: they have their own checkpoint/bootstrap lifecycle.
    """
    # Candidate architecture supplies the expected set too, so missing reference
    # tensors cannot disappear from the denominator.
    keys = set(reference) | set(candidate) if expected_keys is None else set(expected_keys)
    original_ref = sorted(
        k
        for k in keys
        if k.startswith("action_head.model.") and not is_new_multimodal_adapter_key(k)
    )
    matched, missing, mismatched = 0, [], []
    max_abs = 0.0
    max_rel = 0.0
    worst: list[dict[str, Any]] = []
    tensors = []
    for name in original_ref:
        ref = reference.get(name)
        got = candidate.get(name)
        if ref is None or got is None:
            missing.append(name)
            continue
        if tuple(ref.shape) != tuple(got.shape):
            mismatched.append(
                {
                    "name": name,
                    "reason": "shape",
                    "reference": list(ref.shape),
                    "candidate": list(got.shape),
                }
            )
            continue
        ref_dtype, got_dtype = str(ref.dtype), str(got.dtype)
        shape = list(ref.shape)
        denominator_floor = torch.finfo(got.dtype).eps if got.is_floating_point() else 1.0
        ref = ref.detach().to(device="cpu", dtype=got.dtype).double()
        got = got.detach().to(device="cpu").double()
        if not torch.isfinite(ref).all() or not torch.isfinite(got).all():
            row = {
                "name": name,
                "shape": shape,
                "reference_dtype": ref_dtype,
                "candidate_dtype": got_dtype,
                "finite": False,
                "allclose": False,
                "reason": "non_finite",
                "max_abs_error": None,
                "max_relative_error": None,
            }
            mismatched.append(row)
            tensors.append(row)
            continue
        diff = (got - ref).abs()
        abs_error = float(diff.max()) if diff.numel() else 0.0
        rel_error = (
            float((diff / ref.abs().clamp_min(denominator_floor)).max()) if diff.numel() else 0.0
        )
        max_abs, max_rel = max(max_abs, abs_error), max(max_rel, rel_error)
        okay = (
            torch.equal(got, ref)
            if atol == rtol == 0
            else torch.allclose(got, ref, atol=atol, rtol=rtol)
        )
        tensors.append(
            {
                "name": name,
                "shape": shape,
                "reference_dtype": ref_dtype,
                "candidate_dtype": got_dtype,
                "finite": True,
                "allclose": okay,
                "torch_equal_after_dtype_conversion": torch.equal(got, ref),
                "max_abs_error": abs_error,
                "max_relative_error": rel_error,
            }
        )
        matched += int(okay)
        if not okay:
            mismatched.append(
                {
                    "name": name,
                    "reason": "value",
                    "shape": list(got.shape),
                    "reference_dtype": str(reference[name].dtype),
                    "candidate_dtype": str(candidate[name].dtype),
                    "max_abs_error": abs_error,
                    "max_rel_error": rel_error,
                }
            )
        worst.append({"name": name, "max_abs_error": abs_error, "max_rel_error": rel_error})
    worst.sort(key=lambda row: row["max_abs_error"], reverse=True)
    unexpected_original = sorted(
        k
        for k in candidate
        if k.startswith("action_head.model.")
        and not is_new_multimodal_adapter_key(k)
        and k not in original_ref
    )
    expected = len(original_ref)
    return {
        "status": "PASS"
        if expected and not missing and not mismatched and not unexpected_original
        else "FAIL",
        "expected_tensors": expected,
        "matched_tensors": matched,
        "coverage": matched / expected if expected else 0.0,
        "missing": missing,
        "mismatched": mismatched,
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "worst_10": worst[:10],
        "tensors": tensors,
        "comparison": "Reference cast to candidate dtype; exact equality by default (no arithmetic in weight loading).",
        "tolerances": {
            "atol": atol,
            "rtol": rtol,
            "relative_denominator": "max(abs(cast reference), candidate dtype epsilon)",
        },
        "unexpected_original_keys": unexpected_original,
        "blocks": {
            str(index): {
                "expected_tensors": sum(
                    k.startswith(f"action_head.model.transformer_blocks.{index}.")
                    for k in original_ref
                ),
                "status": "FAIL"
                if any(
                    k.startswith(f"action_head.model.transformer_blocks.{index}.") for k in missing
                )
                or any(
                    row["name"].startswith(f"action_head.model.transformer_blocks.{index}.")
                    for row in mismatched
                )
                else "PASS",
            }
            for index in sorted(
                {
                    int(k.split("transformer_blocks.")[1].split(".")[0])
                    for k in original_ref
                    if "transformer_blocks." in k
                }
            )
        },
        "adapter_keys_excluded": sum(is_new_multimodal_adapter_key(k) for k in candidate),
    }


def overall_status(checks: list[str]) -> str:
    """FAIL dominates; SKIP makes an otherwise clean report INCOMPLETE."""
    if "FAIL" in checks:
        return "FAIL"
    if "SKIP" in checks or "INCOMPLETE" in checks:
        return "INCOMPLETE"
    return "PASS"


def validate_loading_info(info, allowed_prefixes=()):
    """Shared by training setup and standalone audit; never forgive base keys."""
    from gr00t.model.gr00t_n1d7.sensor_loading import unexpected_missing_keys

    errors = {}
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        values = info.get(field, [])
        if field == "missing_keys":
            values = unexpected_missing_keys(values, allowed_prefixes)
        if values:
            errors[field] = values
    return {"status": "FAIL" if errors else "PASS", "errors": errors, "loading_info": info}


def forward_parity(reference, candidate, config, device="cpu", dtype=torch.float32):
    """Sequential real-DiT forward passes; never instantiate the VLM."""
    candidate.eval().to(device="cpu", dtype=dtype)
    reference.eval().to(device=device, dtype=dtype)
    generator = torch.Generator().manual_seed(1729)
    width = reference.inner_dim
    cross_width = config["cross_attention_dim"]
    inputs = dict(
        hidden_states=torch.randn(1, 5, width, generator=generator).to(device, dtype),
        encoder_hidden_states=torch.randn(1, 6, cross_width, generator=generator).to(device, dtype),
        image_mask=torch.tensor([[False, False, False, True, True, True]], device=device),
        backbone_attention_mask=torch.ones(1, 6, dtype=torch.bool, device=device),
        hidden_attention_mask=torch.ones(1, 5, dtype=torch.bool, device=device),
        return_all_hidden_states=True,
    )
    sensors = dict(
        point_tokens=torch.randn(1, 7, width, generator=generator).to(device, dtype),
        tactile_tokens=torch.randn(1, 3, width, generator=generator).to(device, dtype),
        point_attention_mask=torch.ones(1, 7, dtype=torch.bool, device=device),
        tactile_attention_mask=torch.ones(1, 3, dtype=torch.bool, device=device),
    )
    results = []
    atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (2e-2, 2e-2)
    with torch.inference_mode():
        if hasattr(candidate, "sidecar_stages"):
            sensors["robot_state"] = torch.randn(1, 1, width, generator=generator).to(device, dtype)
            for stage in candidate.sidecar_stages:
                stage.set_injection_enabled("action", False)
            for injection in candidate.contact_injections:
                injection.scale.zero_()
        else:
            for gate in (candidate.point_gates, candidate.tactile_gates):
                gate.zero_()
        # Run reference first, then offload it before moving candidate to GPU.
        outputs = []
        for t in (0, 127, 999):
            output, layers = reference(**inputs, timestep=torch.tensor([t], device=device))
            outputs.append((output.cpu(), [v.cpu() for v in layers]))
        reference.cpu()
        candidate.to(device)
        for t, (ref, ref_layers) in zip((0, 127, 999), outputs):
            output, layers = candidate(
                **inputs, **sensors, timestep=torch.tensor([t], device=device)
            )

            def compare(a, b):
                a, b = a.float().cpu(), b.float().cpu()
                diff = (a - b).abs()
                return {
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                    "relative_error": float(diff.norm() / a.norm().clamp_min(1e-12)),
                    "allclose": torch.allclose(a, b, atol=atol, rtol=rtol),
                }

            results.append(
                {
                    "timestep": t,
                    **compare(ref, output),
                    "layers": [compare(a, b) for a, b in zip(ref_layers, layers)],
                    "layer_count_match": len(ref_layers) == len(layers),
                }
            )
    return {
        "status": "PASS"
        if all(
            r["allclose"]
            and r["layer_count_match"]
            and all(layer["allclose"] for layer in r["layers"])
            for r in results
        )
        else "FAIL",
        "atol": atol,
        "rtol": rtol,
        "cases": results,
    }
