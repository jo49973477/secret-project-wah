"""Training-free WAH interventions, paired inference and geometry diagnostics.

Only action gates are temporarily changed. Sensor removal is deliberately not an
intervention: ContactSurprise may also consume the original scene points.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
from itertools import combinations
import math

import numpy as np
import torch

from gr00t.experiment.sidecar_evaluation import initial_noise_probe, replay_rng


TASK_LABELS = {
    "lift_can": {"lift_can", "lift the can"},
    "insert_HDMI": {"insert_hdmi", "insert the hdmi connector"},
}


def canonical_task(value):
    value = str(value).strip().lower()
    for task, labels in TASK_LABELS.items():
        if value in labels:
            return task
    return None


def stage_mapping(dit):
    """Validate actual runtime stage/block mapping, never assume evenly spaced blocks."""
    stages = getattr(dit, "sidecar_stages", ())
    indices = list(getattr(dit, "stage_indices", ()))
    variant = getattr(dit, "sidecar_variant", "legacy")
    if variant not in {"legacy", "multilevel_local"}:
        raise ValueError(f"Unsupported Sidecar variant: {variant}")
    if not stages or len(stages) != len(indices):
        raise ValueError("A Sidecar checkpoint with a complete stage_indices mapping is required")
    if indices != sorted(set(indices)) or any(
        type(i) is not int or not 0 <= i < len(dit.transformer_blocks) for i in indices
    ):
        raise ValueError("Invalid runtime DiT block indices")
    if any(hasattr(p, "ds_id") for p in dit.parameters()):
        raise ValueError(
            "Live ZeRO-3 partitions are unsupported. Export a complete inference checkpoint "
            "with scripts/export_zero3_checkpoint.py first; evaluation never consolidates shards."
        )
    names = list(getattr(dit, "sidecar_point_stage_names", ()))
    if variant == "multilevel_local" and len(names) != len(stages):
        raise ValueError("Missing multilevel point-stage mapping")
    result = []
    for index, (block, stage) in enumerate(zip(indices, stages)):
        row = dict(stage=index, block_index=block, block_number=block + 1, variant=variant)
        for name in ("point_gate", "action_gate"):
            gate = getattr(stage, name, None)
            if gate is None and name == "action_gate":
                raise ValueError(f"Stage {index} has no action_gate")
            if gate is not None and (gate.numel() != 1 or not torch.isfinite(gate).all()):
                raise ValueError(f"Stage {index} has an invalid scalar {name}")
            row[name] = None if gate is None else float(gate.detach().float().cpu())
        row["point_gate_status"] = "unavailable" if row["point_gate"] is None else "available"
        row["point_gate_reason"] = (
            "This variant has no point_gate" if row["point_gate"] is None else None
        )
        row["point_stage"] = names[index] if variant == "multilevel_local" else None
        result.append(row)
    return result


def observed_residual(features, delta, mask=None):
    """FP32 Frobenius norms of A and the actual gated delta A, over valid tokens."""
    a, d = features.detach().float(), delta.detach().float()
    if a.shape != d.shape or a.ndim != 3:
        raise ValueError("Observer requires matching [batch, tokens, width] tensors")
    if mask is not None:
        if tuple(mask.shape) != tuple(a.shape[:2]):
            raise ValueError("Residual token mask shape mismatch")
        valid = mask.bool()[..., None].expand_as(a)
        a, d = a[valid], d[valid]
    if not torch.isfinite(a).all() or not torch.isfinite(d).all():
        raise ValueError("Nonfinite Sidecar activation/residual")
    an, dn = float(a.norm()), float(d.norm())
    return dict(
        feature_norm=an,
        residual_norm=dn,
        residual_ratio=dn / an if an > 0 else None,
        cosine_similarity=float((a * d).sum()) / (an * dn) if an > 0 and dn > 0 else None,
        residual_ratio_reason=None if an > 0 else "A has zero norm or no valid tokens",
        cosine_similarity_reason=None
        if an > 0 and dn > 0
        else "A or injected delta A has zero norm",
    )


@contextmanager
def sidecar_intervention(dit, *, zero_stages=(), swapped_points=None, rows=None):
    """Keep all contact/MoE paths intact, including their original point inputs.

    swapped_points is the donor (encoded_tokens, mask). Legacy donors replace
    only the pool input; multilevel donors replace only each local stage level.
    The DiT itself continues receiving the original point/tactile/state inputs.
    """
    mapping = stage_mapping(dit)
    zero_stages = tuple(zero_stages)
    if len(set(zero_stages)) != len(zero_stages) or any(
        type(i) is not int or not 0 <= i < len(mapping) for i in zero_stages
    ):
        raise ValueError("Invalid or duplicate stage ablation index")
    saved, handles, masks, calls = [], [], {}, {}
    had_observer = hasattr(dit, "_diagnostic_observer")
    previous = getattr(dit, "_diagnostic_observer", None)
    observer_installed = False

    def capture(index, module, args, kwargs):
        masks[index] = (
            args[2] if len(args) > 2 else kwargs.get("point_mask"),
            args[3] if len(args) > 3 else kwargs.get("action_mask"),
        )

    def observe(kind, index, features, delta):
        if previous is not None:
            previous(kind, index, features, delta)
        if rows is None or kind not in {"sidecar_action", "sidecar_point"}:
            return
        if index not in masks:
            raise RuntimeError("Observer invoked without the corresponding stage forward")
        pm, am = masks[index]
        key = index, kind
        call = calls.get(key, 0)
        calls[key] = call + 1
        row = dict(mapping[index], kind=kind, forward_call=call)
        row["action_gate"] = float(dit.sidecar_stages[index].action_gate.detach().float().cpu())
        row.update(observed_residual(features, delta, am if kind == "sidecar_action" else pm))
        rows.append(row)

    def swap_pool(module, args, kwargs):
        tokens, mask = swapped_points
        original = args[0] if args else kwargs["tokens"]
        if tokens is None or not isinstance(tokens, torch.Tensor):
            raise ValueError("Legacy swap requires encoded point tokens")
        return (tokens.to(device=original.device, dtype=original.dtype), mask), {}

    def swap_level(index, module, args, kwargs):
        levels, _ = swapped_points
        name = mapping[index]["point_stage"]
        if not isinstance(levels, dict) or name not in levels:
            raise ValueError(f"Missing donor multilevel stage {name}")
        if len(args) >= 2:
            return (args[0], levels[name], *args[2:]), kwargs
        return args, dict(kwargs, level=levels[name])

    try:
        for index in zero_stages:
            gate = dit.sidecar_stages[index].action_gate
            saved.append((gate, gate.detach().clone()))
            with torch.no_grad():
                gate.zero_()
        if swapped_points is not None and mapping[0]["variant"] == "legacy":
            handles.append(dit.sidecar_pool.register_forward_pre_hook(swap_pool, with_kwargs=True))
        for index, stage in enumerate(dit.sidecar_stages):
            if swapped_points is not None and mapping[0]["variant"] == "multilevel_local":
                handles.append(
                    stage.register_forward_pre_hook(
                        lambda m, a, k, i=index: swap_level(i, m, a, k), with_kwargs=True
                    )
                )
            if rows is not None:
                handles.append(
                    stage.register_forward_pre_hook(
                        lambda m, a, k, i=index: capture(i, m, a, k), with_kwargs=True
                    )
                )
        if rows is not None:
            dit._diagnostic_observer = observe
            observer_installed = True
        yield
    finally:
        for handle in handles:
            handle.remove()
        if observer_installed:
            if had_observer:
                dit._diagnostic_observer = previous
            else:
                del dit._diagnostic_observer
        with torch.no_grad():
            for gate, value in saved:
                gate.copy_(value)


def masked_action_metrics(original, changed, mask):
    """Use the processor's full structural horizon/dimension mask, excluding padding."""
    a, b = np.asarray(original, dtype=np.float64), np.asarray(changed, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("Expected matching [batch, horizon, dimension] action arrays")
    raw_mask = np.asarray(mask)
    if not np.isfinite(raw_mask).all() or not np.isin(raw_mask, [0, 1]).all():
        raise ValueError("Action mask must contain finite 0/1 values")
    valid = np.broadcast_to(raw_mask.astype(bool), a.shape)
    if not valid.any() or not np.isfinite(a[valid]).all() or not np.isfinite(b[valid]).all():
        raise ValueError("No valid action elements or nonfinite valid actions")
    diff = np.where(valid, b - a, 0)

    def by_axis(axes):
        count = valid.sum(axis=axes)
        total = abs(diff).sum(axis=axes)
        return [float(t / n) if n else None for t, n in zip(total, count)]

    return dict(
        mae=float(abs(diff).sum() / valid.sum()),
        rmse=float(np.sqrt(np.square(diff).sum() / valid.sum())),
        valid_elements=int(valid.sum()),
        per_dimension_mae=by_axis((0, 1)),
        per_timestep_mae=by_axis((0, 2)),
    )


def prepare_observation(policy, observation, seed):
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import _prepare_policy_model_inputs

    observation = deepcopy(observation)
    observation.actions = {}  # Never pass demonstration actions/RTC into inference.
    with replay_rng(seed):
        processed = policy.processor(
            [{"type": MessageType.EPISODE_STEP.value, "content": observation}]
        )
        inputs = _prepare_policy_model_inputs(policy.collate_fn([processed]), policy.model)
    if "inputs" not in inputs or "action" in inputs["inputs"]:
        raise ValueError("Unsupported inference collator contract or leaked ground-truth action")
    if "action_mask" not in inputs["inputs"]:
        raise ValueError("Checkpoint processor must supply the structural action_mask")
    return inputs


def encode_swapped_points(policy, original_inputs, original, donor, seed):
    """Preprocess only a cloud replacement, verify every other collated field is identical."""
    if not original.pointclouds or not donor.pointclouds:
        raise ValueError("Swapping requires nonempty point clouds in both observations")
    if original.pointclouds.keys() != donor.pointclouds.keys():
        raise ValueError("Donor point-cloud fields differ")
    swapped = deepcopy(original)
    for key in original.pointclouds:
        a, b = original.pointclouds[key], donor.pointclouds[key]
        if a.shape != b.shape or not np.isfinite(b).all():
            raise ValueError("Donor point-cloud shape mismatch or nonfinite cloud")
    swapped.pointclouds = deepcopy(donor.pointclouds)
    inputs = prepare_observation(policy, swapped, seed)
    left, right = original_inputs["inputs"], inputs["inputs"]
    if left.keys() != right.keys() or "points" not in right:
        raise ValueError("Unsupported point preprocessing contract")
    for key in left.keys() - {"points"}:
        if not torch.equal(left[key], right[key]):
            raise ValueError(f"Point swap unexpectedly changed non-point field {key}")
    with replay_rng(seed), torch.inference_mode():
        _, action_input = policy.model.prepare_input(deepcopy(right))
        tokens, mask, _ = policy.model.action_head._encode_multimodal_inputs(action_input)
    if tokens is None:
        raise ValueError("Checkpoint produced no encoded point tokens")
    return tokens, mask


def paired_inference(policy, inputs, *, seed, zero_stages=(), swapped_points=None, reference=None):
    head = policy.model.action_head
    rows = []
    with (
        sidecar_intervention(
            head.model, zero_stages=zero_stages, swapped_points=swapped_points, rows=rows
        ),
        replay_rng(seed),
        torch.inference_mode(),
        initial_noise_probe(head.action_encoder) as noise,
    ):
        output = policy.model.get_action(**deepcopy(inputs))
    if not noise:
        raise RuntimeError("Could not observe initial action-encoder noise")
    if reference is not None and not torch.equal(reference, noise[0]):
        raise RuntimeError("Initial diffusion noise differs across paired conditions")
    observed = {r["stage"] for r in rows if r["kind"] == "sidecar_action"}
    if observed != set(range(len(head.model.sidecar_stages))):
        raise RuntimeError("Not every Sidecar stage emitted its actual injected residual")
    actions = output["action_pred"].detach().float().cpu().numpy().copy()
    return actions, noise[0], rows


def noise_identity(noise):
    # NumPy cannot represent BF16, the normal inference dtype. Hash raw bytes
    # without converting/rounding the actual captured diffusion noise.
    value = noise.contiguous()
    return hashlib.sha256(
        str((tuple(value.shape), value.dtype)).encode() + value.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def compatible_donor(source_id, metadata, candidates):
    """Scene identity is explicit; a different episode alone is insufficient."""
    source = metadata.get(str(source_id), {})
    fields = ("task", "embodiment", "coordinate_frame", "sensor", "scene")
    if any(not source.get(key) for key in fields):
        return None, "Missing verified episode task/embodiment/frame/sensor/scene provenance"
    for candidate in candidates:
        other = metadata.get(str(candidate), {})
        if (
            candidate != source_id
            and other.get("scene")
            and other["scene"] != source["scene"]
            and all(other.get(k) == source[k] for k in fields if k != "scene")
        ):
            return candidate, None
    return None, "No compatible donor with a different explicit scene ID"


def geometry_metrics(source, donor, source_step, donor_step, normalized_mae):
    """Object displacement versus action sensitivity; this is not a correctness score."""
    records = []
    for metadata, step in ((source, source_step), (donor, donor_step)):
        records.append(dict(metadata, **metadata.get("frames", {}).get(str(step), {})))
    source, donor = records
    fields = ("object_id", "object_xyz_frame", "object_xyz_units", "object_xyz_source")
    if any(not record.get(k) for record in records for k in fields):
        return dict(status="skipped", reason="Object XYZ provenance/identity/frame/units missing")
    if (
        source["object_id"] != donor["object_id"]
        or source["object_xyz_frame"] != donor["object_xyz_frame"]
        or source["object_xyz_frame"] != source.get("coordinate_frame")
        or donor["object_xyz_frame"] != donor.get("coordinate_frame")
        or any(r["object_xyz_units"] != "m" for r in records)
    ):
        return dict(
            status="skipped", reason="Object identity, coordinate frame or metre units disagree"
        )
    try:
        a, b = (np.asarray(r.get("object_xyz"), dtype=float) for r in records)
        if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
            raise ValueError("invalid XYZ")
    except (ValueError, TypeError):
        return dict(status="skipped", reason="Object XYZ must be finite three-vectors")
    distance = float(np.linalg.norm(b - a))
    return dict(
        status="available",
        object_displacement_xyz_m=(b - a).tolist(),
        object_displacement_m=distance,
        normalized_action_mae=normalized_mae,
        normalized_mae_per_m=normalized_mae / distance if distance > 0 else None,
        gain_reason=None if distance > 0 else "Object displacement is zero",
        interpretation="Sensitivity to a mismatched cloud; not evidence of correct grounding",
    )


def comparison_rows(results, mask, context):
    """All unordered condition pairs, with decoded named fields kept separate."""
    rows = []
    for left, right in combinations(results, 2):
        a, b = results[left], results[right]
        pair = f"{left}__vs__{right}"
        rows.append(
            dict(
                context,
                comparison=pair,
                condition_a=left,
                condition_b=right,
                space="normalized",
                status="available",
                **masked_action_metrics(a[0], b[0], mask),
            )
        )
        if a[1].keys() != b[1].keys():
            raise ValueError("Decoded action fields differ across conditions")
        offset = 0
        for key in a[1]:
            values = np.asarray(a[1][key])
            width, horizon = values.shape[-1], values.shape[-2]
            field_mask = mask[:, :horizon, offset : offset + width]
            if field_mask.shape != values.shape:
                raise ValueError("Decoded action layout does not match structural mask")
            rows.append(
                dict(
                    context,
                    comparison=pair,
                    condition_a=left,
                    condition_b=right,
                    space=f"decoded:{key}",
                    units="checkpoint native units",
                    status="available",
                    **masked_action_metrics(values, b[1][key], field_mask),
                )
            )
            offset += width
    return rows


def finite_stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    return dict(
        mean=float(np.mean(values)) if values else None,
        std=float(np.std(values, ddof=0)) if values else None,
        n=len(values),
    )
