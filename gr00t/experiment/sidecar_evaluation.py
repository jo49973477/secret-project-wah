"""Training-free, temporary Sidecar interventions and portable result export."""

from contextlib import contextmanager
import csv
import json
from pathlib import Path
import random
import re

import numpy as np
import torch


def sensitivity(original, changed, mask=None):
    a, b = np.asarray(original, dtype=float), np.asarray(changed, dtype=float)
    if a.shape != b.shape or a.ndim != 3 or not np.isfinite([a, b]).all():
        raise ValueError("Expected matching finite [batch, timestep, dimension] actions")
    valid = np.ones(a.shape, bool) if mask is None else np.broadcast_to(mask, a.shape)
    error = np.where(valid, b - a, 0)

    def mean(values, axes):
        count = valid.sum(axis=axes)
        return np.divide(
            values.sum(axis=axes), count, out=np.zeros_like(count, dtype=float), where=count > 0
        )

    if not valid.any():
        raise ValueError("No valid action elements")
    return dict(
        mae=float(mean(abs(error), None)),
        rmse=float(np.sqrt(mean(error**2, None))),
        per_dimension_mae=mean(abs(error), (0, 1)).tolist(),
        per_timestep_mae=mean(abs(error), (0, 2)).tolist(),
        chunk_l2=np.linalg.norm(error.reshape(a.shape[0], -1), axis=1).tolist(),
    )


def residual_metrics(features, residual, mask=None):
    a, b = features.detach().float(), residual.detach().float()
    if mask is not None:
        a = a.masked_fill(~mask.bool()[..., None], 0)
        b = b.masked_fill(~mask.bool()[..., None], 0)
    an, bn = a.norm(), b.norm()
    return dict(
        feature_norm=float(an),
        residual_norm=float(bn),
        residual_ratio=float(bn / (an + 1e-12)),
        cosine_similarity=float((a * b).sum() / (an * bn).clamp_min(1e-12)),
    )


@contextmanager
def replay_rng(seed):
    """Replay all torch generators; leave caller RNG streams untouched."""
    devices = list(range(torch.cuda.device_count()))
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def initial_noise_probe(action_encoder):
    """Capture the first actual action-encoder input without modifying it."""
    values = []

    def capture(module, args):
        if not values:
            values.append(args[0].detach().cpu().clone())

    handle = action_encoder.register_forward_pre_hook(capture)
    try:
        yield values
    finally:
        handle.remove()


def gate_rows(dit):
    from gr00t.experiment.multimodal_diagnostics import full_value

    if not hasattr(dit, "sidecar_stages"):
        raise ValueError("Diagnostics require SidecarConditionedDiT")
    rows = []
    for i, (block, stage) in enumerate(zip(dit.stage_indices, dit.sidecar_stages)):
        rows.append(
            dict(
                stage=i,
                block_index=block,
                block_number=block + 1,
                gate_mode=getattr(stage, "gate_mode", "scalar"),
                point_gate=float(full_value(stage.point_gate))
                if hasattr(stage, "point_gate")
                else None,
                point_stage=getattr(dit, "sidecar_point_stage_names", [])[i]
                if getattr(dit, "sidecar_variant", "legacy") == "multilevel_local"
                else None,
                point_gate_init_mode=stage.point_gate_init["mode"]
                if hasattr(stage, "point_gate")
                else None,
                point_gate_initial_value=getattr(stage, "initial_point_gate", None),
                action_gate=float(full_value(stage.action_gate))
                if hasattr(stage, "action_gate")
                else None,
                action_gate_init_mode=stage.action_gate_init["mode"],
                action_gate_initial_value=stage.initial_action_gate,
            )
        )
    return rows


@contextmanager
def intervention(dit, *, disabled=False, zero_stages=(), rows=None):
    if not hasattr(dit, "sidecar_stages"):
        raise ValueError("Diagnostics require SidecarConditionedDiT")
    if disabled and dit.tactile_fusion_mode != "none":
        raise ValueError(
            "Point removal unsupported with point-dependent Contact conditioning; use a point-only checkpoint"
        )
    if any(hasattr(p, "ds_id") for p in dit.parameters()):
        raise ValueError(
            "Use a complete exported inference checkpoint for interventions, not live ZeRO-3 shards"
        )
    saved, handles, masks = [], [], {}
    previous = getattr(dit, "_diagnostic_observer", None)

    def capture(module, args):
        masks[id(module)] = (args[2], args[3] if len(args) > 3 else None)

    def observe(kind, stage, features, residual):
        if previous is not None:
            previous(kind, stage, features, residual)
        if rows is not None and kind in {"sidecar_point", "sidecar_action"}:
            pm, am = masks[id(dit.sidecar_stages[stage])]
            rows.append(
                dict(
                    stage=stage,
                    block_index=dit.stage_indices[stage],
                    block_number=dit.stage_indices[stage] + 1,
                    kind=kind,
                    **residual_metrics(features, residual, pm if kind == "sidecar_point" else am),
                )
            )

            if getattr(dit, "sidecar_variant", "legacy") == "multilevel_local":
                rows[-1].update(dit.sidecar_stages[stage].last_diagnostics)
                rows[-1]["point_stage"] = dit.sidecar_point_stage_names[stage]

    def remove_points(module, args, kwargs):
        kwargs = dict(kwargs, point_tokens=None, point_attention_mask=None)
        return args, kwargs

    try:
        for i in zero_stages:
            stage = dit.sidecar_stages[i]
            if getattr(stage, "gate_mode", "scalar") == "adaptive":
                previous_enabled = getattr(stage, "_action_injection_enabled", True)
                saved.append(("enabled", stage, previous_enabled))
                stage.set_injection_enabled("action", False)
            else:
                parameter = stage.action_gate
                saved.append(("parameter", parameter, parameter.detach().clone()))
                with torch.no_grad():
                    parameter.zero_()
        if disabled:
            handles.append(dit.register_forward_pre_hook(remove_points, with_kwargs=True))
        if rows is not None:
            for stage in dit.sidecar_stages:
                handles.append(stage.register_forward_pre_hook(capture))
            dit._diagnostic_observer = observe
        yield
    finally:
        for handle in handles:
            handle.remove()
        dit._diagnostic_observer = previous
        for kind, target, value in saved:
            if kind == "enabled":
                target.set_injection_enabled("action", value)
            else:
                with torch.no_grad():
                    target.copy_(value)


def offline_rows(path):
    rows = []
    pattern = re.compile(r"sidecar_(action|point)_(gate|residual_ratio)_stage_(\d+)$")
    with Path(path).open() as stream:
        for line, text in enumerate(stream, 1):
            record = json.loads(text)
            for mapping in record.get("sidecar_stage_mapping", []):
                rows.append(
                    dict(
                        source_line=line,
                        global_step=record.get("global_step"),
                        stage=mapping["stage"],
                        point_stage=mapping["point_stage"],
                        block_index=mapping["dit_block"],
                        block_number=mapping["dit_block"] + 1,
                        kind="sidecar_action",
                        action_gate=mapping.get("action_gate"),
                        residual_ratio=mapping.get("action_injection_ratio"),
                        **{
                            k: v
                            for k, v in mapping.items()
                            if k
                            not in {
                                "stage",
                                "point_stage",
                                "dit_block",
                                "action_gate",
                                "action_injection_ratio",
                            }
                        },
                    )
                )
            for key, value in record.items():
                match = pattern.fullmatch(key)
                if match and value is not None:
                    modality, metric, stage = match.groups()
                    rows.append(
                        dict(
                            source_line=line,
                            global_step=record.get("global_step"),
                            stage=int(stage),
                            kind=f"sidecar_{modality}",
                            metric=metric,
                            value=value,
                        )
                    )
    return rows


def export(output, summary, samples, stages):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sample_metrics", samples), ("stage_metrics", stages)):
        fields = sorted({key for row in rows for key in row}) or ["status"]
        with (output / f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(
                {k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()}
                for row in rows
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = []
    for metric in ("gate", "residual_ratio"):
        fig, ax = plt.subplots()
        found = False
        for kind in ("sidecar_action", "sidecar_point"):
            values = [
                (r["stage"], r["value"])
                for r in stages
                if r.get("kind") == kind and r.get("metric") == metric
            ]
            values += [
                (r["stage"], r[metric])
                for r in stages
                if r.get("kind") == kind and r.get(metric) is not None
            ]
            if metric == "gate":
                values += [
                    (r["stage"], r[kind.removeprefix("sidecar_") + "_gate"])
                    for r in stages
                    if r.get(kind.removeprefix("sidecar_") + "_gate") is not None
                ]
            if values:
                x = sorted(set(s for s, _ in values))
                ax.plot(x, [np.mean([v for s, v in values if s == i]) for i in x], "o-", label=kind)
                found = True
        if found:
            ax.set(
                xlabel="Sidecar stage (zero-based)",
                ylabel=f"{metric} (dimensionless)",
                title=f"Stage {metric}: mean over recorded calls",
            )
            ax.legend()
            fig.savefig(output / f"stage_{metric}.png")
            plots.append(metric)
        plt.close(fig)
    if samples:
        fig, ax = plt.subplots()
        for row in samples:
            if row.get("space") == "normalized":
                ax.plot(row["per_dimension_mae"], alpha=0.5, label=row["condition"])
        ax.set(
            xlabel="Action dimension (checkpoint padded layout)",
            ylabel="MAE (normalized units)",
            title="Fixed-observation action sensitivity",
        )
        fig.savefig(output / "action_dimension_sensitivity.png")
        plt.close(fig)
    summary["plots"] = plots
    summary["missing_plots"] = {
        "geometry": summary.get("geometry", {}).get("reason", "Verified geometry unavailable")
        if isinstance(summary.get("geometry"), dict)
        else "Verified geometry unavailable",
        "action_sensitivity": None if samples else "No action comparisons in offline logs",
        "stage_gate": None if "gate" in plots else "No stage gate measurements",
        "stage_residual_ratio": None
        if "residual_ratio" in plots
        else "No stage residual measurements",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
