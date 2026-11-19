"""Command-line orchestration for WAH experiments 7.1.1 and 7.1.2. No training imports."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from enum import Enum
import json
import os
from pathlib import Path
import signal


def parser():
    result = argparse.ArgumentParser(
        description="Training-free, fixed-noise WAH grounding/Sidecar diagnostics (7.1.1 / 7.1.2)."
    )
    result.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Local complete inference checkpoint, including ZeRO-3 exports",
    )
    result.add_argument("--dataset-path", type=Path, required=True, help="UniVTAC LeRobot dataset")
    result.add_argument(
        "--experiment", required=True, choices=["point_ablation", "gate_diagnostics"]
    )
    result.add_argument("--task", choices=["lift_can", "insert_HDMI"], default="lift_can")
    result.add_argument("--num-samples", type=int, default=100)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--output-dir", type=Path, required=True, help="New or empty directory")
    result.add_argument("--device", default="cuda:0", help="cpu or cuda[:index]")
    result.add_argument("--embodiment-tag", default="new_embodiment")
    result.add_argument(
        "--inference-steps",
        type=int,
        help="Override checkpoint flow steps; default preserves checkpoint setting",
    )
    result.add_argument(
        "--scene-metadata",
        type=Path,
        help="Verified episode scene/object provenance JSON; defaults to meta/wah_diagnostics.json",
    )
    result.add_argument(
        "--rollout-result",
        action="append",
        default=[],
        metavar="CONDITION=DIR",
        help=(
            "Attach an existing UniVTAC closed-loop result to a diagnostic condition; "
            "repeat for multiple conditions"
        ),
    )
    result.add_argument(
        "--wandb", action="store_true", help="Opt into a separate W&B inference run"
    )
    result.add_argument("--wandb-project", default="wah-univtac")
    result.add_argument("--wandb-group", help="Default: diagnostics-<experiment>")
    result.add_argument("--wandb-name", help="Default: diagnostics-<experiment>-<task>")
    result.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    return result


def validate_args(args):
    if args.num_samples < 1 or not 0 <= args.seed < 2**32:
        raise ValueError("--num-samples must be positive and --seed must be in [0, 2**32)")
    if args.inference_steps is not None and args.inference_steps < 1:
        raise ValueError("--inference-steps must be positive")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for source in (args.checkpoint, args.dataset_path):
        if not source.is_dir():
            raise ValueError(f"Input directory does not exist: {source}")
        if (
            source == args.output_dir
            or source in args.output_dir.parents
            or args.output_dir in source.parents
        ):
            raise ValueError("Output directory must not overlap the checkpoint or dataset tree")
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        raise ValueError("--output-dir must be a new or empty directory")
    if not (args.checkpoint / "config.json").is_file() or not any(
        p.is_file() and p.suffix in {".safetensors", ".bin"} for p in args.checkpoint.iterdir()
    ):
        raise ValueError(
            "No complete inference checkpoint. Export native ZeRO-3 shards with "
            "scripts/export_zero3_checkpoint.py first; this CLI never exports or trains."
        )
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json", "stats.json"):
        if not (args.dataset_path / "meta" / name).is_file():
            raise ValueError(f"Missing LeRobot metadata: meta/{name}")
    if args.scene_metadata is not None:
        args.scene_metadata = args.scene_metadata.expanduser().resolve()
        if not args.scene_metadata.is_file():
            raise ValueError(f"Scene metadata not found: {args.scene_metadata}")
    for specification in args.rollout_result:
        if "=" not in specification:
            raise ValueError("--rollout-result must use CONDITION=DIR")
        condition, directory = specification.split("=", 1)
        if not condition or not Path(directory).expanduser().is_dir():
            raise ValueError(f"Invalid --rollout-result: {specification}")


def load_closed_loop_outcomes(specifications, conditions, output_dir):
    """Read authoritative rollout artifacts without mixing them with frame metrics."""
    from examples.UniVTAC.univtac_rollout.reporting.analytics import episode_summary, task_summaries
    from examples.UniVTAC.univtac_rollout.reporting.parser import load_run

    results = []
    seen = set()
    for specification in specifications:
        condition, directory = specification.split("=", 1)
        if condition not in conditions:
            raise ValueError(
                f"Rollout condition {condition!r} is not one of the active conditions: {conditions}"
            )
        if condition in seen:
            raise ValueError(f"Duplicate closed-loop result for condition {condition!r}")
        seen.add(condition)
        run = load_run(directory, label=condition, output_dir=output_dir)
        results.append(
            {
                "condition": condition,
                "run_dir": str(run.root),
                "source": run.source,
                "overall": episode_summary(run.episodes),
                "tasks": task_summaries(run),
                "warnings": run.warnings,
                "interpretation": (
                    "Success, failure, and error are separate closed-loop outcomes; "
                    "errors are excluded from the success-rate denominator."
                ),
            }
        )
    return results


@contextmanager
def interrupted_by_sigterm():
    """SIGINT raises KeyboardInterrupt normally; handle scheduler SIGTERM as well."""
    previous = signal.getsignal(signal.SIGTERM)

    def stop(signum, frame):
        raise InterruptedError("Received SIGTERM")

    signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _json_config(value):
    if isinstance(value, Enum):
        return value.value
    raise TypeError(type(value).__name__)


def load_scene_metadata(args):
    path = args.scene_metadata or args.dataset_path / "meta" / "wah_diagnostics.json"
    if not path.exists():
        return {}, {}, None

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value in scene metadata: {value}")

    record = json.loads(path.read_text(), parse_constant=reject_constant)
    if not isinstance(record, dict) or not isinstance(record.get("episodes"), dict):
        raise ValueError("Scene metadata must contain an 'episodes' mapping keyed by episode ID")
    episodes = record["episodes"]
    for episode, meta in episodes.items():
        if not episode.isdecimal() or not isinstance(meta, dict):
            raise ValueError(
                "Scene metadata requires nonnegative episode ID keys and object values"
            )
        for key in ("task", "embodiment", "coordinate_frame", "sensor", "scene"):
            if key in meta and (not isinstance(meta[key], str) or not meta[key].strip()):
                raise ValueError(f"Episode {episode}: {key} must be a nonempty string")
        if "frames" in meta and (
            not isinstance(meta["frames"], dict)
            or any(
                not str(k).isdecimal() or not isinstance(v, dict) for k, v in meta["frames"].items()
            )
        ):
            raise ValueError("Geometry 'frames' must map frame indices to metadata objects")
    contract = record.get("cartesian_action", {})
    if not isinstance(contract, dict):
        raise ValueError("cartesian_action must be an object")
    return episodes, contract, str(path)


def select_samples(loader, configs, metadata, task, count, seed):
    import numpy as np

    from gr00t.experiment.wah_grounding import canonical_task

    offsets = [i for config in configs.values() for i in config.delta_indices]
    if not offsets:
        raise ValueError("Checkpoint has no observation/action temporal offsets")
    first, future = max(0, -min(offsets)), max(0, max(offsets))
    ranges, total, ids = [], 0, set()
    for index, episode in enumerate(loader.episodes_metadata):
        eid = episode["episode_index"]
        if type(eid) is not int or eid < 0 or eid in ids:
            raise ValueError("LeRobot episode IDs must be unique nonnegative integers")
        ids.add(eid)
        meta_task = canonical_task(metadata.get(str(eid), {}).get("task"))
        tasks = {canonical_task(t) for t in episode.get("tasks", [])}
        tasks.discard(None)
        if meta_task is not None and tasks and meta_task not in tasks:
            raise ValueError(f"Task provenance conflicts with LeRobot episode {eid}")
        if meta_task != task and (meta_task is not None or tasks != {task}):
            continue
        length = int(episode["length"]) - future - first
        if length > 0:
            ranges.append(dict(index=index, episode=eid, first=first, length=length, start=total))
            total += length
    if not total:
        raise ValueError(
            f"No valid {task} samples. Check task metadata and temporal/action horizons"
        )
    selection = sorted(
        np.random.default_rng(seed).choice(total, min(count, total), replace=False).tolist()
    )
    samples = []
    for position in selection:
        interval = next(r for r in reversed(ranges) if position >= r["start"])
        samples.append(
            dict(
                index=interval["index"],
                episode=interval["episode"],
                step=interval["first"] + position - interval["start"],
            )
        )
    return samples, ranges


def cartesian_response(results, mask, config, contract, source, donor):
    """Only explicit EEF XYZ formats are supported; joint-space requires verified FK."""
    import numpy as np

    if not contract:
        return dict(
            status="skipped",
            reason="No verified Cartesian action contract or joint-space FK adapter",
        )
    field = contract.get("field")
    if (
        contract.get("units") != "m"
        or not contract.get("source")
        or contract.get("coordinate_frame") != source.get("object_xyz_frame")
        or contract.get("coordinate_frame") != donor.get("object_xyz_frame")
    ):
        return dict(
            status="skipped", reason="Cartesian action provenance/frame/metre units not verified"
        )
    keys = config.modality_keys
    if field not in keys or not config.action_configs:
        return dict(
            status="skipped", reason="Cartesian field missing in checkpoint action configuration"
        )
    action_config = config.action_configs[keys.index(field)]
    if action_config.type.value != "eef" or action_config.format.value not in {
        "xyz+rot6d",
        "xyz+rotvec",
    }:
        return dict(
            status="skipped",
            reason="Joint/default-format actions have no verified FK; XYZ response unavailable",
        )
    decoded = results["original"][1]
    offset = sum(decoded[k].shape[-1] for k in keys[: keys.index(field)])
    a, b = decoded[field], results["swapped"][1][field]
    valid = mask[:, : a.shape[-2], offset : offset + 3].all(-1)
    if a.shape[-1] < 3 or not valid.any():
        return dict(status="skipped", reason="No valid decoded Cartesian XYZ timesteps")
    displacement = (b[..., :3] - a[..., :3])[valid]
    return dict(
        status="available",
        field=field,
        units="m",
        mean_action_displacement_xyz_m=displacement.mean(0).tolist(),
        mean_action_displacement_norm_m=float(np.linalg.norm(displacement, axis=-1).mean()),
        interpretation="Decoded absolute EEF response, not task success or target error",
    )


def evaluate(args, report, dashboard, tracker, policy_factory=None, loader_factory=None):
    import numpy as np
    import torch

    from gr00t.experiment.wah_grounding import (
        canonical_task,
        comparison_rows,
        compatible_donor,
        encode_swapped_points,
        geometry_metrics,
        noise_identity,
        paired_inference,
        prepare_observation,
        replay_rng,
        stage_mapping,
    )

    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("--device must be cpu or cuda[:index]")
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError(f"Requested CUDA device is unavailable: {args.device}")
    if policy_factory is None:
        if os.environ.get("GROOT_SKIP_HF_MODEL_WEIGHTS", "").lower() not in {
            "",
            "0",
            "false",
            "no",
            "off",
        }:
            raise ValueError("Unset GROOT_SKIP_HF_MODEL_WEIGHTS for real checkpoint evaluation")
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        policy_factory = Gr00tPolicy
    if loader_factory is None:
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

        loader_factory = LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data

    policy = policy_factory(args.embodiment_tag, str(args.checkpoint), device=args.device)
    policy.model.eval()
    policy.processor.eval()
    head = policy.model.action_head
    dit = head.model
    gates = stage_mapping(dit)
    if args.inference_steps is not None:
        head.num_inference_timesteps = args.inference_steps
    if head.num_inference_timesteps < 1:
        raise ValueError("Checkpoint inference step count must be positive")
    configs = deepcopy(policy.get_modality_config())
    if not {"video", "language", "state", "action", "pointcloud"}.issubset(configs):
        raise ValueError(
            "Checkpoint must configure RGB, language, state, action and pointcloud modalities"
        )
    loader = loader_factory(args.dataset_path, configs)
    metadata, cartesian_contract, metadata_path = load_scene_metadata(args)
    for eid, meta in metadata.items():
        if meta.get("embodiment") and meta["embodiment"] != policy.embodiment_tag.value:
            raise ValueError(f"Scene metadata embodiment differs from policy for episode {eid}")
        if meta.get("task"):
            task = canonical_task(meta["task"])
            if task is None:
                raise ValueError(f"Unknown scene metadata task for episode {eid}")
            meta["task"] = task
    selection, ranges = select_samples(
        loader, configs, metadata, args.task, args.num_samples, args.seed
    )
    if len(selection) < args.num_samples:
        report.issue(
            "sample_count",
            f"Only {len(selection)} unique valid frames available; requested {args.num_samples}",
        )
    conditions = (
        ["original", "disabled", "swapped"]
        if args.experiment == "point_ablation"
        else ["original", *[f"zero_stage_{i}" for i in range(len(gates))], "zero_all"]
    )
    report.summary["closed_loop_outcomes"] = load_closed_loop_outcomes(
        args.rollout_result, conditions, report.output
    )
    report.summary.update(gates=gates, scene_metadata=metadata_path, scene_provenance=metadata)
    report.summary["config"].update(
        checkpoint_identity=policy.checkpoint_identity,
        model_variant=getattr(dit, "sidecar_variant", "legacy"),
        inference_steps=head.num_inference_timesteps,
        conditions=conditions,
        seed_rule="(seed + sample_index) % 2**32; separate replay for preprocessing and inference",
        sensor_contract=json.loads(
            json.dumps({k: asdict(v) for k, v in configs.items()}, default=_json_config)
        ),
        tactile_fusion_mode=getattr(dit, "tactile_fusion_mode", None),
        contact_input_mode=getattr(policy.model.config, "contact_input_mode", None),
        temporal_window_size=getattr(policy.model.config, "temporal_window_size", None),
        sampling="Uniform unique valid frames, seeded selection; chronological processing",
        selected_samples=selection,
        sample_seeds=[(args.seed + i) % 2**32 for i in range(len(selection))],
        intervention="Zero action gates only; donor points routed exclusively to Sidecar",
        normalized_units="Checkpoint normalization, structural action_mask applied",
        decoded_units="processor.decode_action; absolute named fields in checkpoint native units",
    )
    report.summary["status"] = "running"
    tracker.start()
    report.save()
    observation_configs = {k: v for k, v in configs.items() if k != "action"}
    cache = OrderedDict()

    def trajectory(index):
        if index not in cache:
            eid = loader.episodes_metadata[index]["episode_index"]
            with replay_rng((args.seed + eid) % 2**32):
                cache[index] = loader[index]
            if len(cache) > 2:
                cache.popitem(last=False)
        cache.move_to_end(index)
        return cache[index]

    for number, sample in enumerate(selection):
        seed = (args.seed + number) % 2**32
        context = dict(
            sample=number, task=args.task, episode=sample["episode"], step=sample["step"], seed=seed
        )
        report.summary["current_sample"] = context
        point = extract_step_data(
            trajectory(sample["index"]),
            sample["step"],
            observation_configs,
            embodiment_tag=policy.embodiment_tag,
        )
        text_task = canonical_task(point.text)
        if text_task is not None and text_task != args.task:
            raise ValueError(f"Observation language conflicts with selected task: {context}")
        inputs = prepare_observation(policy, point, seed)
        mask = inputs["inputs"]["action_mask"].detach().cpu().numpy()
        if mask.ndim != 3 or not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError(
                "Expected finite structural [batch, horizon, dimension] 0/1 action mask"
            )
        donor_id = donor_step = None
        swap_reason = None
        if args.experiment == "point_ablation":
            donor_id, swap_reason = compatible_donor(
                sample["episode"], metadata, [r["episode"] for r in ranges]
            )
        results, reference = {}, None
        sample_rows = []
        for condition in conditions:
            report.summary["current_condition"] = condition
            dashboard.update(number, condition)
            swapped = None
            if condition == "swapped" and donor_id is not None:
                donor_range = next(r for r in ranges if r["episode"] == donor_id)
                donor_step = donor_range["first"]
                try:
                    donor = extract_step_data(
                        trajectory(donor_range["index"]),
                        donor_step,
                        observation_configs,
                        embodiment_tag=policy.embodiment_tag,
                    )
                    swapped = encode_swapped_points(policy, inputs, point, donor, seed)
                except (ValueError, KeyError, IndexError) as exc:
                    swap_reason = f"Donor cloud is incompatible: {exc}"
            if condition == "swapped" and swapped is None:
                for left in ("original", "disabled"):
                    sample_rows.append(
                        dict(
                            context,
                            comparison=f"{left}__vs__swapped",
                            condition_a=left,
                            condition_b="swapped",
                            space="normalized",
                            status="skipped",
                            reason=swap_reason,
                            mae=None,
                            rmse=None,
                        )
                    )
                report.issue("swapped_pointcloud", swap_reason, sample=number)
                continue
            zero = (
                tuple(range(len(gates)))
                if condition in {"disabled", "zero_all"}
                else (int(condition.removeprefix("zero_stage_")),)
                if condition.startswith("zero_stage_")
                else ()
            )
            actions, noise, stages = paired_inference(
                policy,
                inputs,
                seed=seed,
                zero_stages=zero,
                swapped_points=swapped if condition == "swapped" else None,
                reference=reference,
            )
            reference = noise if reference is None else reference
            digest = noise_identity(noise)
            state = {k: np.stack([v]) for k, v in point.states.items()}
            decoded = policy.processor.decode_action(actions.copy(), policy.embodiment_tag, state)
            # Enforce checkpoint action-key ordering for slicing the normalized mask.
            if set(decoded) != set(configs["action"].modality_keys):
                raise ValueError("Decoded fields disagree with checkpoint action layout")
            decoded = {key: np.asarray(decoded[key]) for key in configs["action"].modality_keys}
            for left in (
                results
                if args.experiment == "point_ablation"
                else (["original"] if results else [])
            ):
                sample_rows.extend(
                    comparison_rows(
                        {left: results[left], condition: (actions, decoded)},
                        mask,
                        dict(
                            context,
                            noise_sha256=digest,
                            initial_noise_verified=True,
                            donor_episode=donor_id,
                        ),
                    )
                )
            results[condition] = actions, decoded
            report.stages.extend(
                dict(context, condition=condition, noise_sha256=digest, **r) for r in stages
            )
            report.summary["conditions_completed"] += 1
            # Commit every completed condition, including partial samples, before moving on.
            report.samples.extend(sample_rows)
            sample_rows = []
            report.save()
        report.samples.extend(sample_rows)
        if args.experiment == "point_ablation":
            if "swapped" in results:
                pair = next(
                    r
                    for r in report.samples
                    if r["sample"] == number
                    and r["space"] == "normalized"
                    and r["comparison"] == "original__vs__swapped"
                )
                source, other = metadata[str(sample["episode"])], metadata[str(donor_id)]
                geometry = geometry_metrics(source, other, sample["step"], donor_step, pair["mae"])
                spatial = (
                    cartesian_response(
                        results, mask, configs["action"], cartesian_contract, source, other
                    )
                    if geometry["status"] == "available"
                    else dict(status="skipped", reason=geometry["reason"])
                )
                geometry["cartesian_response"] = spatial
                if spatial["status"] == "available":
                    object_delta = np.asarray(geometry["object_displacement_xyz_m"], dtype=float)
                    action_delta = np.asarray(
                        spatial["mean_action_displacement_xyz_m"], dtype=float
                    )
                    denominator = np.linalg.norm(object_delta) * np.linalg.norm(action_delta)
                    geometry["geometry_alignment_cosine"] = (
                        float(np.dot(object_delta, action_delta) / denominator)
                        if denominator > 0
                        else None
                    )
                    geometry["geometry_alignment_interpretation"] = (
                        "Directional EEF response to the swapped object displacement; "
                        "not a task-success or grounding-correctness score"
                    )
                if spatial["status"] != "available":
                    report.issue("cartesian_response", spatial["reason"], sample=number)
            else:
                geometry = dict(status="skipped", reason=swap_reason)
            report.geometry.append(
                dict(context, donor_episode=donor_id, donor_step=donor_step, **geometry)
            )
            if geometry["status"] != "available":
                report.issue("geometry_sensitivity", geometry["reason"], sample=number)
        report.summary["samples_completed"] = number + 1
        tracker.sample(number, [r for r in report.samples if r["sample"] == number])
        report.save()
        dashboard.update(number + 1, "sample complete")
    report.summary["status"] = "complete"
    report.summary["initial_noise_verified"] = True
    report.summary["paired_comparisons"] = len(
        [r for r in report.samples if r.get("mae") is not None and r["space"] == "normalized"]
    )
    for gate in gates:
        if gate["point_gate_status"] == "unavailable":
            report.issue("point_gate", gate["point_gate_reason"], stage=gate["stage"])


def main(argv=None, *, policy_factory=None, loader_factory=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        validate_args(args)
    except (ValueError, OSError) as exc:
        cli.error(str(exc))
    from gr00t.experiment.wah_reporting import (
        DiagnosticDashboard,
        DiagnosticReport,
        DiagnosticWandb,
    )

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = DiagnosticReport(args.output_dir, config)
    dashboard = DiagnosticDashboard(config)
    tracker = DiagnosticWandb(args, report)
    exit_code = 0
    try:
        with interrupted_by_sigterm(), dashboard:
            evaluate(args, report, dashboard, tracker, policy_factory, loader_factory)
    except (KeyboardInterrupt, InterruptedError) as exc:
        report.summary["status"] = "interrupted"
        report.issue(
            "evaluation",
            str(exc) or "KeyboardInterrupt",
            error=True,
            sample=report.summary.get("current_sample"),
            condition=report.summary.get("current_condition"),
        )
        exit_code = 130
    except Exception as exc:
        report.summary["status"] = "failed"
        report.issue(
            "evaluation",
            f"{type(exc).__name__}: {exc}",
            error=True,
            sample=report.summary.get("current_sample"),
            condition=report.summary.get("current_condition"),
        )
        exit_code = 1
    finally:
        dashboard.close()
        report.save()
        try:
            report.figures()
        except Exception as exc:
            report.issue("figures", f"{type(exc).__name__}: {exc}", error=True)
        report.save()
        tracker.finish()
        report.save()
        dashboard.final(report)
    return exit_code
