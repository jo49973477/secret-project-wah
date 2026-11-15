#!/usr/bin/env python3
"""Fixed-observation LeRobot evaluation; never starts training."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from gr00t.experiment.sidecar_evaluation import (
    export,
    gate_rows,
    initial_noise_probe,
    intervention,
    offline_rows,
    replay_rng,
    sensitivity,
)
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--task", choices=["lift_can", "insert_HDMI"])
    parser.add_argument("--episode-ids", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--point-modes",
        nargs="+",
        choices=["original", "disabled", "swapped"],
        default=["original", "disabled"],
    )
    parser.add_argument(
        "--swap-metadata",
        type=Path,
        help="JSON episode IDs -> task, embodiment, coordinate_frame, sensor, scene",
    )
    parser.add_argument("--gate-diagnostics", action="store_true")
    parser.add_argument("--gate-ablation", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offline-jsonl", type=Path)
    args = parser.parse_args()
    # Fresh directory prevents accidental overwrite, including checkpoint/data paths.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty")
    summary = dict(
        mode="offline" if args.offline_jsonl else "fixed_observation",
        seed=args.seed,
        geometry={
            "status": "unavailable",
            "reason": "No verified object-pose GT, Cartesian action contract or FK supplied",
        },
        interpretation="Swapped points measure OOD sensitivity, not evidence of grounding",
    )
    if args.offline_jsonl:
        rows = offline_rows(args.offline_jsonl)
        summary.update(
            source=str(args.offline_jsonl),
            stage_mapping="Preserved when present in logs; unavailable for legacy records without a mapping",
            gradients="not inferred",
        )
        export(args.output_dir, summary, [], rows)
        return
    if not args.checkpoint or args.dataset_path is None:
        parser.error("Evaluation requires --checkpoint and --dataset-path")
    if args.seed < 0 or min(args.max_samples, args.inference_steps) < 1:
        parser.error("Sample count and inference steps must be positive")
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import Gr00tPolicy, _prepare_policy_model_inputs

    policy = Gr00tPolicy(args.embodiment_tag, args.checkpoint, device=args.device)
    policy.model.eval()
    head = policy.model.action_head
    dit = head.model
    head.num_inference_timesteps = args.inference_steps
    gates = gate_rows(dit)
    configs = deepcopy(policy.get_modality_config())
    loader = LeRobotEpisodeLoader(args.dataset_path, configs)
    configs.pop("action")
    metadata = json.loads(args.swap_metadata.read_text()) if args.swap_metadata else {}
    if (args.task or "swapped" in args.point_modes) and not metadata:
        parser.error(
            "Task selection and swaps require --swap-metadata with verified episode provenance"
        )
    stages, samples = list(gates), []
    count = 0
    offsets = [i for config in configs.values() for i in config.delta_indices]
    first_step = max(0, -min(offsets))
    last_offset = max(0, max(offsets))
    for episode in args.episode_ids:
        meta = metadata.get(str(episode), {})
        if args.task and meta.get("task") != args.task:
            continue
        trajectory = loader[episode]
        for step in range(first_step, len(trajectory) - last_offset):
            if count >= args.max_samples:
                break
            point = extract_step_data(
                trajectory, step, configs, embodiment_tag=policy.embodiment_tag
            )
            variants = {"original": point}
            if "swapped" in args.point_modes:
                required = ("embodiment", "coordinate_frame", "sensor")
                if (
                    any(not meta.get(k) for k in (*required, "scene"))
                    or meta["embodiment"] != policy.embodiment_tag.value
                ):
                    raise ValueError(
                        "Swap provenance missing or embodiment differs from loaded policy"
                    )
                donors = [
                    int(k)
                    for k, v in metadata.items()
                    if int(k) != episode
                    and all(v.get(f) == meta[f] for f in required)
                    and v.get("scene")
                    and v.get("task") == meta.get("task")
                ]
                if not donors:
                    raise ValueError("No different compatible episode for swapped points")
                donor = donors[0]
                other = extract_step_data(
                    loader[donor], first_step, configs, embodiment_tag=policy.embodiment_tag
                )
                swapped = deepcopy(point)
                if (
                    not point.pointclouds
                    or not other.pointclouds
                    or point.pointclouds.keys() != other.pointclouds.keys()
                ):
                    raise ValueError("Swap requires matching nonempty pointcloud fields")
                for key in point.pointclouds:
                    if point.pointclouds[key].shape != other.pointclouds[key].shape:
                        raise ValueError("Swap pointcloud shapes differ")
                swapped.pointclouds = other.pointclouds
                variants["swapped"] = swapped
            conditions = [("original", False, ())]
            conditions += [
                (mode, mode == "disabled", ()) for mode in args.point_modes if mode != "original"
            ]
            if args.gate_ablation:
                conditions += [(f"zero_stage_{i}", False, (i,)) for i in range(len(gates))]
                conditions += [("zero_all", False, tuple(range(len(gates))))]
            results = {}
            initial_noise = None
            for condition, disabled, zero in conditions:
                observation = variants.get(condition, point)
                # Seed preprocessing independently, then replay the exact model RNG state.
                with replay_rng(args.seed + count):
                    processed = policy.processor(
                        [{"type": MessageType.EPISODE_STEP.value, "content": deepcopy(observation)}]
                    )
                    inputs = _prepare_policy_model_inputs(
                        policy.collate_fn([processed]), policy.model
                    )
                rows = [] if args.gate_diagnostics else None
                with (
                    intervention(dit, disabled=disabled, zero_stages=zero, rows=rows),
                    replay_rng(args.seed + count),
                    torch.inference_mode(),
                    initial_noise_probe(head.action_encoder) as noise,
                ):
                    output = policy.model.get_action(**inputs)
                if not noise:
                    raise RuntimeError("Could not verify initial action noise")
                if initial_noise is None:
                    initial_noise = noise[0]
                elif not torch.equal(initial_noise, noise[0]):
                    raise RuntimeError("Initial action noise differs across conditions")
                normalized = output["action_pred"].float().cpu().numpy()
                state = {k: np.stack([v]) for k, v in point.states.items()}
                physical = policy.processor.decode_action(normalized, policy.embodiment_tag, state)
                results[condition] = normalized, physical
                for row in rows or []:
                    stages.append(
                        dict(sample=count, episode=episode, step=step, condition=condition, **row)
                    )
                if condition != "original":
                    samples.append(
                        dict(
                            sample=count,
                            episode=episode,
                            step=step,
                            condition=condition,
                            space="normalized",
                            **sensitivity(results["original"][0], normalized),
                        )
                    )
                    for key, value in physical.items():
                        samples.append(
                            dict(
                                sample=count,
                                episode=episode,
                                step=step,
                                condition=condition,
                                space=f"decoded:{key}",
                                **sensitivity(results["original"][1][key], value),
                            )
                        )
            count += 1
    if not count:
        raise ValueError("No evaluation samples matched the selection")
    summary.update(
        checkpoint=args.checkpoint,
        checkpoint_identity=policy.checkpoint_identity,
        task=args.task,
        episode_ids=args.episode_ids,
        point_modes=args.point_modes,
        swap_provenance=metadata,
        seed_rule="seed + sample_index; replayed separately for preprocessing and inference",
        dataset=str(args.dataset_path),
        samples=count,
        inference_steps=args.inference_steps,
        initial_noise_verified=True,
        gates=gates,
        action_units="Normalized: checkpoint padded layout; decoded: processor.decode_action per named field, native dataset units (not Cartesian metres)",
    )
    export(args.output_dir, summary, samples, stages)


if __name__ == "__main__":
    main()
