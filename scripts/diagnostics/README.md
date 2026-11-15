# Diagnostics

Run these from the repository root when investigating a specific issue. They
are not normal training entrypoints.

- `benchmark_checkpoint_io.py`, `summarize_model_load.py`, and
  `smoke_distributed_model_loading.py`: checkpoint and distributed loading.
- `diagnose_concerto_precision.py`, `diagnose_fixed_batch.py`,
  `smoke_test_concerto_point_encoder.py`, and `smoke_test_sparsh_tactile.py`:
  sensor and batch diagnostics.
- `validate_hf_config_alignment.py` and `verify_droid_rotation_correction.py`:
  configuration and data checks.

Example: `uv run python scripts/diagnostics/validate_hf_config_alignment.py`.
Some checks require local model files or GPUs; inspect their help before use.

## Training-free point / Sidecar evaluation

Run from the repository root with the existing environment. No training, checkpoint
writes or Git operations are performed. Choose a new/empty output directory.

```bash
PYTHONPATH=. .venv/bin/python scripts/diagnostics/evaluate_sidecar.py \
  --checkpoint /path/to/exported_checkpoint --dataset-path /path/to/lerobot \
  --embodiment-tag new_embodiment --episode-ids 0 1 \
  --point-modes original disabled --gate-diagnostics --gate-ablation \
  --seed 42 --inference-steps 4 --max-samples 16 --output-dir /tmp/wah-eval

PYTHONPATH=. .venv/bin/python scripts/diagnostics/evaluate_sidecar.py \
  --offline-jsonl /path/to/multimodal_diagnostics.jsonl --output-dir /tmp/wah-offline
```

For swaps or task filtering, supply `--swap-metadata /path/to/provenance.json`:

```json
{
  "0": {"task": "lift_can", "embodiment": "new_embodiment", "coordinate_frame": "robot_base_v1", "sensor": "calibrated_scene_camera_v1", "scene": "scene_a"},
  "1": {"task": "lift_can", "embodiment": "new_embodiment", "coordinate_frame": "robot_base_v1", "sensor": "calibrated_scene_camera_v1", "scene": "scene_b"}
}
```

Add `--task lift_can --point-modes original disabled swapped`. Metadata must be
verified against dataset provenance; labels cannot prove calibration. The donor
is a different episode of the same task, embodiment, sensor and coordinate frame.
Different episodes may share a scene. Donor history is extracted at the first
valid observation step; incompatible fields/shapes fail explicitly. Swaps measure
OOD sensitivity, not grounding evidence. `insert_HDMI` is also accepted.

The evaluator reuses Gr00tPolicy checkpoint restoration, LeRobotEpisodeLoader,
extract_step_data, the checkpoint processor and decode_action. Only complete
inference exports are supported for functional interventions; ZeRO-3 gate reads
use the existing full_value helper, but live sharded interventions are rejected.
Point removal uses a temporary DiT input hook, removing both tokens and mask.
Contact modes that require scene points reject removal; their architecture is
unchanged. Input Contact-MoE is not disabled by the intervention.

Preprocessing and model inference independently replay Python, NumPy and Torch
RNG streams. The first actual action-encoder input is compared exactly across
all conditions and failure aborts evaluation. This checks initial noise equality;
GPU kernel nondeterminism may still affect outputs. Full valid temporal histories
are selected; episode boundary padding is not introduced.

`summary.json`, `sample_metrics.csv`, and `stage_metrics.csv` contain provenance,
gates, zero-based block indices, one-based block numbers, per-call residual norms,
ratios and cosines, and action MAE/RMSE, dimension/timestep MAE and per-chunk L2.
Normalized metrics include the checkpoint padded action layout. Decoded metrics
are separate per named action field, in checkpoint/dataset native units; mixed
joint/gripper dimensions are not combined into a physical-distance claim.
Residual norms are FP32 Frobenius norms over valid tokens, per stage and flow
step. Gate graphs show recorded values and residual graphs average recorded calls.
Offline JSONL values are reused directly; absent stage/block mappings and gradients
are not inferred. Output CSV array cells are JSON arrays.

Geometry is explicitly `unavailable`: this tool has no verified object-pose GT,
Cartesian action contract or FK adapter. No geometry scatter or physical EE error
is fabricated. Missing plot reasons are in the summary. Closed-loop integration
is deferred; existing UniVTAC rollout files are untouched.

For the optional new architecture and its diagnostic stage mapping, see
[Multi-level Local Sidecar](../../docs/multilevel_local_sidecar.md).
