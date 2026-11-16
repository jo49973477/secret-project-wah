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

## WAH 7.1.1 / 7.1.2: paired diagnostics and reporting

`evaluate_wah_grounding.py` is the dedicated training-free CLI for these two
experiments. It reuses `Gr00tPolicy` (including WAH restoration validation and
checkpoint identity), `LeRobotEpisodeLoader`, `extract_step_data`, the saved
UniVTAC processor/collator, `get_action`, and `processor.decode_action`. It never
starts a Trainer, exports weights, saves model parameters, or invokes Git.
Use complete local inference checkpoints, including exports produced by
`scripts/export_zero3_checkpoint.py`. Raw/live ZeRO-3 partitions are rejected
with export instructions. Both `legacy` and `multilevel_local` Sidecars are
supported; the latter has no `point_gate`, which is reported as N/A.

From the repository root, in the existing environment:

```bash
uv run --no-sync python scripts/diagnostics/evaluate_wah_grounding.py --help

# 7.1.1: original, Sidecar Point-to-Action disabled, different-scene cloud
uv run --no-sync python scripts/diagnostics/evaluate_wah_grounding.py \
  --experiment point_ablation \
  --checkpoint "$CKPT" --dataset-path "$DATASET" \
  --task lift_can --num-samples 100 --seed 42 --device cuda:0 \
  --output-dir outputs/diagnostics/7_1_1

# 7.1.2: baseline, each stage action_gate=0, all action_gates=0
uv run --no-sync python scripts/diagnostics/evaluate_wah_grounding.py \
  --experiment gate_diagnostics \
  --checkpoint "$CKPT" --dataset-path "$DATASET" \
  --task lift_can --num-samples 100 --seed 42 --device cuda:0 \
  --output-dir outputs/diagnostics/7_1_2
```

The output directory must be new/empty and outside the checkpoint/dataset trees.
`--embodiment-tag` defaults to `new_embodiment`. Inference uses the checkpoint's
flow step count; `--inference-steps N` explicitly overrides it and records the
override. `--task insert_HDMI` uses the same machinery later, with matching data.
Task selection accepts the converter's exact labels `lift the can` and
`insert the HDMI connector`, or explicit task provenance below. It does not
guess task identities from file paths. CPU is accepted via `--device cpu`,
although actual Concerto/sparse model dependencies may require CUDA; the unit
tests use small CPU modules instead of a full checkpoint.

### What the interventions measure

- **7.1.1:** all three unordered condition pairs are compared. `disabled` sets
  every Sidecar **action_gate** to zero temporarily. Point encoders, point
  evolution, tactile, point-dependent ContactSurprise, and input MoE remain
  active. This differs from the older `evaluate_sidecar.py` token-removal mode.
  For `swapped`, only the donor cloud replaces the source cloud during donor
  preprocessing; all other collated fields must match exactly. The donor's
  encoded points are routed only to the legacy Sidecar pool or the corresponding
  multilevel local stages. DiT/contact/MoE continue receiving the original
  sensor inputs. Temporal Sidecars receive donor-point-derived fused tokens
  computed using the original tactile and state histories.
- **7.1.2:** reads the actual runtime `stage_indices`, scalar gate values and
  point-level mapping. Existing forward observer callbacks measure **A** and
  the actual gated **delta A** immediately before addition; no residual is
  reconstructed by subtracting final activations. Records FP32 Frobenius norms,
  `||delta A|| / ||A||`, and `cos(A, delta A)` over valid tokens for every stage
  and flow call. Reports original-versus-each-stage and original-versus-all-stage
  action differences. Temporary gates, hooks, and previous observers are restored
  even if a forward fails. Checkpoint files are never modified.

All conditions reuse the exact same preprocessed RGB, language, robot state,
tactile/history and structural action mask. Python, NumPy, CPU and CUDA RNG
states are replayed without advancing the caller's streams. The sample seed is
`(seed + sample_index) % 2**32`. The first actual action-encoder input (masked
initial diffusion noise) is checked for exact equality and hashed, including
BF16 support. A mismatch fails the evaluation. This verifies paired initial
noise; it does not eliminate nondeterminism in GPU kernels.

Samples are unique, seeded, uniformly selected valid frames, processed in
chronological order for efficient loading. Full observation histories **and**
the checkpoint action horizon fit inside their episode; no boundary padding is
introduced. Demonstration actions are never passed to `get_action`. When fewer
frames are available, the shortfall is explicit. These are descriptive frame
statistics, not independent trials or closed-loop success estimates.

### Scene provenance and geometry

Swaps require explicit different scene IDs: different episode IDs alone do not
prove different scenes. Supply `--scene-metadata /path/to/provenance.json` or
place the file at `$DATASET/meta/wah_diagnostics.json`:

```json
{
  "episodes": {
    "0": {
      "task": "lift_can",
      "embodiment": "new_embodiment",
      "coordinate_frame": "robot_base",
      "sensor": "head+wrist-calibration-v1",
      "scene": "scene-a",
      "object_id": "can-1",
      "object_xyz": [0.30, 0.10, 0.05],
      "object_xyz_units": "m",
      "object_xyz_frame": "robot_base",
      "object_xyz_source": "verified annotation reference"
    },
    "1": {
      "task": "lift_can",
      "embodiment": "new_embodiment",
      "coordinate_frame": "robot_base",
      "sensor": "head+wrist-calibration-v1",
      "scene": "scene-b",
      "object_id": "can-1",
      "object_xyz": [0.35, 0.10, 0.05],
      "object_xyz_units": "m",
      "object_xyz_frame": "robot_base",
      "object_xyz_source": "verified annotation reference"
    }
  }
}
```

These numbers illustrate the schema; replace them with verified dataset
annotations. The evaluator checks declared compatibility, not the truth of a
calibration/annotation. Donors must share task, embodiment, sensor and coordinate
frame and have a distinct scene ID. The first compatible episode in dataset
order supplies its first valid history window, and donor episode/frame are
recorded. Per-frame object positions can override episode-level geometry via
`"frames": {"12": {"object_xyz": [0.3, 0.1, 0.05]}}`; do not declare a static
episode position for a moving object. Missing/incompatible cloud provenance or
shapes skip the swapped comparisons explicitly while retaining original versus
disabled results.

For compatible object identity, metre units, a shared point/object coordinate
frame, and finite XYZ, the report includes object displacement, normalized
action MAE versus displacement, and normalized MAE per metre (undefined when
displacement is zero). The scatter is **sensitivity to mismatched observations**;
large differences do not imply correct geometry grounding. Missing object XYZ
is N/A; neither object poses nor coordinates are inferred from point centroids.

UniVTAC currently predicts relative **joint** targets. `decode_action` uses the
saved normalization and original state to recover absolute named fields in
native units, but there is no verified robot-specific FK adapter in this repo.
Therefore Cartesian end-effector response, geometric target error, directional
alignment and physical grasp accuracy remain **unsupported/N/A** for joint
checkpoints. Joint components are never relabeled XYZ or metres.

For a checkpoint that already predicts EEF XYZ in an explicit `xyz+rot6d` or
`xyz+rotvec` action format, an optional top-level `cartesian_action` object may
declare `field`, `units: "m"`, `coordinate_frame`, and `source`. Only when it
matches the saved EEF action config and both object frames does the evaluator
report mean decoded XYZ response across valid timesteps. This is still a
response metric, not accuracy against a verified desired action. The CLI does
not assume an EEF contract for `default` or joint-space formats.

### Terminal and local reports

The terminal uses the project's Rich `Live` / `Panel` / `Table` conventions,
with a live bar, task, checkpoint, samples, elapsed time, ETA and current
condition. The display closes on exceptions. Non-TTY/SSH redirected logs use
bounded plain progress lines without terminal escapes; `TERM=dumb` selects
this fallback too. Final tables show MAE mean/std/count, native decoded fields,
stage gates, residual ratios/cosines, stage-ablation MAE and available geometry.

Every evaluation that passes path validation writes these files, including on
loading/inference errors:

| File | Contents |
| --- | --- |
| `summary.json` | Config, identity, sample selection/seeds, sensor contract, provenance, status, aggregated metrics, errors and skipped reasons |
| `summary.csv` | Tidy action/stage/geometry statistics and explicit errors/skips |
| `per_sample.csv` | Paired masked MAE/RMSE, valid element counts, dimension/time MAE arrays, condition labels, seed and noise hash |
| `stage_metrics.csv` | Every observed stage/flow call: gates, actual block mapping, norms, ratio and cosine |
| `stage_summary.csv` | Stage aggregates, flow-call counts and contributing sample counts |
| `geometry.csv` | Scene pair metadata displacement/sensitivity and optional decoded Cartesian response |
| `results.json` | Detailed sample, stage and geometry records, with JSON `null` for unavailable values |
| `*.png` | Available action bars, stage gate/residual/cosine bars and geometry scatter |

Normalized MAE applies the processor's complete horizon/dimension mask, so padded
actions cannot dilute it. Decoded metrics retain separate field names and native
units. Statistics use population std (`ddof=0`) and contributing sample counts.
Stage statistics first average calls within each frame, then weight frames
equally; repeated diffusion calls are not counted as independent samples.
CSV scalar unavailable values display `N/A`; array cells use JSON, with `null`
for masked dimensions/timesteps. A zero injected residual has zero norm/ratio
when A is nonzero, but cosine is **N/A**. If A itself has zero norm, the ratio
is also N/A. No arbitrary pass/fail thresholds are applied.

Completed conditions are saved using atomic file replacement. Ctrl-C or SIGTERM
records `interrupted`, preserves completed conditions (including partial samples),
restores gates and exits 130. Other evaluation failures exit 1; argument/path
errors exit 2 before creating output. Files are individually atomic, not a
cross-file transaction; abrupt power loss or SIGKILL cannot guarantee the latest
condition. There is no automatic resume into a nonempty directory.

### Optional Weights & Biases

W&B is off unless `--wandb` is present. Local evaluation never needs W&B
connectivity. Append, for example:

```bash
--wandb --wandb-project wah-univtac \
--wandb-group diagnostics-7.1.2 --wandb-name lift-can-gates \
--wandb-mode offline
```

`--wandb-mode` accepts `online`, `offline`, or `disabled`. The project default
matches `configs/training/finetune.yaml` (`wah-univtac`), while each diagnostic
gets a new run ID, `job_type=wah-diagnostics`, diagnostic tags, and a default
`diagnostics-<experiment>` group. It never resumes a training run. Git discovery
and code saving are explicitly disabled in W&B settings. Initialization uses a
bounded timeout; initialization/logging/finalization errors are recorded locally
and do not fail an otherwise successful evaluation.

Run config records identity, variant, task, conditions, exact sample seeds,
sample selection/count, saved sensor/action contracts and inference step count.
W&B receives aggregate and per-task mean/std/count metrics, sample sensitivity
series, stage and geometry tables, available bar/scatter PNGs, and a
`wah-diagnostics` Artifact containing detailed JSON/CSV/PNG files. Series use
`diagnostic_sample` and `diagnostic_stage` axes; no `training_step` is used.
Offline runs stay local under the output directory until the user explicitly
syncs them. `--wandb-mode disabled` does not import or initialize W&B.

CPU contract tests (no checkpoints, training, simulator, W&B service or Git):

```bash
uv run --no-sync python -m pytest \
  tests/gr00t/model/test_wah_grounding_diagnostics.py -m "not gpu" -q --timeout=300
```
