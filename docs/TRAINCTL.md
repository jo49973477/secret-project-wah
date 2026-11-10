# Training supervision with trainctl

WAH Hydra launches `trainctl run → Torchrun → GR00T Trainer`. trainctl owns the
start, completed, failed, and checkpoint notifications. The Trainer callback
only forwards numeric metrics and completed checkpoint paths over the public
`from trainctl import tracker` SDK. Final success requires a zero process exit,
including final saving, loss assertions, and dataset cleanup.

Rollout supervision is a separate integration. It does not import Hugging Face
callbacks or alter the training persona packs described below. The deployment
controller launches `trainctl run → rollout SDK bridge → Isaac evaluator`; the
bridge consumes `logs/rollout_events.jsonl` and uses only the public
`tracker.init`, `define_metrics`, `log`, `notify`, `finish`, and `fail` APIs.
trainctl remains the authority for the supervised process exit status.

## Install

Python 3.12 is supported. WAH pins `trainctl[discord,nvml,telegram]==0.2.0` from
PyPI; `uv.lock` includes the release hashes and optional transports. The live
PyPI index was checked during migration: 0.2.0 is available. The published CLI,
SDK, supervisor, watcher and config match reference commit
`dd7a0a71b58df6ffce490b28f184d0e5ed51ae9f`.

For a fresh WAH environment, follow the platform installation guide, then:

```bash
uv sync --locked --all-extras
source .venv/bin/activate
trainctl --help
```

For an **existing working GPU environment**, install just the new dependencies
with exact direct versions, without syncing/replacing platform GPU packages:

```bash
uv pip install --python .venv/bin/python \
  'trainctl[discord,nvml,telegram]==0.2.0' 'tomli-w==1.2.0'
```

This pins direct dependencies; use the lockfile for fully reproducible
transitive dependencies. No training/model/GPU dependency versions were changed
by this migration.

## Rich dashboard and Discord integration

With trainctl enabled, the interactive parent displays the existing Rich panels
while rank zero publishes CPU scalar observations to a unique
`OUTPUT_DIR/logs/dashboard-*.jsonl`. The trainctl supervisor still owns Torchrun.
No worker is forced to claim a TTY. Nonzero ranks never publish display telemetry.
Progress, losses, gradients, learning rates, GPU allocation, speed/ETA, Samples
Seen, Effective Epoch, save status and optional multimodal diagnostics retain the
existing renderer and measurements. Resume starts at Trainer's restored step.

The parent coordinates worker warnings/tracebacks with Rich. It uses the primary
screen so important output remains in scrollback when the transient dashboard
closes. Resize is detected by Rich; normal exit, exceptions, Ctrl+C and SIGTERM
restore streams and the cursor. The display remains in finalizing state after
HF train_end until the supervised process, including final saving, actually exits.
Non-TTY runs retain clean text logging, with no dashboard control sequences.

Discord environment-variable precedence is:

1. `TRAINCTL_DISCORD_WEBHOOK_URL`
2. `DISCORD_WEBHOOK_URL`
3. Legacy `DISCORD_WEBHOOK`

WAH normalizes the selected value in the child environment. Conflicting values
produce a safe warning naming the precedence, never the URLs. An explicit
`[notifications.discord] enabled = false` in the trainctl TOML remains authoritative.
Startup prints persona, Discord configuration status, display mode and checkpoint
notification status. Missing/invalid webhooks and missing httpx produce actionable
warnings; configured means validated offline, not a claim of remote delivery.

trainctl's original transport reports rejected HTTP 401/403/404, exhausted 429
rate limits, server errors and network failures, without leaking request URLs or
interrupting training. These appear in the terminal and `logs/train.log`.
A missing SDK endpoint produces an explicit worker warning: while training can
continue, SDK metrics and completed-checkpoint alerts need IPC to work. Filesystem
polling remains disabled to avoid announcing half-written ZeRO checkpoints.

Default `training.save_steps=500` means Step 100 alone does not produce a
checkpoint alert. Metrics never trigger chat notifications.

```bash
export FLORENCE_DATASET=/home/spl_yeongyoo/datasets/univtac_new
# Set a newly rotated webhook through your private shell/secret manager:
export TRAINCTL_DISCORD_WEBHOOK_URL="<configured-securely>"

CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/train_florence_ablations.sh B \
  hardware=dual_3090_zero3 \
  hardware.global_batch_size=2 \
  hardware.gradient_checkpointing=true \
  training.gradient_accumulation_steps=16 \
  trainctl.enabled=true \
  trainctl.message_mode=sopia \
  training.enable_training_dashboard=true
```

### Notification smoke test without a model

First inspect the exact WAH environment-variable resolution and TOML enablement
without network access or creating a trainctl run:

```bash
.venv/bin/python scripts/smoke_trainctl_notifications.py
```

After reviewing its safe diagnostics, explicitly send a tiny supervised run:

```bash
# Sends one start, one completed-checkpoint and one success notification:
.venv/bin/python scripts/smoke_trainctl_notifications.py --send
# Sends one start, one completed-checkpoint and one failure notification:
.venv/bin/python scripts/smoke_trainctl_notifications.py --send --fail
# Explicit persona override:
.venv/bin/python scripts/smoke_trainctl_notifications.py --send --message-mode koyori
```

`--fail` deliberately returns a nonzero status. Neither command loads GR00T,
starts Torchrun, allocates GPU tensors, or touches existing training artifacts.
The helper retains a unique `/tmp/wah-trainctl-smoke-*` directory with its own
SQLite database, completed smoke artifact and trainctl logs. An explicit TOML
Discord disable is still respected; remove it if you intend to test delivery.
Do not paste actual webhook values into commands captured by shared logs.

## Credentials and persona

Set secrets in the launching shell or a private environment file, never in
Hydra overrides, YAML, W&B config, or a checked-in script:

```bash
export TRAINCTL_DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/REPLACE/REPLACE'
export TRAINCTL_TELEGRAM_TOKEN='REPLACE'
export TRAINCTL_TELEGRAM_ALLOWED_USERS='123456789'
```

Either transport can be used alone. Create a Discord server webhook under
Server Settings → Integrations → Webhooks. For Telegram, create a BotFather bot,
start a conversation with it, and allowlist your numeric user ID. Multiple IDs
are comma-separated. No actual credentials are needed for local tracking.

```yaml
trainctl:
  enabled: true
  message_mode: sopia
  notify_checkpoints: true
```

Modes are `standard`, `koyori`, `sopia`, `roboco`, and `neuro`. For example:

```bash
bash scripts/train_florence_ablations.sh A trainctl.message_mode=roboco
bash scripts/train_florence_ablations.sh A trainctl.notify_checkpoints=false
bash scripts/train_florence_ablations.sh A trainctl.enabled=false
bash scripts/train_florence_ablations.sh A dry_run=true
```

`notify_checkpoints=false` disables checkpoint registration and its notifications
for this run; metrics and lifecycle still work. The current public SDK has no
per-checkpoint silent-registration option. `enabled=false` disables the adapter
and supervisor, including legacy notifications. `dry_run=true` only prints
configuration/preflight: it creates no trainctl run or success notification and
does not require trainctl to be installed.

The persona is passed to the CLI **before** training, so the automatic start
message uses the chosen mode. Messages come from trainctl's packaged definitions.
Do not wrap `train_hydra.py` in another `trainctl run`; Hydra rejects nesting.

## Rollout trainctl configuration

Rollouts use CLI settings because the existing rollout launchers do not use
Hydra. Keep transport credentials in the same private `TRAINCTL_*` environment
variables documented above:

```bash
bash scripts/run_gr00t_univtac_local.sh \
  --checkpoint "$CHECKPOINT" --policy wah \
  --sensor-contract /path/to/training-dataset/meta/modality.json \
  --model-python .venv/bin/python \
  --mode smoke --episodes-per-task 1 --tasks lift_can \
  --execution-horizon 1 --output-dir "$OUT" \
  --dashboard auto --rollout-diagnostics standard \
  --trainctl-enabled --trainctl-message-mode roboco \
  --rollout-message-style 3 --notify-task-complete \
  --notify-halfway --notify-errors
```

The control/model Python must contain trainctl; the Isaac/UniVTAC environment
does not need it. `--trainctl-enabled` fails before evaluation if the control
Python lacks a compatible tracker API. With trainctl disabled, the same command
still has the local dashboard and produces identical evaluation outputs.

Rollout metric names and definitions are:

| Metric | Definition |
|---|---|
| `rollout/completed_episodes` | Episode completion events, including errors |
| `rollout/total_episodes` | Selected tasks × planned episodes per task |
| `rollout/successes` / `failures` / `errors` | Existing rollout classification |
| `rollout/success_rate` | Successes ÷ (successes + failures); errors excluded |
| `rollout/error_rate` | Errors ÷ completed episodes |
| `rollout/episodes_per_second` | Completed episodes ÷ observed elapsed time |
| `rollout/inference_latency_ms` | Mean sampled policy request latency |
| `rollout/simulation_fps` | Mean measured simulation action rate |
| `rollout/gpu_memory_used_mib` | Available control-side GPU sample |

Unavailable and non-finite values are skipped. Metric updates never create chat
notifications. Notifications are limited to rollout start, global 50%, each task
completion, first/throttled error categories, critical infrastructure failure,
and final completion or abort. Task notifications always include the real
episode counts, S/F/E counts, rate, and elapsed time; final messages include both
macro and micro rates when defined. Delivery errors are isolated from simulation.

Rollout character modes are `koyori`, `roboco`, `neuro`, `sopia`, and `standard`.
The `sopia` identifier is preserved; `sophia` is only a compatibility alias.
`--rollout-message-style {1,2,3}` deterministically selects one of the three
Rollout variants from the canonical Notion specification for the entire run.
These messages are stored separately from trainctl's Training packs. Task
completion and recoverable episode errors are deliberately factual rather than
invented character dialogue. Dramatic Error dialogue is reserved for an aborted
rollout, and optimistic completion dialogue is suppressed if error episodes make
it contradict the result.

Discord and Telegram setup is identical to training. Never place webhook URLs or
tokens in checked-in YAML, shell history, or documentation. The complete rollout
dashboard, diagnostic levels, artifact list, full eight-task example, and failure
troubleshooting are in
[`examples/UniVTAC/CLOSED_LOOP_ROLLOUT.md`](../examples/UniVTAC/CLOSED_LOOP_ROLLOUT.md).

## Commands

Run from the WAH repository root after platform activation. A-D retain the
existing four-GPU ZeRO-3 profiles, optimizer settings, LR, W&B project/groups,
and effective batch 32. These commands start real training; use `dry_run=true`
first to inspect configuration without loading model weights.

```bash
export FLORENCE_DATASET=/absolute/path/to/full_UniVTAC_LeRobot_dataset
export FLORENCE_OVERFIT_DATASET=/absolute/path/to/independent_five_episode_dataset
# Optional local point encoder weights for offline setups:
export FLORENCE_CONCERTO_CHECKPOINT=/absolute/path/to/concerto_small.pth

# Temporal + DiT LoRA (A)
bash scripts/train_florence_ablations.sh A
# MoE + DiT LoRA (B)
bash scripts/train_florence_ablations.sh B
# Temporal + DiT Full (C)
bash scripts/train_florence_ablations.sh C
# MoE + DiT Full (D)
bash scripts/train_florence_ablations.sh D

# Overfit-5 debugging, explicit independent subset (not the full corpus)
bash scripts/train_florence_ablations.sh overfit5_temporal_lora
bash scripts/train_florence_ablations.sh overfit5_moe_lora
bash scripts/train_florence_ablations.sh overfit5_temporal_full
bash scripts/train_florence_ablations.sh overfit5_moe_full

# Single GPU, effective batch 32 (four samples × eight accumulation steps)
bash scripts/train_florence_ablations.sh A hardware=single_3090 \
  hardware.cuda_visible_devices=0 hardware.global_batch_size=4

# Two GPUs, effective batch 32 (two samples/GPU × two GPUs × eight accumulation)
bash scripts/train_florence_ablations.sh A hardware=dual_3090_zero3 \
  'hardware.cuda_visible_devices="0,1"' hardware.global_batch_size=4

# Four GPUs, same default effective batch 32
bash scripts/train_florence_ablations.sh A hardware=quad_3090_zero3 \
  'hardware.cuda_visible_devices="0,1,2,3"'

# Direct Hydra entrypoint; same supervisor and callback
uv run --no-sync python scripts/train_hydra.py \
  --config-name experiment/temporal_sidecar_ditlora-full

# Resume with the original name/output directory and W&B identity
bash scripts/train_florence_ablations.sh A \
  output_dir=/absolute/path/to/original/run \
  resume.enabled=true resume.auto_find_latest=true training.max_steps=20000
```

Device memory must fit the selected model and microbatch; these examples do not
claim a measured fit on every GPU. Changing GPU count is an explicit user choice,
not a monitoring change. The model, optimizer, scheduler, accumulation, and
checkpoint implementations are untouched. Notification settings may change on
resume without invalidating an otherwise compatible checkpoint. Hydra restores
the original run ID; trainctl creates a separate supervised invocation with the
same human-readable experiment/run-ID name.

## Status, logs, and W&B

```bash
uv run --no-sync trainctl config show
uv run --no-sync trainctl status
# Optional interactive Telegram /status, /gpu, /tail 20, /disk:
uv run --no-sync trainctl daemon
```

By default trainctl persists its database in
`~/.local/share/trainctl/trainctl.db` (`TRAINCTL_HOME` or `TRAINCTL_DATABASE` can
override it). WAH places trainctl's worker logs under
`OUTPUT_DIR/logs/trainctl/wah-trainctl-*/` per invocation. The existing combined
terminal output is retained in `OUTPUT_DIR/logs/train.log`, including worker
tracebacks. The launcher relays the public trainctl log because 0.2.0 captures
worker stdout/stderr without echoing it. Under supervision the console uses the
parent Rich dashboard in an interactive terminal, refreshed from rank-zero JSONL telemetry. Redirected stdout retains the existing clean text dashboard. The worker never needs a TTY and Rich frames never enter either log.

W&B remains enabled according to the existing `training.use_wandb` setting. Its
callbacks, run identity, project/group, model diagnostics, and step semantics are
unchanged. trainctl gets finite numeric values from the same `on_log` events,
including action/total losses, available gradient diagnostics, learning rate,
GPU allocation, timing, and throughput. It does not perform GPU synchronization
or send notifications on every step. Sample Seen and Effective Epoch use the
existing `SampleExposure` object, including resume-history suppression.

Checkpoints are registered after HF `on_save`, or after the explicit final save.
The SDK adapter deduplicates resolved paths; trainctl also deduplicates run/path.
Filesystem polling is disabled using its public TOML configuration because a
checkpoint directory appears before all ZeRO shards have been saved. Older
checkpoints are not re-announced on resume. Native ZeRO optimizer/scheduler/RNG
state, rotation, and resume behavior remain in the original code.

## Legacy migration and current library limits

`training.discord_*`, `training.telegram_*`, `DISCORD_WEBHOOK_URL`,
`TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID` are deprecated for Hydra. Move to the
`TRAINCTL_*` variables above. The old fields remain accepted for saved config
compatibility, but Hydra never instantiates their transports, even when trainctl
is disabled. Old notification modules and tests are retained for non-Hydra
entrypoints. The legacy Torchrun supervisor suppresses its fallback notification
when trainctl is active; Hydra no longer launches that supervisor.

WAH invokes the public `trainctl.cli.main` through a tiny entrypoint that maps
SIGTERM to the CLI's existing SIGINT cancellation path. That lets trainctl stop
and reap its separate Torchrun process group before exit. Failure diagnosis,
redaction, lifecycle events, transport isolation, and persona rendering remain
owned by the library. SIGKILL alone is not classified as CUDA OOM.

Minimal upstream improvements that would remove the compatibility bridges:

- Add public CLI worker-output streaming and SIGTERM cancellation support (WAH currently bridges both).
- Add an SDK-only checkpoint mode and silent checkpoint registration.
- Add periodic resource sampling/history to `run` if desired. In 0.2.0 NVML and
  system resources are queried by the daemon `/gpu` and `/disk`; the supervisor
  does not continuously persist resource history. WAH still forwards its existing
  GPU memory and throughput metrics, without inventing missing samples.

If trainctl itself or the host is killed, in-process cleanup and a final remote
notification cannot be guaranteed; an independent watchdog is an upstream
capability. No trainctl source is copied into WAH.

## Migration verification

Verified with Python 3.12 using CPU fixtures and mocked transports:

- 222 focused/regression tests passed (one pre-existing assertion deselected).
- 11 live CLI/SDK tests passed with local Unix sockets enabled, including real
  persisted metrics/checkpoints and SIGINT/SIGTERM cleanup of a supervisor child
  and its worker process.
- Ruff lint/format checks, `git diff --check`, and
  `UV_CACHE_DIR=/tmp/wah-uv-cache uv lock --locked --offline` passed.
- Dry-run command smoke checks passed for A/B/C/D, overfit-5, single GPU and
  two GPUs; no model loading or training processes were launched by these checks.
- Lockfile comparison found no removed/changed existing package versions or
  sources. Only trainctl and its lightweight dependencies were added.

The unfiltered initial regression run exposed
`test_legacy_flags_precision_and_loading_policy`: Hydra's existing
`save_steps: 500` disagrees with the legacy FinetuneConfig default `100`. Both
values are present at HEAD, independent of this migration. The assertion and
training settings were left unchanged. This is the one test deselected above.

The regression selection covered Hydra construction/resume/worker startup,
sample exposure, strict resume checks, optimizer parameter groups, multimodal
instrumentation, memory diagnostics, and retained legacy transports/callbacks.
A tiny CPU Hugging Face Trainer exercised real checkpoint state saving,
rotation, failed saving, and resume. Full GR00T/ZeRO-3 GPU runtime tests and real
Discord/Telegram delivery were intentionally not run.

Run the new focused tests locally (Unix socket permission is required):

```bash
TEST_CACHE_PATH=/tmp/wah-test-cache .venv/bin/python -m pytest \
  tests/gr00t/experiment/test_trainctl_callback.py \
  tests/gr00t/experiment/test_trainctl_launch.py \
  tests/gr00t/experiment/test_trainctl_trainer.py -q --timeout=60
```

## Modified files (original migration)

- `README.md`
- `configs/experiment/florence_ablation_base.yaml`
- `configs/experiment/florence_full_base.yaml`
- `configs/experiment/so100_baseline.yaml`
- `configs/experiment/univtac_full.yaml`
- `configs/experiment/univtac_overfit5.yaml`
- `configs/experiment/univtac_rgb.yaml`
- `configs/trainctl/default.yaml`
- `configs/training/finetune.yaml`
- `docs/HYDRA_CONFIGURATION.md`
- `docs/TRAINCTL.md`
- `docs/florence_ablation_suite.md`
- `gr00t/configs/hydra_config.py`
- `gr00t/configs/training/training_config.py`
- `gr00t/experiment/discord_notifier.py`
- `gr00t/experiment/experiment.py`
- `gr00t/experiment/hydra_launch.py`
- `gr00t/experiment/telegram_notifier.py`
- `gr00t/experiment/torchrun_supervisor.py`
- `gr00t/experiment/trainctl_callback.py`
- `gr00t/experiment/trainctl_cli.py`
- `gr00t/experiment/trainctl_launch.py`
- `gr00t/experiment/training_notification_callback.py`
- `pyproject.toml`
- `tests/gr00t/configs/test_hydra_config.py`
- `tests/gr00t/experiment/test_hydra_artifacts.py`
- `tests/gr00t/experiment/test_torchrun_supervisor.py`
- `tests/gr00t/experiment/test_trainctl_callback.py`
- `tests/gr00t/experiment/test_trainctl_launch.py`
- `tests/gr00t/experiment/test_trainctl_trainer.py`
- `uv.lock`

## Rich/Discord follow-up verification

The integration tests use real trainctl 0.2.0 CLI/SDK IPC with tiny CPU worker
stubs and httpx MockTransport. They do not send real notifications. PTY tests
exercise parent rendering, resizing, normal completion, worker failure, Ctrl+C,
SIGTERM, clean log retention and terminal restoration. Rank-zero guards are
verified for simulated two-rank callbacks; no real two-GPU training is claimed.

```bash
TEST_CACHE_PATH=/tmp/wah-test-cache .venv/bin/python -m pytest \
  tests/gr00t/experiment/test_dashboard_terminal.py \
  tests/gr00t/experiment/test_trainctl_launch.py \
  tests/gr00t/experiment/test_trainctl_callback.py \
  tests/gr00t/experiment/test_trainctl_dashboard.py \
  tests/gr00t/experiment/test_trainctl_discord.py -q --timeout=75
```

Additional follow-up files: `gr00t/experiment/dashboard_telemetry.py`,
`scripts/smoke_trainctl_notifications.py`,
`tests/gr00t/experiment/test_trainctl_dashboard.py`, and
`tests/gr00t/experiment/test_trainctl_discord.py`. Changes to the launcher,
existing dashboard, callback selection, startup diagnostics and schema defaults
leave model/optimizer/ZeRO/checkpoint code intact. Existing `pyproject.toml` and
`uv.lock` working-tree edits were preserved, not generated by this follow-up.

Follow-up results: 62 focused tests passed across the full 59-test run and three
additional cases (Trainer telemetry selection, offline-only smoke inspection,
and a rejected Discord transport preserving successful training exit). The six
lifecycle delivery variants were rerun together and passed. Another 146 existing
regression tests passed, with the previously documented checkpoint-default
assertion deselected. Ruff lint/format and `git diff --check` passed. The exact
B/two-GPU/Sopia command above passed `dry_run=true`, resolving effective batch
32; no workers or model weights were loaded by that dry run.

Confirmed integration causes were the explicit interactive-rendering exclusion
plus captured worker stdout, and missing support for the `DISCORD_WEBHOOK`
legacy alias. Actual remote webhook validity, permissions, rate limiting and
network access were not probed. Their failure paths were exercised with mocked
HTTP and remain visible in trainctl's original safe transport diagnostics.

### Files changed by this follow-up

- `README.md`
- `configs/trainctl/default.yaml` (existing Sopia edit preserved)
- `docs/HYDRA_CONFIGURATION.md`
- `docs/TRAINCTL.md`
- `docs/florence_ablation_suite.md`
- `gr00t/configs/hydra_config.py`
- `gr00t/experiment/dashboard_telemetry.py`
- `gr00t/experiment/hydra_launch.py`
- `gr00t/experiment/trainctl_callback.py`
- `gr00t/experiment/trainctl_launch.py`
- `gr00t/experiment/trainer.py`
- `gr00t/experiment/training_dashboard.py`
- `scripts/smoke_trainctl_notifications.py`
- `tests/gr00t/experiment/test_trainctl_dashboard.py`
- `tests/gr00t/experiment/test_trainctl_discord.py`
