# CLAUDE.md — Isaac GR00T N1.7

## Project overview

Isaac GR00T N1.7 is an open vision-language-action (VLA) model for generalized humanoid robot skills.
The repo contains the model, training pipeline, evaluation harness, and deployment tooling.

- **Language:** Python 3.12
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Build system:** setuptools (see `pyproject.toml`)

## Quick-start commands

```bash
# Install (dev mode with all extras)
uv sync --all-extras

# Lint and format (uses ruff via pre-commit)
pre-commit run --all-files

# Run CPU tests
python -m pytest tests/ -m "not gpu" -v --timeout=300

# Run GPU tests
python -m pytest tests/ -m gpu -v --timeout=300

# Build package
uv build

# Validate lockfile
uv lock --locked
```

## Code style

- Formatter: `ruff format` (double quotes, spaces, line-length 100)
- Linter: `ruff check` with rules E, F, I (ignores E501)
- Config lives in `pyproject.toml` under `[tool.ruff]`
- Run `pre-commit run --all-files` before committing

## Directory layout

```
gr00t/              # Main package
  configs/          #   Training, data, and model configs
  data/             #   Data loading, embodiment tags, dataset processing
  eval/             #   Evaluation (run_gr00t_server.py)
  experiment/       #   Training pipeline (launch_finetune.py, trainer.py)
  model/            #   Model architecture (N1.7, base, modules)
  policy/           #   Policy inference (Gr00tPolicy, server/client)
examples/           # Per-embodiment example configs and READMEs
scripts/            # Deployment, conversion, and utility scripts
  deployment/       #   Platform install scripts (dgpu, orin, thor, spark)
tests/              # pytest suite (markers: gpu, not gpu)
getting_started/    # User-facing guides and notebooks
```

## Key entry points

- **Fine-tune:** `bash examples/finetune.sh --base-model-path <path> --dataset-path <path> --embodiment-tag <tag> --output-dir <dir>`
- **Inference server:** `python gr00t/eval/run_gr00t_server.py --model-path <path> --embodiment-tag <tag>`
- **ONNX export:** `python scripts/deployment/export_onnx_n1d7.py`
- **TensorRT build:** `python scripts/deployment/build_trt_pipeline.py`
- **Benchmark:** `python scripts/deployment/benchmark_inference.py`

## Testing

- Test markers: `gpu` (requires GPU), default is CPU-safe
- Fixtures live in `tests/fixtures/` and `demo_data/`
- CI runs CPU and GPU tests in separate jobs with 300s timeout

## Deployment platforms

- **dGPU (H100, A100, RTX):** CUDA 12.8 — install via `scripts/deployment/dgpu/install_deps.sh`, container via top-level `docker/Dockerfile` (supports x86_64 and aarch64)
- **Jetson Orin:** JetPack 7.2 / CUDA 13.2 — install via `scripts/deployment/orin/install_deps.sh`, container via `scripts/deployment/orin/Dockerfile`
- **Jetson Thor:** JetPack 7.2 / CUDA 13.2 — install via `scripts/deployment/thor/install_deps.sh`, container via `scripts/deployment/thor/Dockerfile`
- **DGX Spark:** CUDA 13.0 — install via `scripts/deployment/spark/install_deps.sh`, container via `scripts/deployment/spark/Dockerfile`

Each Jetson/Spark platform ships an `activate_*.sh` helper (`scripts/activate_orin.sh`, `scripts/activate_spark.sh`, `scripts/activate_thor.sh`) that exports platform-specific library paths. For dGPU, the standard `source .venv/bin/activate` is sufficient.

# WAH Project Instructions

Follow the global AGENTS.md instructions in addition to
these project-specific rules.

## 1. Logging & Monitoring
- Integrate new functionality and experiments into the
  existing Rich Terminal and WandB workflows.
- Use Rich Terminal for readable real-time progress,
  metrics, and status reporting.
- Use WandB to persist training metrics, configurations,
  and experiment comparisons.
- Reuse existing logging infrastructure.
- For rollouts, Rich Terminal monitoring is sufficient.
  Do not modify rollouts solely to add WandB logging.
- Avoid meaningless WandB metrics for non-training code.

## 2. Experiment Naming
- Name new training methods and architectures with
  three components in this order:
  [specific change]_[overall architecture]_[DiT scope]
- Component 1: e.g., temporal, pointonly.
- Component 2: e.g., sidecar, multilevel.
- Component 3: ditfrozen, ditlora, or ditfull.
- Examples:
  temporal_sidecar_ditlora
  pointonly_multilevel_ditfull
- Follow existing repository separator conventions.
- User-specified names always take precedence.
- Never rename existing experiments or checkpoints
  without explicit permission.

## 3. Experiment Reproducibility
- Preserve existing training configurations by default.
- Never silently change datasets, preprocessing,
  hyperparameters, or evaluation conditions.
- Keep configurations, seeds, and checkpoints traceable.
- Report changes affecting experiment comparisons.

## 4. Compatibility
- Preserve training, inference, and rollout interfaces.
- Avoid breaking existing configurations, checkpoints,
  and execution commands.
- Explain unavoidable breaking changes before proceeding.