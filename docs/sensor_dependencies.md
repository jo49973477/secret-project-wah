# Sensor dependency preflight

## Automated setup and checks (WAH)

The repository provides `scripts/setup_sensor_dependencies.sh` to handle the
reviewed **Linux x86_64 / Python 3.12 / PyTorch 2.9.0+cu128** combination.
Use the target WAH `.venv`; create/synchronize its **base** environment first
(`uv sync --locked`) if needed, **before** manually installing sensors:

```bash
# Read-only: inspect the current installation and run the WAH import preflight.
bash scripts/setup_sensor_dependencies.sh --check

# Explicitly opt in to install pinned wheels + official Concerto, then check.
bash scripts/setup_sensor_dependencies.sh --install

# Optional: test actual one-GPU torch-scatter/spconv forward and backward CUDA ops.
CUDA_VISIBLE_DEVICES=0 bash scripts/setup_sensor_dependencies.sh --gpu-smoke
```

The script never runs `uv sync`, deletes packages, changes the CUDA driver,
replaces Torch/torchvision/Transformers/FlashAttention, loads GR00T weights, or
launches training. `--install` first verifies exact versions and refuses an
unknown OS/architecture/build; it also refuses conflicting installed sparse
packages. It uses `--no-deps` to keep the existing core libraries intact.
The official Concerto revision is pinned and built with `setuptools==80.9.0`
and `--no-build-isolation` because upstream `setup.py` still imports removed
`pkg_resources`. If a required support package is absent, inspect the failure
and resolve it intentionally rather than automatically replacing core packages.

`--check` validates imports, the official Concerto API, trainctl presence and
the same Hydra-based sensor preflight used before training. `--gpu-smoke` adds
a minimal CUDA operator/backward smoke test but **does not** establish BF16
Concerto checkpoint behavior or multi-GPU ZeRO-3 correctness. For that, use
the full diagnostic described below with representative real point-cloud data.

Running `uv sync` again can remove separately installed sensor wheels and
Concerto because they are not currently locked into the base project
dependencies; rerun `--install` after any such sync. To support different
hardware, design an appropriate wheel/source-build matrix before extending
the guarded installer.

`scripts/train_hydra.py` now checks enabled sensor imports once in the parent,
before dataset/resume access, output snapshots, Torchrun, or GR00T loading.
`scripts/train_univtac_sidecar.py` and its shell wrapper delegate to that same
Hydra launcher. The internal `train_hydra_worker.py` is launched by the parent;
invoke the parent entry point rather than starting that internal worker directly.
`dry_run=true` remains a configuration-only check and does not probe imports.

The import probe uses `sys.executable` (also used to launch workers) in a fresh
process. It does not import the GR00T encoders or construct models, read weights,
call download APIs, run kernels, initialize distributed training, or call CUDA
initialization. Import-side RNG/module state cannot reach training. Python
bytecode writes, Hugging Face online access, and spconv/cumm import-time JIT
compilation are disabled **in the probe only**. Source installations needing JIT
must be built separately first. No environment variables are changed in the
parent or workers. A timeout, native crash, missing package, invalid report, or
binary import error stops launch with the interpreter, failed import, original
exception, package name, and this installation guide.

| Enabled configuration | Checked imports |
| --- | --- |
| `point_encoder_cfg=concerto_small` / `concerto_base` | `spconv.pytorch`, `torch_scatter`, official `concerto`, `torch` |
| Concerto with `point_encoder_enable_flash=true` | Also `flash_attn` |
| `point_encoder_cfg=pointnet2` / `point_transformer` | `torch`; no Concerto stack required |
| `tactile_encoder_cfg=resnet18` | `torch`, `torchvision.models`, `numpy`, `PIL.Image` |
| `tactile_encoder_cfg=sparsh_dino_base` | The tactile imports above, plus `huggingface_hub`, `safetensors.torch` for bootstrap |

Disabled modalities require no optional sensor imports. Checks apply even if
backbones are frozen or embedded in a resumed GR00T checkpoint: architecture
construction still needs their code. Concerto's `load` and
`model.PointTransformerV3` APIs must be callable, preventing confusion with an
unrelated package named `concerto`. Transitive import errors (for example,
missing `addict` or an incompatible `timm`) are reported too. The local SPARSH
implementation is `gr00t/model/extension/sparsh_vit.py`; an external `sparsh`
package is not required, nor is `timm` required solely for this tactile backend.
`torchvision.models` is imported by `tactile_encoder.py` for both tactile choices.
With `point_encoder_enable_flash=null`, the eventual checkpoint controls whether
FlashAttention is needed; this import-only check does not read that checkpoint
to infer the setting. It does not change the configured attention backend.

## Install without replacing the GR00T stack

Use the same Python interpreter as the launcher. The target environment here is
**Python 3.12, PyTorch 2.9.0+cu128, torchvision 0.24.0+cu128, Transformers 4.57.3**.
Do not copy Concerto's standalone Conda environment: its upstream example uses a
different PyTorch/CUDA stack. Use its [package-mode instructions](https://github.com/Pointcept/Concerto#installation)
as a reference, preserving existing GR00T versions.

First inventory the environment without changing it:

```bash
.venv/bin/python -c 'import sys, torch; print(sys.executable, sys.version); print(torch.__version__, torch.version.cuda)'
uv pip freeze --python .venv/bin/python
nvcc --version
nvidia-smi
```

PyTorch's wheel runtime (`torch.version.cuda`), the system toolkit (`nvcc`), and
the driver's supported CUDA level are separate. For Python 3.12 on Linux x86_64,
the [official PyG wheel index](https://data.pyg.org/whl/torch-2.9.0+cu128.html)
contains `torch_scatter` wheels tagged `cp312` and `pt29cu128`. For that exact
Python/PyTorch/CUDA combination:

```bash
uv pip install --python .venv/bin/python --no-deps --only-binary=:all: --no-index \
  --find-links 'https://data.pyg.org/whl/torch-2.9.0+cu128.html' \
  'torch-scatter==2.1.2+pt29cu128'
```

Other Python versions, CPU architectures, or Torch builds need their matching
artifacts. If no matching binary exists, stop and plan a source build against
the existing stack; do not let installation silently fall back to a mismatched
wheel or replace Torch.

Install the **official** package, reviewing and recording its revision:

```bash
git clone https://github.com/Pointcept/Concerto.git /path/to/Concerto
git -C /path/to/Concerto rev-parse HEAD
uv pip install --python .venv/bin/python --no-deps /path/to/Concerto
```

The official package's setup metadata does not enumerate all import dependencies.
Review missing support packages such as `timm`, `addict`, `scipy`, and
`huggingface-hub` against the existing environment. Use `--no-deps` for additions
and retain GR00T's versions of already installed core packages. This deliberately
requires reviewing transitive dependencies rather than letting a resolver change
Torch, CUDA libraries, Transformers, NumPy, or FlashAttention. Do not install
Concerto's full visualization/demo environment just to use the encoder.

## spconv-cu126 with cu128 Torch

`spconv-cu126` denotes a CUDA 12.6 build, whereas `torch==2.9.0+cu128` carries the
CUDA 12.8 runtime. This difference is **not by itself proof of incompatibility**:
[spconv documents CUDA minor-version compatibility](https://github.com/traveller59/spconv#prebuilt)
and independence from the PyTorch binary ABI. It is also **not proof that this
specific pair works**. Driver support, Python `cp312`/platform tags, `cumm`
version, shared-library resolution, GPU architecture, runtime compilation, and
forward/backward kernels still matter. `torch_scatter`, unlike spconv, needs the
correct PyTorch/CUDA extension build.

The local environment at validation time already contained:

```text
Python          3.12.14
PyTorch         2.9.0+cu128
torchvision     0.24.0+cu128
concerto        1.0
spconv-cu126    2.3.8
cumm-cu126      0.7.11
torch-scatter   2.1.2+pt29cu128
Transformers    4.57.3
FlashAttention  2.8.3
```

The **import-only** preflight passed for this installed stack. No packages were
installed or replaced by this change. That result does not validate CUDA kernel
execution or two-GPU training.

For a separately reviewed installation reproducing this candidate sparse stack,
[spconv-cu126](https://pypi.org/project/spconv-cu126/) 2.3.8 declares
`cumm-cu126>=0.7.11,<0.8.0` and requires `pccm`, `ccimport`, `pybind11`, `fire`,
and NumPy. Inventory those support packages first. An explicit candidate install
that will not resolve or replace core dependencies is:

```bash
uv pip install --python .venv/bin/python --no-deps --only-binary=:all: \
  'spconv-cu126==2.3.8' 'cumm-cu126==0.7.11'
```

`--no-deps` leaves missing prerequisites for you to resolve deliberately; it is
not a complete dependency installer. Do not mix CPU `spconv` or multiple
`spconv-cu*`/`cumm-cu*` distributions in one environment. Review the existing
installation before changing these packages; no automatic uninstall is performed.
Do not downgrade Torch to cu126, upgrade CUDA, or rebuild FlashAttention simply
because this package suffix differs.

Once imports pass, an authorized GPU validation should exercise the existing
`scripts/diagnostics/diagnose_concerto_precision.py` and sensor smoke tests on the target GPU,
including backward propagation in the real training configuration. This remains
necessary to validate FP16 sparse kernels and the deployed driver/runtime pair.
The preflight deliberately does not run those GPU operations or load the 3B model.

## Check dependencies without launching training

From the repository root, with the same environment and overrides as training:

```bash
NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from gr00t.configs.hydra_config import build_gr00t_config
from gr00t.utils.sensor_dependency_preflight import check_sensor_dependencies

with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base='1.3'):
    cfg = compose(config_name='experiment/univtac_full', overrides=['model=gr00t_sidecar'])
check_sensor_dependencies(build_gr00t_config(cfg).model)
print('Sensor dependency imports passed; no model loaded.')
PY
```

Focused CPU tests use fake installed/missing/broken packages in real child
interpreters, exercising the import probe without requiring a sparse CUDA stack:

```bash
NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python -m pytest \
  tests/gr00t/utils/test_sensor_dependency_preflight.py -q
```
