#!/usr/bin/env bash
# WAH optional sensor dependency setup for the validated x86_64/cp312/cu128 stack.
# This script never calls uv sync or replaces the GR00T core stack.
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: bash scripts/setup_sensor_dependencies.sh [--check | --install] [--gpu-smoke]' \
    '' \
    '  --check       Validate installed packages and WAH sensor preflight (default).' \
    '  --install     Install pinned sensor wheels and official Concerto, then check.' \
    '  --gpu-smoke   Additionally test simple spconv and torch-scatter CUDA backward ops.' \
    '  -h, --help    Show this help.' \
    '' \
    "Run from any directory; targets only this repository's .venv." \
    'For full Concerto/ZeRO-3 testing, see docs/sensor_dependencies.md.'
}

mode=check
gpu_smoke=false
while (($#)); do
  case "$1" in
    --check) mode=check ;;
    --install) mode=install ;;
    --gpu-smoke) gpu_smoke=true ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="$repo_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  printf 'Missing %s. Create the base environment first with uv sync --locked.\n' "$python_bin" >&2
  exit 1
fi

# Check the target interpreter, not the shell's unrelated system Python.
stack="$($python_bin - <<'PY'
import platform
import sys

import torch
import torchvision
import transformers

print('|'.join((
    platform.system(),
    platform.machine(),
    f'{sys.version_info.major}.{sys.version_info.minor}',
    torch.__version__,
    str(torch.version.cuda),
    torchvision.__version__,
    transformers.__version__,
)))
PY
)"
IFS='|' read -r os arch py torch cuda torchvision transformers <<< "$stack"
printf 'WAH target: %s %s, Python %s, torch %s (CUDA %s), torchvision %s, transformers %s\n' \
  "$os" "$arch" "$py" "$torch" "$cuda" "$torchvision" "$transformers"

if [[ "$mode" == install ]]; then
  if [[ "$os/$arch/$py/$torch/$cuda/$torchvision/$transformers" != \
        'Linux/x86_64/3.12/2.9.0+cu128/12.8/0.24.0+cu128/4.57.3' ]]; then
    printf '%s\n' \
      'Refusing to install: pinned wheels are only reviewed for Linux x86_64,' \
      'Python 3.12 / torch 2.9.0+cu128 / torchvision 0.24.0+cu128 / transformers 4.57.3.' \
      'See docs/sensor_dependencies.md to plan a different platform build.' >&2
    exit 2
  fi
  if ! command -v uv >/dev/null 2>&1; then
    printf 'uv executable is required for --install.\n' >&2
    exit 1
  fi

  # Do not silently mix an older CPU spconv or another CUDA variant with cu126.
  "$python_bin" - <<'PY'
from importlib.metadata import distributions

names = {dist.metadata['Name'].lower().replace('_', '-') for dist in distributions() if dist.metadata.get('Name')}
conflicts = sorted(name for name in names if
                   (name == 'spconv' or name.startswith('spconv-cu')) and name != 'spconv-cu126'
                   or (name == 'cumm' or name.startswith('cumm-cu')) and name != 'cumm-cu126')
if conflicts:
    raise SystemExit('Conflicting sparse packages found: ' + ', '.join(conflicts) +
                     '. Review them manually before installation; nothing was removed.')
PY

  printf '%s\n' 'Installing pinned torch-scatter wheel (no Torch upgrades)...'
  uv pip install --python "$python_bin" --no-deps --only-binary=:all: --no-index \
    --find-links 'https://data.pyg.org/whl/torch-2.9.0+cu128.html' \
    'torch-scatter==2.1.2+pt29cu128'

  printf '%s\n' 'Installing reviewed spconv/cu126 candidates (no dependency resolution)...'
  uv pip install --python "$python_bin" --no-deps --only-binary=:all: \
    'spconv-cu126==2.3.8' 'cumm-cu126==0.7.11'

  # Official Concerto setup.py still imports pkg_resources, removed in setuptools 82+.
  printf '%s\n' 'Preparing legacy Concerto build tools...'
  uv pip install --python "$python_bin" 'setuptools==80.9.0' wheel
  "$python_bin" -c 'import pkg_resources'
  printf '%s\n' 'Installing official Pointcept/Concerto at its reviewed revision...'
  uv pip install --python "$python_bin" --no-deps --no-build-isolation \
    'git+https://github.com/Pointcept/Concerto.git@5b41117160cdd92c3538be1911277b17e7ef5871'
fi

printf '%s\n' 'Checking optional sensors, trainctl, and WAH Hydra preflight...'
NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$python_bin" - <<'PY'
from importlib.metadata import version
from pathlib import Path

import torch
import spconv.pytorch
import torch_scatter
import concerto
import concerto.model
import trainctl
import tomli_w
from hydra import compose, initialize_config_dir

from gr00t.configs.hydra_config import build_gr00t_config
from gr00t.utils.sensor_dependency_preflight import check_sensor_dependencies

if not callable(getattr(concerto, 'load', None)) or not callable(getattr(concerto.model, 'PointTransformerV3', None)):
    raise RuntimeError('Installed concerto is not the official Pointcept API')

with initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base='1.3'):
    cfg = compose(config_name='experiment/univtac_full', overrides=['model=gr00t_sidecar'])
check_sensor_dependencies(build_gr00t_config(cfg).model)
for package in ('spconv-cu126', 'cumm-cu126', 'torch-scatter', 'concerto', 'trainctl'):
    print(f'{package}: {version(package)}')
print('PASS: WAH optional sensor import preflight (no checkpoint/GPU kernels loaded)')
PY

if [[ "$gpu_smoke" == true ]]; then
  printf '%s\n' 'Running optional single-GPU spconv/torch-scatter backward smoke test...'
  "$python_bin" - <<'PY'
import torch
import spconv.pytorch as spconv
from torch_scatter import scatter_add

if not torch.cuda.is_available():
    raise RuntimeError('--gpu-smoke requires CUDA_VISIBLE_DEVICES to include an available GPU')
device = 'cuda:0'
values = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
indices = torch.tensor([0, 0], device=device)
summed = scatter_add(values, indices, dim=0, dim_size=1)
assert summed.shape == (1,) and torch.allclose(summed, torch.tensor([3.0], device=device))
summed.sum().backward()
assert values.grad is not None and torch.isfinite(values.grad).all()

features = torch.randn(2, 4, device=device, requires_grad=True)
coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int32, device=device)
sparse = spconv.SparseConvTensor(features, coordinates, spatial_shape=[2, 2, 2], batch_size=1)
layer = spconv.SubMConv3d(4, 4, kernel_size=1, bias=False).to(device)
output = layer(sparse).features
loss = output.square().sum()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(output).all() and features.grad is not None and torch.isfinite(features.grad).all()
print('PASS: one-GPU torch-scatter and spconv forward/backward kernels')
PY
fi

printf '%s\n' 'All requested WAH sensor checks passed.'
