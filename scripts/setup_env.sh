#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required (Miniconda or Mambaforge)." >&2
  exit 1
fi

conda env create -f environment.yml --yes || conda env update -f environment.yml --prune
conda run -n survvtg python - <<'PY'
import torch
import torchtext
print('torch:', torch.__version__)
print('torchtext:', torchtext.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
echo "Environment ready. Activate with: conda activate survvtg"
