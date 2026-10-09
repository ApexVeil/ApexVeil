#!/usr/bin/env bash
set -euo pipefail
apex_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$apex_root"
conda_cmd="${CONDA_EXE:-conda}"
if ! command -v "$conda_cmd" >/dev/null 2>&1; then
  echo "Install Conda and make conda available before running setup." >&2
  exit 1
fi
git submodule update --init --recursive
if ! "$conda_cmd" run -n apexveil python -c 'import sys; assert sys.version_info[:2] == (3, 12)' >/dev/null 2>&1; then
  "$conda_cmd" env create -f environment.yml
fi
"$conda_cmd" run --no-capture-output -n apexveil python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
"$conda_cmd" run --no-capture-output -n apexveil python -m pip install -r requirements/runtime.txt
"$conda_cmd" run --no-capture-output -n apexveil python -m pip install --no-deps -e third_party/sam3
"$conda_cmd" run --no-capture-output -n apexveil python -m pip install -e '.[dev]'
"$conda_cmd" run --no-capture-output -n apexveil apexveil doctor
echo "Environment ready. Authenticate with Hugging Face, then run: conda run -n apexveil apexveil models download"
