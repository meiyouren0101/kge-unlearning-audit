#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
base=${1:?Pass the existing torch2.1.1+cu121 Python}
test ! -e .venv || { echo "Own .venv already exists; keep it and run check.sh"; exit 1; }
"$base" -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python -I -B job_schema.py check-env
