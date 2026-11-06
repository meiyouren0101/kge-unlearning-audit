#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
py=${GPU_PYTHON:-$PWD/.venv/bin/python}
"$py" -I -B job_schema.py verify
"$py" -I -B job_schema.py check-env
"$py" -B -m unittest -v test_schema_adapter test_schema_job test_scheduler20
if [[ $# -gt 0 ]]; then "$py" -I -B job_schema.py check-data "$@"; fi
