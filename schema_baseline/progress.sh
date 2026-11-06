#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${GPU_PYTHON:-$PWD/.venv/bin/python}" -I -B job_schema.py progress
