#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=
py=${1:-${GPU_PYTHON:-python3}}
fb=${2:-../fb15k-237-20.zip}
wn=${3:-../wn18rr-20.zip}
"$py" -B job20.py verify
"$py" -B job20.py verify-data --fb-zip "$fb" --wn-zip "$wn"
"$py" -B -m unittest discover -s engine_tests -v
"$py" -B -m unittest -v test_matrix20 test_scheduler20 test_seed_chains test_safe_stop
