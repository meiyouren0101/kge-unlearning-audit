#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ "${1:-}" == all ]] || { echo "Use bash launch.sh all [arguments]"; exit 2; }
shift
mkdir -p launch_logs
log=$(mktemp launch_logs/all_XXXXXXXX.log)
nohup setsid "${GPU_PYTHON:-$PWD/.venv/bin/python}" -I -B -u job_schema.py all "$@" >"$log" 2>&1 < /dev/null &
printf 'PID=%s\nLOG=%s\n' "$!" "$PWD/$log"
