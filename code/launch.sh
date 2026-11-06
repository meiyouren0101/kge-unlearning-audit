#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == check ]]; then shift; exec bash check.sh "$@"; fi
[[ "${1:-}" == all ]] || { echo "Use launch.sh check or launch.sh all [options]"; exit 2; }
shift
mkdir -p launch_logs
log=$(mktemp launch_logs/all_XXXXXXXX.log)
nohup setsid "${GPU_PYTHON:-python3}" -B -u job20.py run "$@" >"$log" 2>&1 < /dev/null &
printf 'PID=%s
LOG=%s
' "$!" "$log"
