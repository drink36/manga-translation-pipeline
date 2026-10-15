#!/usr/bin/env bash
#SBATCH --job-name=manga-translate
#SBATCH --cluster=ascend
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=manga-%j.log

# Explicit submission only. Script-first invocation works with the user's wrapper.
# MANGA_STAGE=all sbatch scripts/ascend.sh --account=YOUR_PROJECT
set -euo pipefail
MANGA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$MANGA_ROOT"
MANGA_CONFIG="${MANGA_CONFIG:-configs/pipeline.yaml}"
MANGA_STAGE="${MANGA_STAGE:-all}"
MANGA_PYTHON="${MANGA_PYTHON:-$MANGA_ROOT/.venv-models/bin/python}"
MANGA_SERVER_PYTHON="${MANGA_SERVER_PYTHON:-$MANGA_ROOT/.venv-vllm/bin/python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

case "$MANGA_STAGE" in
  plan|doctor)
    exec "$MANGA_PYTHON" -m manga_pipeline "$MANGA_STAGE" --config "$MANGA_CONFIG"
    ;;
  all|audit|reconcile|context|translate|qa)
    if [[ "${MANGA_START_SERVER:-1}" == "1" ]]; then
      server_log="logs/qwen-${SLURM_JOB_ID:-local}.log"
      "$MANGA_SERVER_PYTHON" -m manga_pipeline serve --config "$MANGA_CONFIG" >"$server_log" 2>&1 &
      server_pid=$!
      "$MANGA_PYTHON" - "$MANGA_CONFIG" "$server_pid" "$server_log" <<'PY'
import json, os, sys, time, urllib.error, urllib.request
from manga_pipeline.config import load_config
config = load_config(sys.argv[1])
pid = int(sys.argv[2])
url = config['vlm']['endpoint'].rstrip('/') + '/models'
deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit(f'Qwen server exited; inspect {sys.argv[3]}')
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            models = json.load(response)
        if any(x['id'] == config['vlm']['model'] for x in models.get('data', [])):
            break
    except (OSError, ValueError, urllib.error.URLError):
        pass
    time.sleep(2)
else:
    raise SystemExit(f'Qwen server not ready after 900 seconds; inspect {sys.argv[3]}')
PY
    fi
    ;;
  prepare|detect|ocr|render|export) ;;
  *) echo "Unknown MANGA_STAGE: $MANGA_STAGE" >&2; exit 2 ;;
esac
"$MANGA_PYTHON" -m manga_pipeline run --config "$MANGA_CONFIG" --stage "$MANGA_STAGE"
