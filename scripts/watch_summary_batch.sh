#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${1:?run root is required}"
CONFIG="${2:-$PROJECT_ROOT/configs/ml32m.toml}"
SUBMISSION="$RUN_ROOT/submission.json"
PLAN="$RUN_ROOT/request_plan.json"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

while true; do
  "$PROJECT_ROOT/.venv/bin/python" -m tears_training.summaries \
    --config "$CONFIG" poll --submission "$SUBMISSION"
  STATUS="$($PROJECT_ROOT/.venv/bin/python -c \
    "import json; p=json.load(open('$RUN_ROOT/poll.json')); print(','.join(sorted({b['status'] for b in p['batches']})))")"
  case "$STATUS" in
    completed)
      break
      ;;
    failed|expired|cancelled)
      echo "Batch entered terminal non-success state: $STATUS" >&2
      exit 1
      ;;
  esac
  sleep 300
done

"$PROJECT_ROOT/.venv/bin/python" -m tears_training.summaries \
  --config "$CONFIG" validate --plan "$PLAN"
"$PROJECT_ROOT/.venv/bin/python" scripts/report_summary_run.py \
  --run-dir "$RUN_ROOT" --sample-seed 2024 --sample-size 25
