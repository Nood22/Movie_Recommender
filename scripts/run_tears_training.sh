#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${SCRATCH:-}" ]]; then
  echo "SCRATCH is required" >&2
  exit 2
fi

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
NPROC="${SLURM_GPUS_ON_NODE:-1}"

exec "$PROJECT_ROOT/.venv/bin/torchrun" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="$NPROC" \
  -m tears_training.train "$@"
