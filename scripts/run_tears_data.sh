#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${SCRATCH:-}" ]]; then
  echo "SCRATCH is required" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
exec "$PROJECT_ROOT/.venv/bin/python" -m tears_training.data "$@"
