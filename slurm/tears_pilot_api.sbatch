#!/usr/bin/env bash
# Run only the GPU-backed pilot API. The public UI/tunnel live in the CPU job.
#SBATCH --job-name=tears_pilot_api
#SBATCH --partition=long
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/mila/a/adls/tears_project_final/logs/tears-pilot-api-%j.out
#SBATCH --error=/home/mila/a/adls/tears_project_final/logs/tears-pilot-api-%j.err

set -euo pipefail

PROJECT_ROOT="/home/mila/a/adls/tears_project_final"
PILOT_UVICORN="$PROJECT_ROOT/.venv/bin/uvicorn"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

api_pid=""

cleanup() {
  trap - EXIT INT TERM USR1
  if [[ -n "$api_pid" ]] && kill -0 "$api_pid" 2>/dev/null; then
    kill "$api_pid" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}

requeue_before_timeout() {
  printf 'requeueing pilot API job %s before its time limit\n' "$SLURM_JOB_ID"
  cleanup
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}

trap cleanup EXIT INT TERM
trap requeue_before_timeout USR1

bash "$PROJECT_ROOT/scripts/run_pilot_api.sh" &
api_pid=$!
wait "$api_pid"
