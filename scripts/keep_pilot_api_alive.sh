#!/usr/bin/env bash
# Compatibility watchdog for Slurm jobs submitted before pilot serving existed.

set -u

PROJECT_ROOT="/home/mila/a/adls/tears_project_final"
PILOT_PORT=8010
JOB_NAME="tears_public"
LOG_PATH="$PROJECT_ROOT/logs/pilot-api-watchdog.log"
PUBLIC_HEALTH_URL="https://tearsgersab.tail9d6ed0.ts.net/pilot/api/health"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG_PATH"
}

running_job_id() {
  local job_id
  while IFS= read -r job_id; do
    if [[ -n "$job_id" ]]; then
      printf '%s\n' "$job_id"
      return 0
    fi
  done < <(squeue --noheader --name "$JOB_NAME" --states=RUNNING --format=%A)
  return 1
}

mkdir -p "$PROJECT_ROOT/logs"
log "pilot API watchdog started"

while true; do
  job_id="$(running_job_id || true)"
  if [[ -z "$job_id" ]]; then
    sleep 15
    continue
  fi

  if curl -fsS --max-time 10 "$PUBLIC_HEALTH_URL" >/dev/null; then
    sleep 15
    continue
  fi

  log "pilot API is unavailable; starting it in Slurm job $job_id"
  srun \
    --jobid="$job_id" \
    --overlap \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=12G \
    --gres=gpu:1 \
    bash -lc \
    "cd '$PROJECT_ROOT' && exec .venv/bin/uvicorn pilot_api:app --host 127.0.0.1 --port '$PILOT_PORT'" \
    >>"$LOG_PATH" 2>&1
  log "pilot API step exited; retrying after allocation settles"
  sleep 15
done
