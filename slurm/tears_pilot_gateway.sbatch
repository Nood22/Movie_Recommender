#!/usr/bin/env bash
# Keep the public pilot UI and Tailscale Funnel independent of the GPU model job.
#SBATCH --job-name=tears_pilot_gateway
#SBATCH --partition=long-cpu
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/mila/a/adls/tears_project_final/logs/tears-pilot-gateway-%j.out
#SBATCH --error=/home/mila/a/adls/tears_project_final/logs/tears-pilot-gateway-%j.err

set -euo pipefail

PROJECT_ROOT="/home/mila/a/adls/tears_project_final"
PILOT_BUILD_ROOT="$PROJECT_ROOT/movie-recommender-pilot/build"
TAILSCALE="$PROJECT_ROOT/.tools/tailscale"
TAILSCALED="$PROJECT_ROOT/.tools/tailscaled"
TAILSCALE_DIR="$PROJECT_ROOT/.tailscale"
TAILSCALE_SOCKET="$TAILSCALE_DIR/pilot-gateway.sock"
PUBLIC_HOST="tearsgersab.tail9d6ed0.ts.net"

mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

gateway_pid=""
tailscaled_pid=""

cleanup() {
  trap - EXIT INT TERM USR1
  for pid in "$gateway_pid" "$tailscaled_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}

requeue_before_timeout() {
  printf 'requeueing gateway job %s before its time limit\n' "$SLURM_JOB_ID"
  cleanup
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}

trap cleanup EXIT INT TERM
trap requeue_before_timeout USR1

if ! "$TAILSCALE" --socket="$TAILSCALE_SOCKET" status >/dev/null 2>&1; then
  rm -f "$TAILSCALE_SOCKET"
fi

"$TAILSCALED" \
  --tun=userspace-networking \
  --state="$TAILSCALE_DIR/state" \
  --statedir="$TAILSCALE_DIR" \
  --socket="$TAILSCALE_SOCKET" \
  >"$PROJECT_ROOT/logs/tailscaled-gateway-${SLURM_JOB_ID}.log" 2>&1 &
tailscaled_pid=$!

for _ in {1..60}; do
  if "$TAILSCALE" --socket="$TAILSCALE_SOCKET" status >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$tailscaled_pid" 2>/dev/null; then
    printf 'tailscaled exited during startup\n' >&2
    exit 1
  fi
  sleep 1
done
"$TAILSCALE" --socket="$TAILSCALE_SOCKET" status >/dev/null

"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/serve_pilot_gateway.py" \
  --build-root "$PILOT_BUILD_ROOT" \
  --host 127.0.0.1 \
  --port 3010 \
  --pilot-job-name tears_pilot_api \
  >"$PROJECT_ROOT/logs/pilot-gateway-${SLURM_JOB_ID}.log" 2>&1 &
gateway_pid=$!

for _ in {1..60}; do
  if curl -fsS --max-time 5 http://127.0.0.1:3010/_gateway/health >/dev/null; then
    break
  fi
  if ! kill -0 "$gateway_pid" 2>/dev/null; then
    printf 'pilot gateway exited during startup\n' >&2
    exit 1
  fi
  sleep 1
done
curl -fsS --max-time 5 http://127.0.0.1:3010/_gateway/health >/dev/null

# Serve the pilot gateway at the public root as well as the legacy /pilot
# mount. The old root target (the retired React dev server on port 3000) is
# persistent Tailscale state, so it must be replaced on every job restart.
"$TAILSCALE" --socket="$TAILSCALE_SOCKET" funnel --bg --yes 3010
"$TAILSCALE" --socket="$TAILSCALE_SOCKET" funnel --bg --yes --set-path=/pilot 3010
"$TAILSCALE" --socket="$TAILSCALE_SOCKET" funnel status
printf 'pilot gateway is ready on job %s at https://%s/pilot/\n' \
  "$SLURM_JOB_ID" "$PUBLIC_HOST"

while true; do
  for service in "tailscaled:$tailscaled_pid" "pilot_gateway:$gateway_pid"; do
    label="${service%%:*}"
    pid="${service##*:}"
    if ! kill -0 "$pid" 2>/dev/null; then
      printf '%s process %s exited\n' "$label" "$pid" >&2
      exit 1
    fi
  done
  sleep 30
done
