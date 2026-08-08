#!/usr/bin/env bash
set -u

ROOT=/home/mila/a/adls/tears_project_final
SESSION=tears_public
FRONTEND_LOG=/tmp/tears_frontend_current.log
BACKEND_LOG=/tmp/tears_backend_current.log
TAILSCALE_LOG=/tmp/tears_tailscaled.log

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

restart_frontend() {
  log "frontend is unavailable; restarting"
  tmux respawn-window -k -t "$SESSION:frontend" \
    "cd '$ROOT/movie-recommender' && export REACT_APP_QUALITY_API_URL=https://tearsgersab.tail9d6ed0.ts.net:8443 && npm start 2>&1 | tee -a '$FRONTEND_LOG'"
}

restart_backend() {
  log "backend is unavailable; restarting"
  tmux respawn-window -k -t "$SESSION:backend" \
    "cd '$ROOT' && /home/mila/a/adls/.conda/envs/tears_env/bin/uvicorn api:app --host 0.0.0.0 --port 8001 2>&1 | tee -a '$BACKEND_LOG'"
}

restart_tailscale() {
  log "Tailscale daemon is unavailable; restarting"
  tmux respawn-window -k -t "$SESSION:tailscaled" \
    "'$ROOT/.tools/tailscaled' --tun=userspace-networking --state='$ROOT/.tailscale/state' --statedir='$ROOT/.tailscale' --socket='$ROOT/.tailscale/tailscaled.sock' 2>&1 | tee -a '$TAILSCALE_LOG'"
  sleep 5
  "$ROOT/.tools/tailscale" --socket="$ROOT/.tailscale/tailscaled.sock" funnel --bg --yes 3000 || true
  "$ROOT/.tools/tailscale" --socket="$ROOT/.tailscale/tailscaled.sock" funnel --bg --yes --https=8443 8001 || true
}

log "public-service watchdog started"
while true; do
  curl -fsS --max-time 5 http://127.0.0.1:8001/ >/dev/null || restart_backend
  curl -fsS --max-time 5 http://127.0.0.1:3000/ >/dev/null || restart_frontend
  "$ROOT/.tools/tailscale" --socket="$ROOT/.tailscale/tailscaled.sock" status >/dev/null 2>&1 || restart_tailscale
  sleep 30
done
