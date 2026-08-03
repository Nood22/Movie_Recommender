# TEARS catalog eligibility and public deployment

Last validated: 2026-07-31 (America/Toronto)

## Recommendation eligibility contract

The TEARS left panel is an onboarding/profile-elicitation catalog. Its source is
`movie-recommender/src/data/fixed_50_movies_ml1m.json`. The frontend passes the
source through `deduplicateSelectableCatalog` before TMDB enrichment and before
storing it in React state.

The imported file has 50 rows and 49 unique canonical MovieLens IDs. MovieLens
ID 356, `Forrest Gump (1994)`, appeared twice. Deduplication preserves the first
record without merging conflicting fields:

```json
{
  "movieId": 356,
  "title": "Forrest Gump (1994)",
  "genres": ["Drama", "Romance"]
}
```

Numeric MovieLens ID is the primary catalog identity. Canonical title plus exact
release year is used only when an ID is missing. React selection membership and
left-card keys also use the canonical MovieLens ID.

The `/recommend` request keeps preference evidence separate from candidate
eligibility:

```json
{
  "summary": "...generated only from selected/rated movies...",
  "liked_movie_ids": [356],
  "excluded_movie_ids": ["all 49 onboarding MovieLens IDs"],
  "alpha": 0.5,
  "top_k": 12
}
```

- `liked_movie_ids` contains only movies the user selected and rated.
- `excluded_movie_ids` contains every unique onboarding-catalog MovieLens ID.
- Unselected onboarding movies are not positive or negative evidence.
- Backend eligibility removes selected IDs, onboarding IDs, duplicate candidate
  IDs, and the pre-existing same-collection matches.
- The frontend repeats the onboarding-ID and duplicate-ID checks immediately
  before rendering. Exact canonical title/year is a secondary safeguard only.
- Retained candidates preserve backend order, scores, ranks, titles, and
  verified TMDB metadata.

The backend requests `min(100, top_k + exclusion_union_size)` candidates, where
100 is the adapter's existing maximum.

Development diagnostics report selected IDs, onboarding exclusion count,
backend-returned IDs, rendered IDs, and selectable-catalog overlap. Backend logs
remain a single concise line per request.

## Validation

The focused checks cover catalog ID 356 deduplication, preservation of the first
record, similar titles, different-year remakes, onboarding eligibility, duplicate
recommendations, and unchanged same-collection exclusion.

Validated commands:

```bash
.venv/bin/python -m py_compile api.py tears_inference_adapter.py tears_candidate_filter.py
.venv/bin/python -m unittest discover -s tests -v
node --test movie-recommender/tests/selectableCatalog.test.mjs movie-recommender/tests/tmdbMetadata.test.mjs
cd movie-recommender && npm run build
git diff --check
```

Results on 2026-07-31: 11 Python tests passed, both frontend helper suites
passed, the React production build compiled, and `git diff --check` passed.

## Seven-day GPU deployment

The Mila Slurm `long` partition has a seven-day maximum. Deployment created job
`10258970` on `cn-l003` with one GPU, 8 CPUs, and 64 GB RAM:

```bash
sbatch --partition=long --time=7-00:00:00 --gres=gpu:1 \
  --cpus-per-task=8 --mem=64G --job-name=tears_public ...
```

The allocation is kept alive by tmux session `tears_public`. Its windows are:

- `backend`: Conda environment `tears_env`, Uvicorn on `0.0.0.0:8001`
- `frontend`: React dev server on `0.0.0.0:3000`
- `tailscaled`: userspace Tailscale daemon providing the stable public tunnels
- `api_tunnel`: legacy anonymous localhost.run API tunnel
- `web_tunnel`: legacy anonymous localhost.run website tunnel

Inspect the deployment with:

```bash
squeue -j 10258970
srun --jobid=10258970 --overlap --pty bash
tmux attach -t tears_public
```

At validation time, the API returned `{"status":"running","device":"cuda"}`
and both stable public HTTPS endpoints returned HTTP 200.

## Stable free public URL with Tailscale Funnel

The anonymous localhost.run tunnels rotated their hostnames whenever their SSH
connections recovered. On 2026-08-02 they were replaced as the primary public
route by Tailscale Funnel. The free Tailscale Personal plan supplies a stable
HTTPS name without requiring a purchased domain.

The public endpoints are:

```text
Website: https://tearsgersab.tail9d6ed0.ts.net/
API:     https://tearsgersab.tail9d6ed0.ts.net:8443/
```

The website proxies to React on `127.0.0.1:3000`. The API endpoint proxies to
Uvicorn on `127.0.0.1:8001`. React must be started with the stable API base URL:

```bash
REACT_APP_QUALITY_API_URL=https://tearsgersab.tail9d6ed0.ts.net:8443 npm start
```

Tailscale `1.98.10` was installed locally rather than system-wide:

```text
.tools/tailscale
.tools/tailscaled
```

Both `.tools/` and `.tailscale/` are ignored by Git. In particular,
`.tailscale/state` is authentication material and must never be committed or
shared. The one-time Tailscale auth key is also a secret: enter it at runtime,
never place it in a command file, log, environment file, or documentation, and
revoke it after the node has enrolled.

The userspace daemon is run inside the allocation because system-wide/root
installation is not required. Both `--state` and `--statedir` are required;
`--statedir` gives Funnel a persistent location for its TLS certificates.

```bash
ROOT=/home/mila/a/adls/tears_project_final

"$ROOT/.tools/tailscaled" \
  --tun=userspace-networking \
  --state="$ROOT/.tailscale/state" \
  --statedir="$ROOT/.tailscale" \
  --socket="$ROOT/.tailscale/tailscaled.sock"
```

For first-time enrollment, generate a reusable or one-time auth key in the
Tailscale admin console and supply it interactively. Do not substitute the key
directly into shell history:

```bash
read -rs TEARS_TS_AUTH
"$ROOT/.tools/tailscale" --socket="$ROOT/.tailscale/tailscaled.sock" up \
  --auth-key="$TEARS_TS_AUTH" \
  --hostname=tearsgersab \
  --accept-dns=false
unset TEARS_TS_AUTH
```

Funnel must be enabled once in the Tailscale admin console. Configure the two
routes after the daemon is running:

```bash
TS="$ROOT/.tools/tailscale --socket=$ROOT/.tailscale/tailscaled.sock"
$TS funnel --bg --yes 3000
$TS funnel --bg --yes --https=8443 8001
$TS funnel status
```

Check the deployment from outside Mila:

```bash
curl -fsS https://tearsgersab.tail9d6ed0.ts.net:8443/
curl -fsSI https://tearsgersab.tail9d6ed0.ts.net/
```

Expected results are the CUDA status JSON from the API and HTTP 200 from the
website. Tailscale automatically reconnects transient network failures without
changing the hostname. The hostname and node identity persist because the state
directory is on shared storage.

The Funnel address is stable, but it cannot keep the application alive after
Slurm terminates the allocation. The `long` partition still has a seven-day
limit. A replacement job must start the backend, frontend, and `tailscaled`
again using the shared state directory. The public hostname remains unchanged.

## Legacy anonymous localhost.run tunnels

The original public website URL created for this run was:

```text
https://57ff507aec7f63.lhr.life
```

This is an anonymous `localhost.run` tunnel. It is usable while the SSH tunnel
connection remains alive, but its hostname may rotate after a reconnect. A
stable multi-day hostname requires a registered tunnel account or owned domain.
If the hostname rotates, read the latest URLs from the tmux tunnel windows or:

```bash
grep -Eo 'https://[a-z0-9.-]+\.lhr\.life' /tmp/tears_web_tunnel.log | tail -1
grep -Eo 'https://[a-z0-9.-]+\.lhr\.life' /tmp/tears_api_tunnel.log | tail -1
```

When the API URL rotates, restart the frontend window with the current URL in
`REACT_APP_QUALITY_API_URL` so remote browsers do not call an expired endpoint.

Stop the deployment intentionally with:

```bash
scancel 10258970
```
