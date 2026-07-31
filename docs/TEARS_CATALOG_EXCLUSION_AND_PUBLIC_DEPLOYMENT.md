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
- `api_tunnel`: reverse HTTPS tunnel for the API
- `frontend`: React dev server on `0.0.0.0:3000`
- `web_tunnel`: reverse HTTPS tunnel for the website

Inspect the deployment with:

```bash
squeue -j 10258970
srun --jobid=10258970 --overlap --pty bash
tmux attach -t tears_public
```

At validation time, the API returned `{"status":"running","device":"cuda"}`
and both public HTTPS endpoints returned HTTP 200.

The public website URL created for this run was:

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
