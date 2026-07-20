# Baseline Recommendation Runbook

## Baseline definition

The baseline is the current root-level path:

1. Convert user feedback to a taste summary with `POST /summarize`.
2. Encode summary and optional context with T5-small.
3. Project the 512-D text representation to the 200-D movie space.
4. Rank all 3,706 movies by similarity.
5. Apply the current genre filter and liked-title boost.

This baseline must be measured before changing its ranking behavior. Because
the projection is randomly initialized, every baseline experiment must record
its random seed. The baseline is expected to expose this instability rather
than hide it.

## Prerequisites

- Python 3.10+
- Access to the `t5-small` Hugging Face model on the first run, or a populated
  local Hugging Face cache
- `OPENAI_API_KEY` only when `/summarize` or `/gers` is exercised
- The artifacts under `model/saved_models/`

The current project environment is incomplete. After dependency metadata is
corrected, install with:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Do not put API keys in repository files. Export them into the shell or inject
them through the job/runtime secret manager:

```bash
export OPENAI_API_KEY='...'
```

## Start the current API

From the repository root:

```bash
. .venv/bin/activate
uvicorn api:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/
```

## Full request: feedback to recommendations

Generate a taste description:

```bash
curl -sS http://127.0.0.1:8000/summarize \
  -H 'Content-Type: application/json' \
  -d '{
    "liked": ["Alien (1979)", "Blade Runner (1982)"],
    "disliked": ["Romance"],
    "context": "Atmospheric, intelligent, and visually distinctive."
  }'
```

Copy the returned summary into the recommendation request:

```bash
curl -sS http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "summary": "Summary: The user enjoys atmospheric science fiction...",
    "context": "Prefer a slower, thoughtful film tonight.",
    "liked": ["Alien (1979)", "Blade Runner (1982)"],
    "disliked": ["Romance"],
    "top_k": 12,
    "alpha": 0.75
  }'
```

## Reproducible offline evaluation

The evaluator added by this project work will use MovieLens 1M positive ratings
and fixed per-user holdouts. OpenAI is not called during evaluation. User input
is constructed only from training-history movies, preventing target leakage.

The fixed protocol is:

- Positive interaction: rating `>= 4.0`.
- Eligible user: at least 5 positive interactions.
- Holdout: the most recent positive interaction by timestamp.
- Input history: all earlier positive interactions.
- Candidate catalog: all 3,706 movies.
- Exclusions: all input-history movies.
- Default cutoff: 10 recommendations.
- Fixed seed: 2026.

Primary metrics:

- `Recall@10`: fraction of held-out items recovered.
- `NDCG@10`: rewards finding the held-out item near the top.
- `MRR@10`: reciprocal rank of the held-out item.

Guardrail metrics:

- catalog coverage;
- mean recommendation popularity;
- intra-list genre diversity;
- repeated-run ranking agreement;
- latency per user.

Commands and exact output paths will be kept here as the evaluator is
implemented. Every result must include the Git state, configuration, seed,
dataset fingerprint, item/user counts, metrics, and wall-clock time.

Run the complete baseline suite:

```bash
. .venv/bin/activate
python scripts/evaluate_recommenders.py --top-k 10 --output-dir results
```

If the full diversity reranking pass exceeds local memory, submit the equivalent
bounded Slurm job:

```bash
sbatch slurm/evaluate_recommenders.sbatch
```

Run a fast smoke evaluation while developing:

```bash
python scripts/evaluate_recommenders.py --max-users 300 --skip-ease \
  --output-dir /tmp/tears-results
```

Run the quality-improved API:

```bash
uvicorn quality_api:app --host 127.0.0.1 --port 8001
```

Then request recommendations directly from a taste description and liked
movies:

```bash
curl -sS http://127.0.0.1:8001/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "description": "Atmospheric science fiction and action",
    "liked": ["Alien (1979)", "Blade Runner (1982)"],
    "disliked_genres": ["Romance"],
    "top_k": 5
  }'
```

## Baseline result record

Status at initial audit:

- Source files parse successfully.
- Catalog alignment is valid: 3,706 titles, genres, and embedding rows.
- The configured `.venv` cannot run the API because key runtime dependencies
  are missing.
- The current random projection remains unsuitable as a scientific baseline;
  reproducible popularity, genre, item-CF, hybrid, and EASE baselines have now
  been measured on the same fixed split.
- The accepted EASE run reaches Recall@10 0.1046 and NDCG@10 0.0510. See
  `results/README.md` for the complete table and interpretation.

Results will be written under `results/` as JSON plus a concise Markdown table.
