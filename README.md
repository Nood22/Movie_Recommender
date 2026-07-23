# TEARS Movie Recommendation Project

This repository contains a legacy TEARS-inspired text recommender, research
models, and a reproducible quality-improvement pipeline for MovieLens 1M.

Start here:

- [`docs/CURRENT_RECOMMENDATION_PIPELINE.md`](docs/CURRENT_RECOMMENDATION_PIPELINE.md)
  explains the original request and ranking flow.
- [`docs/BASELINE_RUNBOOK.md`](docs/BASELINE_RUNBOOK.md) contains end-to-end
  commands for the legacy and improved services and the offline evaluator.
- [`docs/RECOMMENDATION_QUALITY_PLAN.md`](docs/RECOMMENDATION_QUALITY_PLAN.md)
  defines metrics, experiment order, quality gates, and the longer-term plan.
- [`results/README.md`](results/README.md) records measured results and accepted
  or rejected experiments.

The current accepted model is EASE with explicit genre control. It is exposed
by `quality_api.py` and evaluated by `scripts/evaluate_recommenders.py`.

```bash
. .venv/bin/activate
uvicorn quality_api:app --host 127.0.0.1 --port 8001
```

The original FastAPI baseline remains in `api.py` for controlled comparison.

## Run the website

Start the quality-improved API from the repository root:

```bash
. .venv/bin/activate
uvicorn quality_api:app --host 127.0.0.1 --port 8001
```

In a second terminal, start the Gradio website:

```bash
. .venv/bin/activate
python TEARS_Project/Tears_space/tears_ui.py
```

The website uses `http://127.0.0.1:8001/recommend` by default. When the API is
hosted elsewhere, set its complete recommendation endpoint before starting the
website:

```bash
export TEARS_RECOMMENDER_API_URL="https://your-api-host/recommend"
python TEARS_Project/Tears_space/tears_ui.py
```
