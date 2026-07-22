# TEARS Project Status and Progress Report

**Report date:** July 22, 2026  
**Repository:** `tears_project_final`  
**Branch:** `main`  
**Current commit:** `0d8a4bc` (`Initial commit`)  
**Commit date:** July 20, 2026  
**Project stage:** Reproducible quality baseline implemented; integration and production validation remain

## Executive summary

The project now has a reproducible MovieLens 1M evaluation pipeline and a
deterministic, quality-improved recommendation service. The accepted model is
EASE with L2 regularization set to 500, supplemented online by explicit genre
controls and seen/liked-title exclusion. It substantially outperforms the
legacy random-projection ranker and the other measured deterministic baselines.

The learned T5 text-alignment experiment and genre-only MMR reranking were
completed but were not enabled because validation selected zero weight for both
additions. The next phase should focus on end-to-end integration, automated
testing, artifact/version management, production validation, and a better
text-to-collaborative alignment experiment.

## Repository and commit status

- `main` and `origin/main` point to commit `0d8a4bc`.
- The repository contains one squashed initial commit; there is no finer-grained
  committed history from which to attribute individual implementation stages.
- Before this report was created, the Git working tree was clean.
- The committed project includes the legacy application, research code, the
  improved recommender/API, evaluation scripts, tests, Slurm jobs, and recorded
  experiment results.
- This report is the current uncommitted documentation addition unless it is
  subsequently committed.

## Current system state

### Accepted recommendation path

The quality path is served by `quality_api.py` and implemented in
`tears_recommender/quality_recommender.py`.

1. MovieLens 1M ratings and the aligned 3,706-item catalog are loaded.
2. Ratings of at least 4.0 are converted to positive interactions.
3. An EASE model is fitted with L2 regularization of 500.
4. Liked titles are used as recommendation evidence and excluded from output.
5. Explicitly excluded titles and disliked genres are removed.
6. A bounded genre/content signal can respond to the supplied description.
7. The service returns ranked titles with component-level scoring information.

The original FastAPI application in `api.py` remains available as the legacy
comparison path. Its runtime-created, untrained 512-to-200 projection is not an
accepted recommendation model.

### Available quality API endpoints

- `GET /` — service/model status.
- `POST /recommend` — quality recommendations.
- `POST /summarize` — OpenAI summary when configured, with a deterministic
  local fallback when no API key is present.
- `POST /summarize_genres` — compatibility summary for genre input.
- `POST /gers` — genre-driven recommendations through the improved ranker.
- `POST /summary_from_ml1m` — summary constructed from MovieLens movie IDs.

### Evaluation protocol

- Dataset: MovieLens 1M aligned to 3,706 saved catalog items.
- Eligible users: 6,024 users with at least five positive interactions.
- Positive threshold: rating greater than or equal to 4.0.
- Split: latest positive item held out per user, followed by a deterministic
  20/80 validation/test user split using seed 2026.
- Test users: 4,820.
- Input interactions: 556,430.
- Ranking cutoff: 10.
- Seen input movies are excluded.
- Dataset SHA-256:
  `506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f`.

## Recorded experiment results

| Model | Recall@10 | NDCG@10 | MRR@10 | Coverage | Genre diversity | Decision |
|---|---:|---:|---:|---:|---:|---|
| Popularity | 0.0411 | 0.0201 | 0.0138 | 0.0302 | 0.7787 | Baseline only |
| Genre content | 0.0100 | 0.0044 | 0.0028 | 0.0753 | 0.6803 | Rejected as default |
| Item-item CF | 0.0598 | 0.0298 | 0.0208 | 0.0872 | 0.7758 | Baseline only |
| Initial hybrid | 0.0581 | 0.0288 | 0.0200 | 0.0866 | 0.7695 | Rejected as default |
| **EASE, L2=500** | **0.1046** | **0.0510** | **0.0351** | **0.2142** | 0.7512 | **Accepted** |
| EASE + MMR | 0.1046 | 0.0510 | 0.0351 | 0.2142 | 0.7512 | Disabled; selected penalty 0 |
| Learned T5 text head | 0.0461 | 0.0227 | 0.0157 | 0.0262 | 0.7784 | Experimental only |
| EASE + learned text | 0.1046 | 0.0510 | 0.0351 | 0.2142 | 0.7512 | Disabled; selected text weight 0 |

EASE improves Recall@10 by 75.0% and NDCG@10 by 71.1% relative to item-item
collaborative filtering. Relative to popularity, it improves NDCG@10 by 154.3%
and more than doubles Recall@10. Genre diversity is approximately 3.2% lower
than item-item CF, so diversity remains a guardrail for future work.

## Progress completed

- Audited and documented the original online recommendation pipeline.
- Identified the random untrained projection, genre delimiter mismatch,
  liked-item leakage, summary lookup mismatch, missing validation, hardcoded
  paths, and incomplete dependency metadata in the legacy path.
- Added a fixed chronological leave-one-out evaluation protocol.
- Added relevance, coverage, popularity, diversity, determinism, and latency
  measurements.
- Implemented popularity, genre/content, item-item CF, hybrid, EASE, and MMR
  evaluation paths.
- Implemented the accepted EASE-based online recommender.
- Added hard disliked-genre filtering and seen/liked-title exclusion.
- Added request bounds for `top_k` and `alpha` in the quality API.
- Added compatibility endpoints for the existing TEARS/GERS interfaces.
- Added local scripts for evaluation, smoke recommendations, and text training.
- Added Slurm jobs and retained their output/error records.
- Completed a learned T5 text-alignment experiment without target leakage.
- Added data-contract and recommendation-behavior tests.
- Updated the website launcher documentation to use the quality API.

## Experiment and job report

- Full MMR evaluation: Slurm job `10137086`; completed successfully on
  `cn-h002` in 20 seconds with exit code 0.
- Initial text-alignment run: Slurm job `10137101`.
- Corrected text-alignment run: Slurm job `10137195`; completed successfully on
  a Tesla V100 in 44 seconds with exit code 0 and approximately 2,002 MiB peak
  GPU allocation. The training/evaluation workload took 19.2 seconds.
- Both text runs selected a blend weight of zero, so the learned text head is
  retained only as an experimental artifact.

## Verification status on July 22

- Git history, committed documentation, JSON result records, and Slurm reports
  were reviewed.
- The recorded metrics are internally consistent between `results/README.md`
  and the project documentation.
- A local test run was attempted with `.venv/bin/python -m pytest -q`.
- The test run could not start because `pytest` is not installed in the current
  virtual environment (`No module named pytest`). This is an environment gap,
  not a passing or failing test result.
- `pytest` is also absent from the declared project dependencies/dev extras.

## Known gaps and risks

1. **No current green test run.** Tests exist, but the active environment
   cannot execute them until test dependencies are declared and installed.
2. **Model fitting at API startup.** EASE is rebuilt from ratings during service
   startup instead of loading a checksummed, versioned fitted artifact.
3. **Broad CORS policy.** The quality API currently allows all origins, methods,
   and headers; production deployment should restrict this.
4. **Text signal is weak.** The learned text head underperformed EASE, and the
   accepted service relies primarily on interaction history plus bounded genre
   controls.
5. **Diversity work is incomplete.** Genre-only MMR did not improve the measured
   tradeoff; franchise and semantic redundancy are not yet handled.
6. **Legacy duplication remains.** Several older applications and research
   implementations coexist with the accepted path, increasing maintenance and
   onboarding cost.
7. **Deployment validation is incomplete.** There is no recorded container,
   CI pipeline, startup benchmark, load test, or production deployment check.
8. **Online quality is unmeasured.** Offline metrics are available, but there is
   no user feedback, click/save, satisfaction, or hide/dislike telemetry.
9. **Single squashed commit.** The current history makes code review, rollback,
   and progress attribution harder than a sequence of focused commits.
10. **Project metadata needs cleanup.** The package description remains the
    placeholder `Add your description here`, and development dependencies are
    not separated from runtime dependencies.

## Prioritized TODOs

### P0 — make the current baseline reliably reproducible

- [ ] Add a development dependency group containing `pytest` and any test-only
  packages.
- [ ] Install the locked development environment and run the complete test
  suite; record the command, platform, and result.
- [ ] Add CI checks for tests, Python parsing/linting, and evaluator smoke runs.
- [ ] Add an API integration test covering startup, health, `/recommend`, genre
  exclusion, liked-title exclusion, validation errors, and fallback summaries.
- [ ] Confirm the Gradio/website flow against the quality API end to end.
- [ ] Save and load the fitted EASE artifact with a configuration version,
  dataset fingerprint, dimensions, and checksum.

### P1 — production hardening

- [ ] Benchmark startup time, recommendation latency, memory, and concurrent
  request behavior on the intended deployment host.
- [ ] Restrict CORS through environment-specific configuration.
- [ ] Add structured logs, request IDs, health/readiness endpoints, and safe
  error reporting.
- [ ] Define deployment configuration for API URL, OpenAI model, secrets,
  artifact location, and allowed origins.
- [ ] Validate cold-start, sparse-history, typo, empty-context, unknown-title,
  and conflicting-feedback cases.
- [ ] Document artifact regeneration and rollback procedures.

### P2 — recommendation quality

- [ ] Establish a separate tuning protocol for genre/content weights and keep
  the final test set read-once.
- [ ] Investigate learned text alignment with stronger objectives, hard
  negatives, popularity-aware sampling, and saved checkpoints.
- [ ] Evaluate semantic or franchise-aware reranking instead of genre-only MMR.
- [ ] Measure relevance and diversity by user-history size, popularity segment,
  and genre cohort to expose uneven performance.
- [ ] Add score calibration and explanations suitable for the UI.
- [ ] Define and collect online success signals before changing the default
  model based only on qualitative examples.

### P3 — repository maintainability

- [ ] Clearly label active, legacy, and research-only directories in the root
  README.
- [ ] Remove or archive duplicate and obsolete implementations only after their
  dependencies and usage have been verified.
- [ ] Replace placeholder package metadata and document supported Python
  versions.
- [ ] Use focused commits for subsequent work and include test/result evidence
  in commit or pull-request descriptions.
- [ ] Add a dated report index under `docs/` as additional progress reports are
  created.

## Recommended next milestone

The next milestone should be **a fully reproducible, CI-verified quality API**.
It is complete when a fresh environment can install runtime and development
dependencies, load a versioned EASE artifact, pass unit and API integration
tests, run an evaluator smoke test, and serve the existing website without
manual code changes. This milestone secures the accepted quality gain before
further model experimentation.

## Source records reviewed

- `results/README.md`
- `results/evaluation_users6024_seed2026.json`
- `results/text_alignment_seed2026.json`
- `results/slurm-evaluate-10137086.err`
- `results/slurm-text-10137195.err`
- `docs/CURRENT_RECOMMENDATION_PIPELINE.md`
- `docs/BASELINE_RUNBOOK.md`
- `docs/RECOMMENDATION_QUALITY_PLAN.md`
- `quality_api.py`
- `tears_recommender/quality_recommender.py`
- `tests/test_data_contract.py`
- `tests/test_quality_recommender.py`
- `pyproject.toml`

