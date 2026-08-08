# ML-32M Implementation and Submission Progress

**Updated:** 2026-08-06 20:43 EDT  
**Repository:** `tears_project_final`  
**Training phase:** Small-dataset smoke phase  
**Tracking:** W&B entity `niita-mila`, project `tears-ml32m`

## Completed implementation

The approved ML-32M scaling plan and operating runbook are stored alongside
this report in `ML32M_TEARS_GERS_SCALING_PLAN.md` and
`ML32M_TRAINING_RUNBOOK.md`.

> Historical report: the pending states below describe the submission-time
> snapshot on August 6. All 61 jobs later completed successfully. The audited
> final status and rung-20 promotion are in
> [`../august7/AUGUST_7_TRAINING_STATUS.md`](../august7/AUGUST_7_TRAINING_STATUS.md).

The new `tears_training` package implements:

- Typed configuration derived from `SCRATCH`, with no hardcoded credential or
  user-specific cluster path.
- Atomic JSON/checkpoint writes, Git and lockfile provenance, stable
  fingerprints, immutable paid-artifact mirroring, and checkpoint hashes.
- ML-32M checksum/schema verification, deterministic activity-stratified user
  splits, chronological 70/30 validation/test histories, training-only item
  support, contiguous item mappings, sparse CSR files, and memory-mappable CSR
  components.
- RecVAE with its residual encoder and composite prior, TEARS Base, GERS Base,
  TEARS-RecVAE, and GERS-RecVAE with paper alpha semantics.
- Same-seed RecVAE initialization for hybrids, a frozen RecVAE encoder, shared
  trainable decoder, OT/KL losses, BF16, DDP via `torchrun`, validation-based
  checkpoint selection, best/final/resume checkpoints, and pre-timeout graceful
  stopping.
- Ranking evaluation with observed-item masking, Recall/NDCG at 20/50,
  eligibility denominators, hybrid alpha sweeps, and explicit test-set
  isolation.
- Structured OpenAI Batch summary planning, prompt/model/history cache keys,
  privacy validation, duplicate detection, two-retry limit, one-shot polling,
  token/cost accounting, a hard spend ceiling with reserve, and explicit paid
  submission confirmation.
- Successive-halving grids and selection, one/two/four-GPU Slurm profiles,
  concurrency caps, deterministic W&B IDs, manual full-phase promotion gates,
  resource measurements, seed aggregation, and paired statistical tests.

## Verification completed before submission

- Repository test suite: **29 passed**.
- Python package compilation: passed.
- `uv.lock` consistency check: passed; 105 packages resolved.
- ML-32M `ratings.csv`, `movies.csv`, `links.csv`, and `tags.csv`: checksum and
  schema verification passed.
- Workspace secret scan: no findings.
- Smoke RecVAE rung-5 orchestration dry run: 54 planned jobs.
- Dry-run external side effects: zero paid API requests and zero Slurm jobs.
- W&B client version: 0.28.1; configured credentials are available through the
  existing user login without embedding a key in repository files.

## Small-dataset phase specification

The smoke dataset uses the deterministic 10,000-user pilot cohort and the top
4,000 training-supported items with split seed 2024. The first successive-
halving rung trains for five epochs. RecVAE candidates span:

- Dropout: 0.1, 0.2, 0.4.
- Learning rate: 1e-3, 1e-4, 1e-5.
- Gamma: 0.0035, 0.004, 0.005.
- Weight decay: 0 and 1e-5.

This produces 54 RecVAE jobs. Six independently runnable GERS Base jobs were
also submitted after the RecVAE lanes. TEARS Base waits for validated summaries,
and hybrids wait for the selected same-seed RecVAE checkpoint.

## Submission ledger

The complete 61-job ledger, including every Slurm ID, hyperparameter
combination, dependency, and initial scheduler state, is recorded in
[`SMOKE_RUNG1_SLURM_REPORT.md`](SMOKE_RUNG1_SLURM_REPORT.md).

- Smoke data preparation: Slurm `10306324`.
- RecVAE rung-1 grid: Slurm `10306331`–`10306384` (54 jobs).
- GERS Base rung-1 grid: Slurm `10306389`–`10306394` (6 jobs).
- Initial state: preparation pending resources; training jobs pending their
  declared dependencies.
- Current status at 20:40 EDT: unchanged and healthy—`10306324` is pending
  solely for resources, all 60 training jobs are pending their intended
  dependencies, and Slurm reports zero failures/cancellations/timeouts.
- Current estimated preparation start: **2026-08-06 22:53:06 EDT** on
  `cn-h001`; this estimate is not a guarantee.
- Generated smoke matrices/logs/checkpoints/per-training W&B runs: **none yet,
  as expected before preparation begins**.
- Verified W&B training runs: **0**. The only smoke-related W&B runs are the
  two terminated submission-manifest runs `submit-7a35d3848c143214` and
  `submit-9044407bc05070ed`; neither represents model training. They were
  intentionally ended after artifact upload, with the combined run ending at
  2026-08-06 20:03:13 EDT. Their W&B artifacts remain available.
- Combined W&B artifact:
  [`niita-mila/tears-ml32m/smoke-rung1-submissions-7a35d3848c14:v0`](https://wandb.ai/niita-mila/tears-ml32m/artifacts/slurm-submission/smoke-rung1-submissions-7a35d3848c14/v0).
- W&B submission run:
  [`submit-7a35d3848c143214`](https://wandb.ai/niita-mila/tears-ml32m/runs/submit-7a35d3848c143214).

## Safety status

- Paid summary request submitted: **No**.
- Full five-seed training promoted: **No**.
- Test set inspected: **No**.
- Slurm jobs submitted: **Yes — one preparation job and 60 rung-1 training
  jobs**.
- TEARS Base submitted: **No — validated paid summaries are not yet present**.
- Hybrid jobs submitted: **No — they require the selected seed-2024 RecVAE
  checkpoint**.
