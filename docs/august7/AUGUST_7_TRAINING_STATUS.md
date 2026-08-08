# ML-32M Training Status — August 7, 2026

**Audited:** 2026-08-07 (America/Toronto)  
**Plan phase:** Phase 3, quick 10,000-user / 4,000-item successive halving  
**Completed rung:** 60 epochs  
**Active rung:** complete; fixed-200 selected for RecVAE and GERS Base  
**Tracking:** W&B entity `niita-mila`, project `tears-ml32m`

## Pilot cohort amendment and Phase 4 continuation

The current pilot is authorized to proceed with all 9,763 summaries that pass
the existing validator. The matched TEARS/GERS cohort is 8,787 train, 491
validation, and 485 test users. The 237 excluded validation failures comprise
213 train, 9 validation, and 15 test users; their regeneration and validation
is now part of the Phase 6 full-training summary-completion gate.

Phase 4 preparation completed as Slurm job `10312906`. The support-20,
support-50, and support-200 full-user RecVAE jobs (`10312931`–`10312933`) are
running. The scientific-pilot submissions remain dependent on completing the
validation-only catalog selection.

## Documentation reconciliation

The contents of the former case-conflicting `docs/August` and `docs/august`
directories were merged into `docs/auguest6`, as requested. The approved
scaling plan, runbook, August 6 progress report, and both machine-readable
rung-1 submission manifests now have one canonical location.

## Rung-1 completion audit

Slurm accounting was checked for every submitted job rather than inferred from
W&B alone.

| Work | Jobs | Final state | Exit code |
|---|---:|---|---|
| Smoke data preparation | 1 | Completed | `0:0` |
| RecVAE, 5 epochs | 54 | Completed | `0:0` |
| GERS Base, 5 epochs | 6 | Completed | `0:0` |
| **Total** | **61** | **Completed** | **All `0:0`** |

- Scheduler interval: preparation began at 2026-08-06 20:52:12 EDT; the last
  training job ended at 22:25:20 EDT.
- Artifact audit: 60/60 training jobs produced `result.json`, five epochs of
  metrics, and best/final/resume checkpoints.
- Log audit: all 61 Slurm logs are present. A case-insensitive scan found no
  traceback, error, exception, NaN/Inf, OOM, failure, cancellation, or timeout.
- W&B audit: the project contains 60 model-training runs for this rung and two
  submission-manifest runs. All 62 report state `finished`; all 60 training
  runs expose the expected validation metric.
- Test isolation remains intact: no test-set evaluation was performed.
- Paid summary generation remains unperformed.

The normalized rung-1 selection input is preserved in
[`SMOKE_RUNG1_RESULTS.jsonl`](SMOKE_RUNG1_RESULTS.jsonl). It includes the model,
seed, complete hyperparameters, best validation selection NDCG@50, measured
epoch GPU time, W&B run ID, Slurm job ID, and checkpoint directory for every
candidate.

## Rung-1 validation results

| Model | Candidates | Best validation NDCG@50 | Best observed configuration |
|---|---:|---:|---|
| RecVAE | 54 | 0.2152834635 | dropout 0.2, LR 0.001, gamma 0.0035; weight decay 0 and 1e-5 tied |
| GERS Base | 6 | 0.1474217153 | LR 0.001; dropout 0.1, 0.2, and 0.4 tied |

The successive-halving selector retained the best third per model. GPU time
breaks exact validation ties, so the two retained GERS candidates use dropout
0.2 and 0.1 at LR 0.001. No test metric influenced selection.

## Rung-20 promotion and launch

The validated rung-5 results were passed to the existing orchestrator. Its dry
run passed all preflight gates: raw ML-32M data and the smoke matrix exist,
W&B credentials are available, `SCRATCH` is configured, and the workspace
secret scan is clean.

| Model | Promoted jobs | Epochs |
|---|---:|---:|
| RecVAE | 18 | 20 |
| GERS Base | 2 | 20 |
| **Total** | **20** | **20** |

- Slurm IDs: `10308477`–`10308496`.
- Plan fingerprint:
  `c66f9b2c2326afe5c0bdffde0acca8b4c2d1e928ec6f700833a434918c75ec82`.
- Durable submission manifest:
  [`SMOKE_RUNG2_SUBMISSION.json`](SMOKE_RUNG2_SUBMISSION.json).
- Concurrency: two one-A100L jobs; later jobs use `afterany` lane dependencies.
- Initial scheduler state: jobs `10308477` and `10308478` are pending priority;
  the remaining 18 are pending their intentional lane dependencies. There are
  no failed, cancelled, or timed-out rung-20 jobs at this snapshot.

### Rung-20 completion update

All 20 rung-20 jobs subsequently completed with exit `0:0`. Every job produced
20 epochs of metrics and its checkpoint/result artifacts; the log scan remained
clean. The normalized results are in
[`SMOKE_RUNG2_RESULTS.jsonl`](SMOKE_RUNG2_RESULTS.jsonl).

| Model | Completed candidates | Best validation NDCG@50 |
|---|---:|---:|
| RecVAE | 18 | 0.2211611842 |
| GERS Base | 2 | 0.1696215155 |

The best third was promoted immediately: six RecVAE candidates and one GERS
Base candidate ran as rung 60 under Slurm IDs `10308741`–`10308747`.
The submission plan and complete job ledger are stored in
[`SMOKE_RUNG3_SUBMISSION.json`](SMOKE_RUNG3_SUBMISSION.json).

### Rung-60 completion and finalist launch

All seven rung-60 jobs completed with exit `0:0`; the log scan found no
traceback, exception, NaN/Inf, OOM, failure, cancellation, or timeout. Their
normalized validation-only results are in
[`SMOKE_RUNG3_RESULTS.jsonl`](SMOKE_RUNG3_RESULTS.jsonl).

| Model | Completed candidates | Best validation NDCG@50 | Finalist rule |
|---|---:|---:|---|
| RecVAE | 6 | 0.2235864256 | Chose the least-GPU candidate within 1% of best: dropout 0.4, LR 0.001, gamma 0.0035, weight decay 1e-5 |
| GERS Base | 1 | 0.1832019783 | Dropout 0.1, LR 0.001 |

The actual finalist comparison was submitted immediately as four synchronized
jobs. Each model receives both a fixed 200-epoch run and an early-stopped run
capped at 300 epochs (minimum 30, patience 20).

| Slurm ID | Model | Schedule | Initial state |
|---|---|---|---|
| `10308826` | RecVAE | fixed 200 | Pending priority |
| `10308827` | RecVAE | early-stopped, cap 300 | Pending priority |
| `10308828` | GERS Base | fixed 200 | Pending lane dependency |
| `10308829` | GERS Base | early-stopped, cap 300 | Pending lane dependency |

The immutable plan fingerprint is
`7357d58724a68a5ff4e7c9a261bc430415ea69372ab68ee1e7e9d73d5e6a7de5`;
the full ledger is
[`SMOKE_FINALIST_SUBMISSION.json`](SMOKE_FINALIST_SUBMISSION.json).

#### Finalist concurrency correction

The first RecVAE pair exposed a schedule-isolation bug: jobs `10308826` and
`10308827` shared a checkpoint directory and raced on `resume.pt.tmp`.
`10308827` failed `1:0`; `10308826` was cancelled because its manifest had been
overwritten by the failed peer. Checkpoint writes now use unique temporary
files, and run directories are keyed by schedule plus retry attempt.

The clean RecVAE early-stopped-300 replacement `10308893` completed `0:0` in
54 epochs with validation NDCG@50 `0.2230677626`. RecVAE fixed-200 replacement
`10308905` and GERS fixed-200 job `10308828` also completed `0:0` with the
corrected code. GERS early-stopped-300 job `10308829` completed cleanly in 92
epochs with validation NDCG@50 `0.1848341447`.
The incident and replacement mapping are preserved in
[`SMOKE_FINALIST_RETRY_SUBMISSION.json`](SMOKE_FINALIST_RETRY_SUBMISSION.json).

#### Finalist completion

All four clean finalist jobs completed with exit `0:0`. Every run has best,
final, and resume checkpoints, and the clean-log scan found no traceback,
exception, OOM, NaN/Inf, timeout, or W&B step-order warning.

| Model | Fixed-200 validation NDCG@50 | Early-stopped validation NDCG@50 | Selected |
|---|---:|---:|---|
| RecVAE | **0.2260193247** | 0.2230677626 at epoch 54 | Fixed 200 |
| GERS Base | **0.1863873653** | 0.1848341447 at epoch 92 | Fixed 200 |

The normalized results and exact checkpoint paths are in
[`SMOKE_FINALIST_RESULTS.jsonl`](SMOKE_FINALIST_RESULTS.jsonl). The smoke
configuration is now frozen from validation metrics only; the test split has
not been inspected.

### W&B tracking correction

Rung 60 originally reused the rung-20 W&B identities because the tracking ID
did not include schedule parameters. Training, checkpoints, and local metrics
were unaffected, but W&B rejected epoch steps 0–19 as non-monotonic. Tracking
identity now includes epochs, minimum epochs, patience, and profile while the
checkpoint fingerprint remains stable for resume lineage.

The seven completed rung-60 metric suffixes were replayed once under fresh IDs
and verified `finished`; the W&B project now has 69 runs. Future jobs, including
the four finalists, use distinct native schedule identities. Replay also
records the source Slurm job and canonical tracking identity for provenance.

## Parallel paid-summary lane

Summary generation is running independently of the GPU lane so neither waits
on the other. The first required 100-user calibration wave was submitted to the
OpenAI Batch API as `batch_6a75f598c0e48190aa332fc0fddadbe0` using
`gpt-5-mini-2025-08-07`.

- Final state: `completed` with 100 requests completed and zero API failures.
- Actual usage: 129,083 input tokens and 24,112 output tokens.
- Actual cost: $0.08049475.
- Corrected privacy/format validation: 100 valid and zero invalid under
  validator `tears-summary-privacy-v2-whole-words`.
- Validation fingerprint:
  `9ab36908989bb1d52cadfa7eedb0688d36616bcc46865f953a969f6946681e70`.
- Plan fingerprint:
  `547ac9344652f6dc039be8fe49135110e865672d36e2920054448b0c864643a9`.
- Conservative projected cost: $0.1195235 before the 15% reserve.
- Prices used by the gate: $0.25/M input tokens and $2.00/M output tokens.
- Live plan/submission directory:
  `/network/scratch/a/adls/FullTrainingTEARS/summaries/smoke/train/`.
- Immutable paid-artifact mirror:
  `/home/mila/a/adls/FullTrainingTEARS_paid_artifacts/547ac9344652f6dc039be8fe49135110e865672d36e2920054448b0c864643a9/`.
- Slurm logs for every training rung:
  `/network/scratch/a/adls/FullTrainingTEARS/logs/`.

The 100-user calibration gate passed. The valid summaries are in
`/network/scratch/a/adls/FullTrainingTEARS/summaries/smoke/train/validated/summaries.jsonl`.
An initial nine-item retry plan is obsolete: all nine flags were substring
false positives (`tolerated`/`rated`, `strong`/`Tron`, `driven`/`Drive`, and
`psychological`/`Psycho`). Whole-word validation fixed the issue without a
second paid request. The next allowable API step is the 1,000-user wave.

### Exclusive 10,000-user summary wave

At the user's direction, the intermediate 1,000-user wave was replaced by one
10,000-user wave that explicitly excludes all 100 calibration users. The cohort
contains the remaining 9,900 pilot users plus 100 deterministic, activity-band-
matched train replacements, preserving the 9,000 train / 500 validation / 500
test composition.

- Cohort fingerprint:
  `2660cbb5de2a8c6b89f5b4962d07acee8d7806dbe332091cedc3346511ac2ac9`.
- Request-plan fingerprint:
  `74b6696146837239fadc68c79ae5974242fdf3b37fbad8fbf7ac2acc1f39f4b8`.
- Audit: 10,000 planned users, 10,000 unique IDs, zero duplicate IDs, and zero
  overlap with the 100-user calibration plan.
- OpenAI Batch: `batch_6a76227763d48190accf6463d131c6d4`.
- Initial state: `in_progress`, 0 completed, 0 failed.
- Conservative projected wave cost: $12.086126; prior actual spend: $0.08049475.
- Run root: `/network/scratch/a/adls/FullTrainingTEARS/10k/`.
- CPU watcher: Slurm `10310114`; it polls, downloads, validates, and prepares a
  reproducible 25-summary manual-review sample and quality report. It cannot
  submit retries or any remaining-user full run.
- W&B live monitor: entity `niita-mila`, project `tears-ml32m`, run
  `tears-ml32m-summaries-10k` (`bf669b8a3b237a61`). The existing watcher
  resumes this single run on every poll and records request progress, elapsed
  time, tokens, cost, validation counts, summary-length statistics, and final
  quality statistics. Its local configuration is
  `/network/scratch/a/adls/FullTrainingTEARS/10k/wandb_monitor_config.json`.

## Plan status and next gates

The RecVAE/GERS Base portion of Phase 3 is complete, and fixed-200 is frozen for
both models. Phase 3 remains open only for the summary-dependent TEARS lanes;
Phase 4 full-catalog selection cannot start until those inputs validate.

The remaining Phase 3 lanes have not been silently skipped:

- TEARS Base needs validated paid summaries. Submission remains behind the
  runbook's separate explicit spend confirmation.
- TEARS-RecVAE and GERS-RecVAE need a selected same-seed RecVAE checkpoint;
  their searches start only after that prerequisite is frozen.
- Phase 4 full-catalog support-threshold selection starts only after the quick
  RecVAE configuration is selected.
- Five-seed full training remains behind the scientific-pilot report and
  fingerprint-matched manual approval.

The paid API activity remains limited to the approved 100-user calibration
batch. No full-training promotion or test-set inspection occurred as part of
this update.
