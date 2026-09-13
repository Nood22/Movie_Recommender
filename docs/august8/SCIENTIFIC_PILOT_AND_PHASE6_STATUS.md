# Scientific-pilot completion and Phase 6 forecast

> Production update (2026-08-19): Phase 6 full training is submitted. Frozen V10
> summaries cover all 200,948 users (`sha256 763605a2…e90b07`). Twenty-five
> seed-2020–2024 jobs are queued on one A100L each: RecVAE `10416410`–`10416414`,
> GERS Base `10416415`–`10416419`, TEARS Base `10416420`–`10416424`, and hybrids
> `10416425`–`10416434`. TEARS Base waits for all RecVAE jobs; each hybrid waits
> for its same-seed RecVAE `best.pt`. Test evaluation remains blocked. Ledger:
> `artifacts/phase6_full_submission.json`.
>
> Production update (2026-08-18): the frozen V10 summary run recovered under
> Slurm job `10403598` and the completion audit was written locally on 2026-08-19.
> See [`../august18/FROZEN_V10_PRODUCTION_RECOVERY_REPORT.md`](../august18/FROZEN_V10_PRODUCTION_RECOVERY_REPORT.md).

**Audited:** 2026-08-08 08:37 EDT  
**Scope:** amended Phase 5 scientific pilot, Phase 6 cohort, and remaining-time forecast  
**Machine-readable snapshot:** `scientific_pilot_artifact_audit.json`

## Executive status

All seven amended scientific-pilot jobs have terminal artifacts. The matrix
preparation completed, and each of the six training jobs wrote 200 metric rows,
`best.pt`, `final.pt`, `resume.pt`, and `result.json`. A case-insensitive scan
of all seven logs found no traceback, exception, error, OOM, NaN/Inf, failure,
cancellation, timeout, or killed-process marker.

Live scheduler confirmation was unavailable during this audit. `squeue` could
not contact the Slurm controller and `sacct` could not connect to `slurmctl02`.
The statuses below are therefore inferred from terminal application artifacts,
not independently confirmed Slurm `COMPLETED/0:0` records. A blank
`squeue -u "$USER"` result from this environment is not authoritative because
`$USER` resolves to `adls`, which Slurm rejects as an invalid user here.

## Amended scientific-pilot jobs

| Job ID | Role | Artifact status | Epochs | Best validation selection NDCG@50 |
|---|---|---|---:|---:|
| `10313166` | Prepare matched 9,763-user matrix | Complete | — | — |
| `10313168` | Full-user RecVAE, seed 2024 | Complete | 200 | 0.234049105812465 |
| `10313178` | Full-user GERS ceiling, seed 2024 | Complete | 200 | 0.177498034839969 |
| `10313183` | Matched TEARS Base, seed 2024 | Complete | 200 | 0.1761030198359976 |
| `10313187` | Matched GERS Base, seed 2024 | Complete | 200 | 0.17572202816301463 |
| `10313189` | Matched TEARS-RecVAE, seed 2024 | Complete | 200 | 0.2050045012211313 |
| `10313190` | Matched GERS-RecVAE, seed 2024 | Complete | 200 | 0.19971354780148487 |

The final pilot artifact was written by TEARS-RecVAE at approximately 07:27
EDT on August 8. Validation-only selection remains intact; the reserved pilot
test cohort has not been used.

The hybrid improvements over their matched Base counterparts are approximately
0.0289015 NDCG@50 for TEARS and 0.0239915 for GERS. The full-user RecVAE and
full-user GERS ceiling use a different cohort from the matched Base/hybrid
runs and should not be treated as direct matched-cohort comparisons.

## Frozen pilot data

The amended matched pilot contains 9,763 users:

| Split | Users |
|---|---:|
| Train | 8,787 |
| Validation | 491 |
| Test | 485 |
| **Total** | **9,763** |

Its selected-catalog matrix contains 22,343 items and 1,531,476 retained
ratings. The pilot cohort fingerprint is
`24794a7a6f5a437d51c4d471da6834e6e484ea647fd22212868ff6e47bdccbe7`.

## Exact Phase 6 cohort

Phase 6 restores the 237 users excluded from the amended pilot and uses the
fixed global cohort without planned exclusions:

| Split | Users |
|---|---:|
| Train | 180,948 |
| Validation | 10,000 |
| Test | 10,000 |
| **Total** | **200,948** |

Models train on 180,948 users, select checkpoints using 10,000 validation
users, and leave the 10,000 test users untouched until all selection decisions
are frozen. Summary generation must cover all 200,948 users. With 9,763 valid
pilot summaries already available, 191,185 summaries remain missing, including
the 237 required pilot retries.

The prepared support-20 full matrix manifest reports 200,948 users, 22,343
items, 31,704,735 retained ratings, and fingerprint
`27596ef71a4ac0f46e81ca97a40c4a9475bb1c2cc62c9db13819fe3cbeb714ce`.

## Remaining-time forecast

Assuming immediate manual promotion approval and authorization for the
remaining paid summary calls, the expected completion window for Phase 6
training and its artifact audit is **9–13 calendar days from August 8**, or
approximately **August 17–21, 2026**. August 22–23 is the safer planning
commitment. If Batch requests consume their full service windows or Slurm queue
delays are substantial, completion can move to approximately August 23–26.

| Remaining stage | Expected duration |
|---|---:|
| Pilot evaluation, checkpoint/hash audit, promotion report and approval | 4–12 hours |
| Generate and validate 191,185 missing summaries | 3.5–5.5 days |
| Full matrix and manifest finalization | 2–6 hours |
| Twenty-five full runs: five models by five seeds | 3–5 days |
| Final checkpoint, log, W&B, and fingerprint audit | 4–12 hours |

The completed 10,000-user Batch request took approximately 3.6 hours. The
remaining summaries require four sequential waves of at most 50,000 requests,
plus the 237-user regeneration and any validation retries. The Phase 6 training
plan contains 25 runs: RecVAE, TEARS Base, GERS Base, TEARS-RecVAE, and
GERS-RecVAE for seeds 2020–2024. The forecast assumes the preferred four-GPU
profile with at most four concurrent jobs and uses measured pilot epoch times.

This forecast ends at completed Phase 6 training and artifact verification. It
does not include Phase 7 full paper evaluation, alpha sweeps, or controllability
experiments.

## Authoritative paths

- Submission manifest: `artifacts/august_phase5_submission.json`
- Frozen recipe: `artifacts/august_pilot_recipe_selection.json`
- Pilot matrix: `/network/scratch/a/adls/FullTrainingTEARS/datasets/pilot/support_20/matrix`
- Full matrix: `/network/scratch/a/adls/FullTrainingTEARS/datasets/catalog_selection/support_20/matrix`
- Pilot checkpoints: `/network/scratch/a/adls/FullTrainingTEARS/checkpoints/pilot`
- Slurm logs: `/network/scratch/a/adls/FullTrainingTEARS/logs`
- Scaling plan: `docs/auguest6/ML32M_TEARS_GERS_SCALING_PLAN.md`
- Runbook: `docs/auguest6/ML32M_TRAINING_RUNBOOK.md`
