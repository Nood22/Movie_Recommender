# August 19, 2026 — full status report

**Recorded:** 2026-08-19 12:39 EDT  
**Scope:** frozen V10 full-cohort summary completion, local production audit, and Phase 6 five-seed full training launch  
**Plan source:** `docs/auguest6/ML32M_TEARS_GERS_SCALING_PLAN.md`  
**Test isolation:** the 10,000 test users remain untouched; no test metrics were used today.

Machine-readable companion: [`phase6_queue_snapshot.json`](phase6_queue_snapshot.json).

## Executive status

Phase 6 summary generation is **complete**. Every user in the frozen 200,948-user
split has a validated V10 summary. The missing production completion audit was
written locally. Five-seed full training is **submitted and underway**: four
RecVAE jobs are running; the remaining 21 jobs are pending GPU resources or
RecVAE dependencies.

| Gate | Status |
|---|---|
| Frozen V10 summaries | **200,948 / 200,948** accepted |
| Every user accounted for exactly once | **true** |
| Production completion audit | **written** |
| Successful users resubmitted | **0** |
| Exact recorded production API cost | **$76.7454** |
| Conservative production cost cap | **$99.7100** |
| Phase 6 training jobs submitted | **25 / 25** |
| Phase 6 jobs running at this snapshot | **4 RecVAE** |
| Phase 6 jobs pending | **21** |
| Phase 7 paper evaluation | **not started** |

## What was done today

Work followed the last remaining production gate, then the first training step
of Phase 6.

### 1. Remaining-shard audit (no new Batch spend)

The earlier recovery report still listed shard 13 as remaining. The live
checkpoint already had shards 0–13 accepted:

- Accepted users: 200,948
- Remaining users: 0
- Shard 13: 5,948 / 5,948
- Successful users resubmitted: 0

Recovery job `10403598` (submitted 2026-08-18) had already generated shards
8–13 and built the corpus, then **failed while writing**
`reports/final_production_completion.json`.

Failure:

```text
KeyError: 'api_failures'
```

Cause: shard 0's final-status file uses the older field `api_failed` instead of
`api_failures`. Shards 1–13 already used the newer schema.

A second Slurm completion attempt, job `10416328`, was submitted at 12:24 EDT
and failed in 28 seconds because `finalize_corpus()` refuses to overwrite the
existing immutable corpus:

```text
RuntimeError: Artifact already exists: .../validated/final_summaries.jsonl
```

The corpus already contained 200,948 lines. Per instruction, that job was **not
resubmitted**. Completion was run locally against the existing artifacts.

### 2. Local production finalization

`scripts/run_frozen_v10_remaining_shards.py` was updated to:

- Map legacy shard-0 counts (`api_failed`, `allowed_cleanup_users`).
- Reuse an existing hash-verified corpus instead of rewriting it.

`final_production_completion.json` was then written locally. No OpenAI Batch
call was made.

Result:

| Item | Value |
|---|---:|
| Accepted / expected | 200,948 / 200,948 |
| API successes | 200,948 |
| API failures | 0 |
| Objective failures remaining | 0 |
| Retries | 1 (user `109468`, shard 7, placeholder) |
| Authorized cleanup users | 908 |
| Input tokens | 293,780,936 |
| Cached input tokens | 3,328 |
| Output tokens | 40,023,193 |
| Exact production cost | $76.74543560000174 |
| Ready for next training stage | true |

Completion report:

`/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/full_cohort/v010_20260816_evidence_gated_emiliano_frozen/reports/final_production_completion.json`

### 3. Phase 6 training launch

The scientific-pilot trained on 9,763 validated-summary users. Phase 6 uses the
full frozen split now that summaries exist for every user:

| Split | Users |
|---|---:|
| Train | 180,948 |
| Validation | 10,000 |
| Test (reserved) | 10,000 |
| **Total** | **200,948** |

Coverage of the new summaries against the support-20 matrix `users.csv` was
checked before launch: **200,948 matched, 0 missing, 0 extra, 0 empty**.

The corpus was hard-linked into the training summary path (same inode, no
second 185 MiB copy):

`/network/scratch/a/adls/FullTrainingTEARS/summaries/validated/v010_frozen/final_summaries.jsonl`  
SHA-256 `763605a2a22dfa77e32c0226fb1f2318b8889b89fa9517e341fb2ac008e90b07`

Missing Slurm wrappers were restored (`slurm/tears_pilot.sbatch`,
`tears_full_2gpu.sbatch`, `tears_full_4gpu.sbatch`). Full/pilot orchestration
now uses the frozen 200-epoch schedule (`minimum_epochs=200`, `patience=201`)
instead of the smoke early-stopping defaults.

Twenty-five jobs were submitted on the one-A100L pilot profile, using the
frozen Phase 5 recipes. TEARS Base waits for all RecVAE jobs so long T5 runs
cannot occupy GPUs first. Each hybrid waits for its same-seed RecVAE `best.pt`.

## Cost report

Recorded OpenAI Batch spend only. Slurm/GPU, W&B, TMDB, hosting, and Codex
session costs are not in these artifacts.

| Item | USD |
|---|---:|
| Paid 1,000-user calibration (prior record) | 0.6908 |
| Frozen V10 production, all 200,948 users | 76.7454 |
| **Total known API spend** | **77.4362** |
| Conservative production authorization cap | 99.7100 |
| New Batch calls made on 2026-08-19 | 0.0000 |

The production figure is the checkpointed exact cost, not an estimate. One
targeted retry (user `109468`) is included.

## Live Slurm queue — 2026-08-19 12:39 EDT

Unrelated jobs still running on this account: `tears_public` `10314035` and
`mila-code` `10416215`. They were not modified.

### Phase 6 training jobs

| Job ID | Name | State | Node / reason | Elapsed at snapshot |
|---|---|---|---|---|
| `10416410` | RecVAE seed 2020 | **RUNNING** | `cn-g016` | 3:25 |
| `10416411` | RecVAE seed 2021 | **RUNNING** | `cn-g024` | 2:58 |
| `10416412` | RecVAE seed 2022 | **RUNNING** | `cn-g019` | 2:25 |
| `10416413` | RecVAE seed 2023 | **RUNNING** | `cn-g029` | 1:43 |
| `10416414` | RecVAE seed 2024 | PENDING | Dependency (`afterany` 2020) | — |
| `10416415` | GERS Base seed 2020 | PENDING | Resources | — |
| `10416416` | GERS Base seed 2021 | PENDING | Priority | — |
| `10416417` | GERS Base seed 2022 | PENDING | Priority | — |
| `10416418` | GERS Base seed 2023 | PENDING | Priority | — |
| `10416419` | GERS Base seed 2024 | PENDING | Dependency (`afterany` GERS 2020) | — |
| `10416420` | TEARS Base seed 2020 | PENDING | Dependency (all RecVAE `afterok`) | — |
| `10416421` | TEARS Base seed 2021 | PENDING | Dependency (all RecVAE `afterok`) | — |
| `10416422` | TEARS Base seed 2022 | PENDING | Dependency | — |
| `10416423` | TEARS Base seed 2023 | PENDING | Dependency | — |
| `10416424` | TEARS Base seed 2024 | PENDING | Dependency | — |
| `10416425` | TEARS-RecVAE seed 2020 | PENDING | Dependency (RecVAE 2020) | — |
| `10416426` | GERS-RecVAE seed 2020 | PENDING | Dependency (RecVAE 2020) | — |
| `10416427` | TEARS-RecVAE seed 2021 | PENDING | Dependency | — |
| `10416428` | GERS-RecVAE seed 2021 | PENDING | Dependency | — |
| `10416429` | TEARS-RecVAE seed 2022 | PENDING | Dependency | — |
| `10416430` | GERS-RecVAE seed 2022 | PENDING | Dependency | — |
| `10416431` | TEARS-RecVAE seed 2023 | PENDING | Dependency | — |
| `10416432` | GERS-RecVAE seed 2023 | PENDING | Dependency | — |
| `10416433` | TEARS-RecVAE seed 2024 | PENDING | Dependency | — |
| `10416434` | GERS-RecVAE seed 2024 | PENDING | Dependency | — |

Count at snapshot: **4 running, 21 pending, 0 failed, 0 completed**.

Stdout/stderr for the four running RecVAE jobs were still empty at this
snapshot (jobs had only been on node for a few minutes). That is consistent
with W&B/torch startup before the first epoch log.

### Completed / failed jobs from this workstream

| Job ID | Role | Final state | Exit | Note |
|---|---|---|---|---|
| `10403598` | V10 remaining-shard recovery | FAILED | 1:0 | Generated shards 8–13; died on completion JSON |
| `10416328` | V10 completion retry | FAILED | 1:0 | Corpus already existed; not retried |
| local `final_completion` | Production audit | **complete** | — | Wrote the missing completion report |

## Frozen training configuration

Recipes are copied from `artifacts/august_pilot_recipe_selection.json` into
`artifacts/phase6_full_selection.json`. All Phase 6 jobs use seed-specific
runs, latent dimension 400, BF16, batch size 64, and paper alpha 0.5 for
hybrids.

| Model | Dropout | LR | Gamma | Weight decay | OT weight |
|---|---:|---:|---:|---:|---:|
| RecVAE | 0.4 | 0.001 | 0.0035 | 0.00001 | 0.1 |
| GERS Base | 0.1 | 0.001 | 0.0035 | 0 | 0.1 |
| TEARS Base | 0.1 | 0.0001 | 0.0035 | 0 | 0.1 |
| TEARS-RecVAE | 0.1 | 0.0001 | 0.0035 | 0 | 1.0 |
| GERS-RecVAE | 0.1 | 0.0001 | 0.0035 | 0 | 1.0 |

Slurm request per job: partition `long`, 16 CPUs, 256 GiB, 1×A100L, 96 hours,
`SIGUSR1` 5 minutes before timeout. Nodes `cn-d[001-004]`, `cn-b[001-005]`,
and `cn-e[002-003]` are excluded.

Predicted RecVAE checkpoint directories:

| Seed | `best.pt` path |
|---:|---|
| 2020 | `/network/scratch/a/adls/FullTrainingTEARS/checkpoints/full/recvae/seed-2020/62eb61407aed/best.pt` |
| 2021 | `/network/scratch/a/adls/FullTrainingTEARS/checkpoints/full/recvae/seed-2021/31fed3592590/best.pt` |
| 2022 | `/network/scratch/a/adls/FullTrainingTEARS/checkpoints/full/recvae/seed-2022/e7a7e560dc42/best.pt` |
| 2023 | `/network/scratch/a/adls/FullTrainingTEARS/checkpoints/full/recvae/seed-2023/864c04c2241c/best.pt` |
| 2024 | `/network/scratch/a/adls/FullTrainingTEARS/checkpoints/full/recvae/seed-2024/561c3325321e/best.pt` |

## Code and artifact changes made today

| Path | Change |
|---|---|
| `scripts/run_frozen_v10_remaining_shards.py` | Legacy shard-0 field mapping; reuse existing finalized corpus |
| `tests/test_run_frozen_v10_remaining_shards.py` | Regression coverage for both behaviors |
| `tears_training/orchestrate.py` | Pilot/full plans use the frozen 200-epoch schedule |
| `slurm/tears_pilot.sbatch` | Restored one-A100L training wrapper |
| `slurm/tears_full_2gpu.sbatch` | Restored two-GPU wrapper (not used for this launch) |
| `slurm/tears_full_4gpu.sbatch` | Restored four-GPU wrapper (not used for this launch) |
| `artifacts/phase6_full_selection.json` | Frozen recipes plus catalog/summary pointers |
| `artifacts/phase6_promotion_approval.json` | Explicit Phase 6 go-ahead record |
| `artifacts/phase6_full_submission.json` | Job-ID ledger |
| `docs/august8/SCIENTIFIC_PILOT_AND_PHASE6_STATUS.md` | Banner updated for the 2026-08-19 launch |
| `docs/august18/PHASE6_FULL_TRAINING_LAUNCH.md` | Short launch note |

## Authoritative paths

- Production root: `/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/full_cohort/v010_20260816_evidence_gated_emiliano_frozen`
- Completion audit: `.../reports/final_production_completion.json`
- Final summaries: `.../validated/final_summaries.jsonl`
- Training summaries: `/network/scratch/a/adls/FullTrainingTEARS/summaries/validated/v010_frozen/final_summaries.jsonl`
- Full matrix: `/network/scratch/a/adls/FullTrainingTEARS/datasets/catalog_selection/support_20/matrix`
- Matrix fingerprint: `27596ef71a4ac0f46e81ca97a40c4a9475bb1c2cc62c9db13819fe3cbeb714ce`
- Checkpoints: `/network/scratch/a/adls/FullTrainingTEARS/checkpoints/full/`
- Logs: `/network/scratch/a/adls/FullTrainingTEARS/logs/<job-name>-<job-id>.out`
- W&B: https://wandb.ai/niita-mila/tears-ml32m

## What is not done yet

1. The 25 Phase 6 runs have not finished. RecVAE 2020–2023 are in early startup;
   TEARS and hybrids have not started.
2. Checkpoint hashes, W&B run IDs, and per-seed validation NDCG@50 are not
   available until each job writes `result.json`.
3. Phase 7 (full paper evaluation, alpha sweep, test-set read, controllability)
   has not started and must wait until all selection decisions are frozen.
4. The four-GPU full profile was restored but not used; this launch matches the
   proven one-A100L scientific-pilot resource shape.

## How to re-check the queue

```bash
squeue -u "$USER" -o '%.10i %.12P %.48j %.2t %.10M %.8N %R'
sacct -j 10416410,10416411,10416412,10416413,10416414 --format=JobID,JobName,State,Elapsed,NodeList
```
