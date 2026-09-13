# Frozen V10 production recovery report

**Audited and submitted:** 2026-08-18 15:42 EDT  
**Protocol:** `frozen_v10_evidence_gated_emiliano`  
**Recovery Slurm job:** `10403598`  
**Submission status:** accepted by `sbatch`; workload completion is intentionally not being watched interactively

## Executive status

The original V10 job `10394118` stopped safely on 2026-08-17 at 19:11:45 EDT
while finalizing shard 7. Shards 0 through 6 are complete and checkpointed.
The shard-7 OpenAI Batch itself completed all 15,000 requests with zero API
failures, but validation found one objective failure: user `109468` contained a
literal placeholder. The retired semantic keyword screen also marked 14,154
otherwise objectively valid rows; those rows do not require regeneration.

Before recovery, the persisted checkpoint reports 105,845 accepted users and
95,103 remaining users. Of the shard-7 users, 845 were already accepted. The
recovery will checkpoint the additional 14,154 objectively valid rows, retry
only the single unresolved user, and then continue automatically through
shards 8 through 13.

| Item | Status |
|---|---:|
| Planned users | 200,948 |
| Finalized shards | 0–6 |
| Accepted before recovery | 105,845 |
| Remaining before recovery | 95,103 |
| Shard-7 Batch requests completed | 15,000 / 15,000 |
| Shard-7 API request failures | 0 |
| Shard-7 objective failures requiring retry | 1 |
| Successful users scheduled for resubmission | 0 |
| Frozen retry limit | 2 targeted retries |
| Existing projected spend for 120,000 submitted requests | $45.80289 |
| Conservative total authorization cap | $99.7099906803 |

## Automatic recovery behavior

The runner now performs the following sequence without manual polling:

1. Preserve the original `STOP-production.json` and write a separate
   `RESUME-production.json` authorization record.
2. Promote only rows that pass objective validation. Retired semantic-screen
   findings remain diagnostic and do not trigger regeneration.
3. Build each retry from the database's unresolved-user set, so accepted users
   cannot be submitted again.
4. Poll each submitted Batch to a terminal state inside the Slurm allocation.
5. Retry unresolved objective failures only, up to the frozen two-retry limit.
6. Finalize the shard, update its checkpoint and W&B metrics, and move to the
   next remaining shard.
7. Stop automatically on a real integrity, duplicate, API terminal-state,
   retry-limit, frozen-code, or cost-cap failure.
8. After shard 13, build the final corpus and production completion audit.

The recovery retains the existing sequential-shard rule and does not start the
next shard until the preceding shard is fully accepted and hash-verified.

## Submission record

The following job was accepted by the Slurm controller:

```text
Submitted batch job 10403598
```

It runs `scripts/slurm_frozen_v10_remaining_shards.sh` on the `long` partition
with a two-day limit, eight CPUs, and 12 GiB of memory. Expected log paths are:

- `/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/full_cohort/v010_20260816_evidence_gated_emiliano_frozen/logs/slurm-v10-full-cohort-10403598.out`
- `/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/full_cohort/v010_20260816_evidence_gated_emiliano_frozen/logs/slurm-v10-full-cohort-10403598.err`

The scheduler's read endpoints were intermittently unavailable during this
audit, so no `PENDING` or `RUNNING` claim is made beyond the authoritative
`sbatch` acceptance and job ID.

## Verification

The updated runner and Slurm wrapper passed syntax checks. The focused targeted
retry regression test and the existing full-cohort suite pass together:

```text
6 passed in 4.22s
```

The regression test proves that clean and semantic-diagnostic rows are accepted,
the objective failure stays unresolved, and the retry result replaces only that
user in the selected shard output.

## Authoritative artifacts

- Production root: `/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/full_cohort/v010_20260816_evidence_gated_emiliano_frozen`
- Current checkpoint: `checkpoints/latest.json`
- Original stop: `reports/STOP-production.json`
- Recovery authorization written on job start: `reports/RESUME-production.json`
- Per-shard status: `reports/shards/shard-NNN-final-status.json`
- Final completion report: `reports/final_production_completion.json`
