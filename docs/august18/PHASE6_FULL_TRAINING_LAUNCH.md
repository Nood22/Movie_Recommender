# Phase 6 — five-seed full training launch

**Submitted:** 2026-08-19 (America/Toronto)  
**Plan source:** `docs/auguest6/ML32M_TEARS_GERS_SCALING_PLAN.md`, Phase 6  
**Summaries:** frozen V10 evidence-gated Emiliano corpus, 200,948 / 200,948 users  
**Catalog:** support 20, matrix fingerprint `27596ef71a4ac0f46e81ca97a40c4a9475bb1c2cc62c9db13819fe3cbeb714ce`  
**Test isolation:** no test split is evaluated in these jobs.

## Why this phase

The scientific-pilot used 9,763 validated summaries. Phase 6 restores the full
fixed split (180,948 train / 10,000 validation / 10,000 reserved test) now that
every user has a validated summary. RecVAE and GERS Base train on ratings/genres
only; TEARS Base and TEARS-RecVAE consume the new summaries; hybrids freeze the
same-seed RecVAE encoder.

## Canonical summaries

- Production corpus: `/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/full_cohort/v010_20260816_evidence_gated_emiliano_frozen/validated/final_summaries.jsonl`
- Training hardlink: `/network/scratch/a/adls/FullTrainingTEARS/summaries/validated/v010_frozen/final_summaries.jsonl`
- SHA-256: `763605a2a22dfa77e32c0226fb1f2318b8889b89fa9517e341fb2ac008e90b07`
- Coverage against the support-20 matrix `users.csv`: 200,948 matched, 0 missing, 0 extra

## Submitted jobs

All jobs use the frozen 200-epoch schedule (`minimum_epochs=200`, `patience=201`),
latent size 400, one A100L, 16 CPUs, 256 GiB, 96-hour limit.

| Model | Seeds | Slurm IDs | Dependency |
|---|---|---|---|
| RecVAE | 2020–2024 | `10416410`–`10416414` | RecVAE 2024 waits on 2020 (`afterany`) |
| GERS Base | 2020–2024 | `10416415`–`10416419` | GERS 2024 waits on 2020 |
| TEARS Base | 2020–2024 | `10416420`–`10416424` | `afterok` all RecVAE; two concurrent |
| TEARS-RecVAE / GERS-RecVAE | 2020–2024 | `10416425`–`10416434` | `afterok` same-seed RecVAE; two concurrent |

Immediately after submission every Phase 6 job was `PENDING`. RecVAE and GERS
were waiting on scheduler priority; TEARS and hybrids were waiting on RecVAE.

## Paths

- Matrix: `/network/scratch/a/adls/FullTrainingTEARS/datasets/catalog_selection/support_20/matrix`
- Checkpoints: `/network/scratch/a/adls/FullTrainingTEARS/checkpoints/full/<model>/seed-<seed>/`
- Logs: `/network/scratch/a/adls/FullTrainingTEARS/logs/<job-name>-<job-id>.out`
- W&B: entity `niita-mila`, project `tears-ml32m`
- Ledger: `artifacts/phase6_full_submission.json`

Predicted RecVAE `best.pt` directories:

| Seed | Run directory prefix |
|---:|---|
| 2020 | `checkpoints/full/recvae/seed-2020/62eb61407aed` |
| 2021 | `checkpoints/full/recvae/seed-2021/31fed3592590` |
| 2022 | `checkpoints/full/recvae/seed-2022/e7a7e560dc42` |
| 2023 | `checkpoints/full/recvae/seed-2023/864c04c2241c` |
| 2024 | `checkpoints/full/recvae/seed-2024/561c3325321e` |
