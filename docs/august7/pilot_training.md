# Phase 5 — Scientific-pilot training

**Submitted:** 2026-08-07 (America/Toronto)  
**Plan source:** `docs/auguest6/ML32M_TEARS_GERS_SCALING_PLAN.md`, including the August 7 cohort amendment  
**Catalog:** support 20, frozen Phase 4 selection fingerprint `d44de97e227b027d9371026a8e4c86cd9e6ca1a4dc2817f769d1e4a76a016fd2`  
**Test isolation:** training and model selection use validation only; pilot test evaluation is not part of these submitted jobs.  
**Current execution state:** matched-matrix preparation completed; full-user RecVAE and GERS ceiling are running; matched Base/hybrid jobs are dependency-queued.

### Live monitoring snapshot

At the latest audited snapshot, full RecVAE job `10313168` had completed epoch
29/200 with best validation NDCG@50 `0.2295615591`, mean epoch time 23.67
seconds, and peak allocated GPU memory 645,808,640 bytes. Full GERS ceiling job
`10313178` had completed epoch 68/200 with best validation NDCG@50
`0.1774980348`, mean epoch time 10.22 seconds, and peak allocated GPU memory
332,238,336 bytes. Both logs were clean; no retry was submitted.

## Matched summary cohort

The approved current pilot uses every summary that passed the existing
validator. TEARS and GERS use exactly the same persisted user cohort and sparse
matrix.

| Split | Users |
|---|---:|
| Train | 8,787 |
| Validation | 491 |
| Test reserved for post-selection pilot evaluation | 485 |
| **Total** | **9,763** |

- Validated summaries: `/network/scratch/a/adls/FullTrainingTEARS/10k/validated/summaries.jsonl`
- Matched cohort: `/network/scratch/a/adls/FullTrainingTEARS/10k/cohort/validated_users.csv`
- Cohort manifest: `/network/scratch/a/adls/FullTrainingTEARS/10k/cohort/validated_users.manifest.json`
- Cohort fingerprint: `24794a7a6f5a437d51c4d471da6834e6e484ea647fd22212868ff6e47bdccbe7`
- Summary SHA-256: `3a764267ae0aa35d52e6e82db8a8fa9922fcc10e99565e0d2092e8c5c12ef75b`
- Prepared matrix fingerprint: `a2cd89f043c8729ac84ebefbd4d38b8f98655d2bc77c598a19c7f51eeb8fd3b4`
- Prepared matrix shape: 9,763 users × 22,343 items; 1,531,476 retained ratings
- Defined ranking targets: 490 validation users and 482 test users

The other 237 users (213 train, 9 validation, and 15 test) are deferred to the
Phase 6 full-training summary-completion gate. No unvalidated or synthesized
summary is used in this pilot.

## Submitted jobs

| Job ID | Job name | Cohort | Initial/current role | Dependencies |
|---|---|---|---|---|
| `10313166` | `tears-pilot-9763-prepare` | Matched 9,763 | **Completed**; prepare selected-catalog pilot matrix | None |
| `10313168` | `tears-pilot-full-recvae-2024-e200` | 180,948 train / 10k validation | **Running**; full-user seed-2024 RecVAE | None |
| `10313178` | `tears-pilot-full-gers-ceiling-2024-e200` | 180,948 train / 10k validation | **Running**; separately labeled GERS ceiling | None |
| `10313183` | `tears-pilot-9763-tears-base-2024-e200` | Matched 9,763 | Summary-based TEARS Base | `afterok:10313166`, lane wait `afterany:10313168` |
| `10313187` | `tears-pilot-9763-gers-base-2024-e200` | Matched 9,763 | Genre-based GERS Base | `afterok:10313166`, lane wait `afterany:10313178` |
| `10313189` | `tears-pilot-9763-tears-recvae-2024-e200` | Matched 9,763 | Summary-based TEARS-RecVAE hybrid | `afterok:10313166:10313168:10313183` |
| `10313190` | `tears-pilot-9763-gers-recvae-2024-e200` | Matched 9,763 | Genre-based GERS-RecVAE hybrid | `afterok:10313166:10313168:10313187` |

The dependency graph limits training to two one-A100L lanes. Both hybrids
require the same new seed-2024 full-user RecVAE checkpoint. An `afterok`
failure prevents dependent hybrids from starting against missing or invalid
artifacts.

## Frozen recipes

All jobs use seed 2024, latent dimension 400, fixed 200 epochs,
`minimum_epochs=200`, and `patience=201`. T5 models use pinned
`google-t5/t5-base`, LoRA rank 64 and alpha 16. Hybrid training uses paper
alpha 0.5, a frozen RecVAE encoder/prior, a trainable shared decoder,
reconstruction plus diagonal Wasserstein OT and KL annealed to 0.5.

| Model | Dropout | LR | Gamma | Weight decay | OT weight | Recipe source |
|---|---:|---:|---:|---:|---:|---|
| Full RecVAE | 0.4 | 0.001 | 0.0035 | 0.00001 | 0.1 | August RecVAE smoke winner |
| TEARS Base | 0.1 | 0.0001 | 0.0035 | 0 | 0.1 | Original TEARS defaults; no extra search |
| GERS Base / ceiling | 0.1 | 0.001 | 0.0035 | 0 | 0.1 | August GERS Base smoke winner |
| TEARS-RecVAE | 0.1 | 0.0001 | 0.0035 | 0 | 1.0 | Original TEARS LR/dropout/epsilon defaults |
| GERS-RecVAE | 0.1 | 0.0001 | 0.0035 | 0 | 1.0 | Paper-faithful genre hybrid counterpart |

The machine-readable recipe is
`artifacts/august_pilot_recipe_selection.json`. No additional recipe
experiment was inserted.

## Paths

- Full selected matrix: `/network/scratch/a/adls/FullTrainingTEARS/datasets/catalog_selection/support_20/matrix`
- Matched pilot matrix: `/network/scratch/a/adls/FullTrainingTEARS/datasets/pilot/support_20/matrix`
- Full RecVAE checkpoint expected at: `/network/scratch/a/adls/FullTrainingTEARS/checkpoints/pilot/recvae/seed-2024/6749417de707/best.pt`
- Checkpoint root: `/network/scratch/a/adls/FullTrainingTEARS/checkpoints/pilot/`
- Logs: `/network/scratch/a/adls/FullTrainingTEARS/logs/<job-name>-<job-id>.out`
- W&B: entity `niita-mila`, project `tears-ml32m`

## W&B live monitoring

- Project: `https://wandb.ai/niita-mila/tears-ml32m`
- Full-user RecVAE: `https://wandb.ai/niita-mila/tears-ml32m/runs/6749417de70736ee`
- Full-user GERS ceiling: `https://wandb.ai/niita-mila/tears-ml32m/runs/2738e1dfe3f669d3`

Both active runs were verified through the W&B API in state `running`. Their
configs include model hyperparameters, Slurm job ID/name/node, execution and
training fingerprints, and local checkpoint/result references. Their summaries
include live train loss, validation Recall/NDCG/coverage, epoch duration,
throughput, peak memory, and output checkpoint paths. Queued Base/hybrid W&B
links are recorded when the runs start and become visible.

## Promotion sequence

1. Verify all seven jobs and checkpoint hashes.
2. Select Base and hybrid checkpoints from validation NDCG@50; hybrids use the
   mean across alpha 0, 0.5, and 1 as defined in the August plan.
3. Perform the one allowed matched-cohort pilot test evaluation only after
   selections are frozen.
4. Produce ranking, controllability, cost, throughput, and projected full-run
   reports.
5. At the Phase 6 gate, regenerate and validate the deferred 237 summaries,
   generate all remaining required summaries, then launch seeds 2020–2024
   using the frozen catalog and recipes.
