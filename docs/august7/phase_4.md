# Phase 4 — Full-catalog selection

**Recorded:** 2026-08-07 (America/Toronto)  
**Plan source:** `docs/auguest6/ML32M_TEARS_GERS_SCALING_PLAN.md`, Phase 4  
**Current state:** complete; support 20 selected and frozen from validation-only metrics.  
**Test isolation:** no test metrics are used in this phase.

## Purpose and selection rule

Phase 4 compares the training-user catalogs produced with minimum item-support
thresholds 20, 50, and 200. Each candidate uses the selected quick-run RecVAE
configuration for a fixed 60 epochs and seed 2024. Validation records NDCG@20,
coverage@20, epoch duration, training throughput, and peak allocated GPU
memory.

The selected catalog will be the largest catalog whose validation NDCG@20 is
within 5% relative of the best candidate and whose peak GPU memory is below
72 GiB/GPU. The selected catalog and fingerprint are frozen before the
scientific pilot.

## Submitted jobs

| Slurm job ID | Job name | Status at last check | Dependency | Purpose |
|---|---|---|---|---|
| `10312906` | `tears-catalog-prepare` | **COMPLETED** on `cn-k004`; all three matrix manifests verified present | None | Build deterministic full-user sparse matrices for support 20, 50, and 200 without overwriting candidate artifacts. |
| `10312931` | `tears-catalog-s20-recvae-e60` | **COMPLETED**, 60/60 epochs | `afterok:10312906` (satisfied) | Train and validate the 60-epoch RecVAE for the support-20 catalog. |
| `10312932` | `tears-catalog-s50-recvae-e60` | **COMPLETED**, 60/60 epochs | `afterok:10312906` (satisfied) | Train and validate the 60-epoch RecVAE for the support-50 catalog. |
| `10312933` | `tears-catalog-s200-recvae-e60` | **COMPLETED**, 60/60 epochs | `afterok:10312906` (satisfied) | Train and validate the 60-epoch RecVAE for the support-200 catalog. |

The `afterok` dependency was intentional: no GPU job could start against a
partial matrix. Job `10312906` succeeded, so all three dependencies are now
satisfied.

## Shared configuration

The RecVAE recipe is the August smoke finalist selected using validation only:

| Setting | Value |
|---|---:|
| Model | `recvae` |
| Seed | `2024` |
| Epochs | `60` fixed (`minimum_epochs=60`, `patience=61`) |
| Dropout | `0.4` |
| Learning rate | `0.001` |
| Gamma | `0.0035` |
| Weight decay | `0.00001` |
| Latent dimension | `400` |
| Batch size | `64` (training default) |
| Validation batch size | `256` (training default) |
| Precision | BF16 autocast on CUDA |
| Validation metrics | Recall/NDCG at 20 and 50, coverage at 20 |

Recipe provenance is recorded in
`artifacts/august_smoke_recipe_selection.json`. The fixed-200 smoke finalist
reached validation NDCG@50 `0.22601932472360875`; the Phase 4 plan intentionally
uses only 60 epochs for the catalog comparison.

## Slurm resources

| Job class | Partition | CPU | Memory | GPU | Time limit |
|---|---|---:|---:|---:|---:|
| Preparation | `long` | 16 | 256 GiB | None | 24 hours |
| Each RecVAE candidate | `long` | 16 | 256 GiB | 1 × A100L | 96 hours |

All jobs exclude `cn-d[001-004]`, `cn-b[001-005]`, and
`cn-e[002-003]`, as required by the August Slurm strategy.

## Logs

| Job ID | Log path |
|---|---|
| `10312906` | `/network/scratch/a/adls/FullTrainingTEARS/logs/tears-catalog-prepare-10312906.out` |
| `10312931` | `/network/scratch/a/adls/FullTrainingTEARS/logs/tears-catalog-s20-recvae-e60-10312931.out` |
| `10312932` | `/network/scratch/a/adls/FullTrainingTEARS/logs/tears-catalog-s50-recvae-e60-10312932.out` |
| `10312933` | `/network/scratch/a/adls/FullTrainingTEARS/logs/tears-catalog-s200-recvae-e60-10312933.out` |

## Dataset outputs

The preparation manifest will be written to:

`/network/scratch/a/adls/FullTrainingTEARS/datasets/catalog_selection/prepare_manifest.json`

Candidate matrices and their manifests are isolated at:

| Support | Items | Retained ratings | Eligible validation users | Matrix fingerprint | Matrix directory |
|---:|---:|---:|---:|---|---|
| 20 | 22,343 | 31,704,735 | 9,920 | `27596ef71a4ac0f46e81ca97a40c4a9475bb1c2cc62c9db13819fe3cbeb714ce` | `/network/scratch/a/adls/FullTrainingTEARS/datasets/catalog_selection/support_20/matrix` |
| 50 | 15,364 | 31,463,399 | 9,920 | `0015e144aca1bd2df6d7048757e90e6256c465f47a9aa665d873352dc4427006` | `/network/scratch/a/adls/FullTrainingTEARS/datasets/catalog_selection/support_50/matrix` |
| 200 | 8,884 | 30,729,086 | 9,918 | `f1920deb1ea2c4c00964534168272b70a4e8376a2a660c11ee91c3199eaad1f0` | `/network/scratch/a/adls/FullTrainingTEARS/datasets/catalog_selection/support_200/matrix` |

Every matrix contains all 200,948 split users (180,948 train, 10,000
validation, and 10,000 untouched test). The completed preparation fingerprint
is `20ae2dd7dacfc2e06fdeb3fe5a1ead5634d069871511b90db48523625a24a7a4`.

The catalog CSV files are written under
`/network/scratch/a/adls/FullTrainingTEARS/datasets/catalog_selection/` as
`catalog_support_20.csv`, `catalog_support_50.csv`, and
`catalog_support_200.csv`.

## Training outputs and checkpoints

Each training run writes beneath:

`/network/scratch/a/adls/FullTrainingTEARS/checkpoints/full/recvae/seed-2024/<run-fingerprint-prefix>/`

The exact fingerprint directory is derived from the completed matrix manifest
and training arguments, so it is not guessed before preparation finishes.
Each run directory will contain:

- `manifest.json` — immutable arguments and data fingerprint;
- `metrics.jsonl` — per-epoch validation and resource measurements;
- `best.pt` — best validation-selection checkpoint;
- `final.pt` — final completed checkpoint;
- `resume.pt` — rolling resumable checkpoint;
- `result.json` — completion summary and checkpoint hashes.

The frozen Phase 4 selection report will be written to:

`/network/scratch/a/adls/FullTrainingTEARS/reports/catalog_selection.json`

### Frozen result

| Support | Validation NDCG@20 | Coverage@20 | Peak GPU memory | Result |
|---:|---:|---:|---:|---|
| 20 | 0.1861883218 | 0.2258425458 | 645,808,640 bytes | **Selected** |
| 50 | 0.1861681045 | 0.3206847175 | 463,575,040 bytes | Eligible |
| 200 | **0.1872822585** | 0.5353444394 | 293,094,400 bytes | Best NDCG@20 |

All candidates passed the 72-GiB memory gate. Support 20 is only 0.584% below
the best validation NDCG@20, so it is within the 5% band and is the largest
eligible catalog. The frozen selection-report fingerprint is
`d44de97e227b027d9371026a8e4c86cd9e6ca1a4dc2817f769d1e4a76a016fd2`.

## Submission and implementation records

- Local submission ledger: `artifacts/august_phase4_submission.json`
- Preparation entry point: `slurm/tears_prepare_catalog_selection.sbatch`
- Training entry point: `slurm/tears_pilot.sbatch`
- Catalog preparation implementation: `tears_training/data.py`
- Catalog selection implementation: `tears_training/catalog_selection.py`
- Validation/resource metrics: `tears_training/train.py`
- Verification at submission: `37 passed`

## Next required actions

1. Continue directly to the August scientific pilot on frozen support 20:
   full-training-user
   seed-2024 RecVAE, matched summary-based TEARS and genre-based GERS Base and
   hybrid models, plus the separately labeled full-user GERS ceiling.
