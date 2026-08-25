# August 23, 2026 - full-corpus TEARS promotion

**Recorded:** 2026-08-23 16:03 EDT
**Operational promotion:** COMPLETE
**Canonical five-seed matrix:** ACTIVE (17 / 25 complete)
**Website:** <https://tearsgersab.tail9d6ed0.ts.net/pilot/>
**Health:** <https://tearsgersab.tail9d6ed0.ts.net/pilot/api/health>

## Decision

The first usable full-corpus TEARS model is TEARS Base seed 2022. It was chosen
strictly on validation NDCG@50 from the two completed TEARS Base seeds and was
frozen before the reserved test split was read. The website now serves this
model for TEARS recommendations. GERS remains the previously promoted
9,763-profile scientific-pilot GERS-RecVAE model.

This closes the single-model operational deployment milestone. It does not
claim that the canonical five-seed research matrix or Phase 7 paper comparison
is complete; those jobs continue under the existing dependency graph.

## Promoted model

| Field | Value |
|---|---|
| Model | `tears_base` |
| Seed | `2022` |
| Training schedule | 200 / 200 epochs completed |
| Deployed checkpoint | best validation checkpoint from epoch 118 |
| Validation selection NDCG@50 | 0.1856043628 |
| Run directory | `/network/scratch/a/adls/FullTrainingTEARS/checkpoints/full/tears_base/seed-2022/10d017146b22` |
| Checkpoint | `/network/scratch/a/adls/FullTrainingTEARS/checkpoints/full/tears_base/seed-2022/10d017146b22/best.pt` |
| Checkpoint SHA-256 | `08f2609c6dae073a1d6c2a8ba14a3121c5ff4ec807201c7f97c7d4aa3c598e2b` |
| Run fingerprint | `969f4b6ddb710f372739a6203724716e3f2a5e77ec9c3cd62b9e5058158ac3bc` |
| GPU used for training | NVIDIA A100-SXM4-80GB |

Seed 2022 was selected over the other completed candidate, seed 2023, whose
best validation NDCG@50 was 0.1843369322. No test metric was used for this
decision. The immutable selection record is
[`artifacts/phase6_tears_seed2022_frozen_selection.json`](../../artifacts/phase6_tears_seed2022_frozen_selection.json).

## Dataset and split

| Field | Value |
|---|---:|
| Total profiles | 200,948 |
| Training users | 180,948 |
| Validation users | 10,000 |
| Test users | 10,000 |
| Eligible validation users | 9,920 |
| Eligible test users | 9,909 |
| Catalog items | 22,343 |
| Catalog support | 20 |
| Ratings | 31,704,735 |
| Matrix fingerprint | `27596ef71a4ac0f46e81ca97a40c4a9475bb1c2cc62c9db13819fe3cbeb714ce` |
| Catalog SHA-256 | `ac8b369749f8305bd712ec011150ff57b21459fe48be052f515ce5e291e00961` |

The exact split is 180,948 / 10,000 / 10,000, or approximately 90.05% / 4.98%
/ 4.98%. Therefore, the two evaluation partitions contain 10,000 users each;
they are not 10% of the 200,948-user population each.

## Frozen evaluation

Validation was reproduced first from the frozen best checkpoint. The test set
was then read once with the frozen-selection artifact and explicit
`--confirm-test` authorization.

| Split | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 |
|---|---:|---:|---:|---:|
| Validation | 0.137382 | 0.150933 | 0.252041 | 0.185604 |
| Test | 0.138613 | 0.149041 | 0.251260 | 0.183928 |

Artifacts:

- [`validation.json`](../../results/phase6-tears-seed2022/validation.json), SHA-256 `513dcfe8318063493689ac38ab285f0ab4fa7a0881f0ee6b9e52f588942e204d`
- [`test.json`](../../results/phase6-tears-seed2022/test.json), SHA-256 `c9580d14864987d459518c610e61c1de98f159c7b25bf3b6c33e566f23c4887b`
- Frozen selection SHA-256: `ce9a661b57ee79cd710ca37ced9d430746653f3c7460ff2b70fa683faadabc45`

Slurm job `10453494` was submitted for the evaluation. To avoid its projected
queue wait, the identical frozen command ran as a GPU job step inside the
already allocated website job `10314035`; the held duplicate was canceled
after both artifacts were written successfully.

## Website deployment

The `/pilot/api/recommend` endpoint now loads TEARS Base seed 2022 from the full
matrix. Startup verifies the matrix, split counts, catalog, links, run,
checkpoint fingerprint, checkpoint SHA-256, model name, and 200-epoch schedule
before accepting traffic.

The active deployment is Slurm job `10314035` on `cn-b001`; it is scheduled
through 2026-08-30 15:40 EDT, subject to the cluster's normal preemption and
automatic requeue behavior.

The public health response reports:

- deployment `full-tears-200948-with-pilot-gers-9763`;
- TEARS model `tears_base`, seed 2022, 200 epochs;
- 200,948 profiles and the 180,948 / 10,000 / 10,000 split;
- 22,343 support-20 items;
- the expected matrix and checkpoint fingerprints.

The public recommendation smoke case selected *The Fellowship of the Ring*
and requested fantasy/science-fiction recommendations. The selected title was
masked. The model returned *The Return of the King*, *The Two Towers*, and
*Inception*.

During promotion, an open browser tab using the old pilot matrix fingerprint
received HTTP 409. The full and pilot support-20 catalogs have the same
byte-verified catalog SHA-256 and model-item mapping, so the API now accepts the
old fingerprint only when the independently verified onboarding fingerprint is
also present. The rebuilt frontend sends the full matrix fingerprint. This
keeps existing tabs compatible without weakening catalog provenance.

## Remaining Phase 6 research jobs

At the recorded snapshot, 17 of 25 canonical model/seed slots are complete:
all five RecVAE, GERS Base, and GERS-RecVAE seeds, plus TEARS Base seeds 2022
and 2023.

| Family | Complete | Running | Dependency-pending |
|---|---|---|---|
| TEARS Base | 2022, 2023 | 2020 (`10429224`), 2021 (`10429231`) | 2024 (`10416424`) |
| TEARS-RecVAE | none | 2021 (`10416427`), 2022 (`10416429`) | 2020 (`10429232`), 2023 (`10416431`), 2024 (`10416433`) |

Finalizer `10426384` remains dependency-pending. It will verify all 25
canonical results after the terminal seeds finish. No additional deployment
decision is needed while those jobs run.

## Verification performed

- 10 focused Python serving tests passed.
- Frontend parity/contract test passed.
- React production build completed successfully.
- Local-node health and recommendation checks passed.
- Public health and recommendation checks passed.
- Cached old-fingerprint and new full-fingerprint request paths passed after
  the compatibility restart.
