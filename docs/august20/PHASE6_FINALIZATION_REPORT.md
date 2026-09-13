# Phase 6 finalization gate

**Recorded:** 2026-09-07T03:12:08.830385+00:00  
**Status:** **PASS**  
**Test isolation:** no test metrics were read.

## Gate summary

- Valid runs: **25 / 25**
- Frozen summary hash verified: **true**
- W&B verification requested: **true**
- Phase 7 ready: **true**

## Validation results

| Model | Valid seeds | Mean best selection NDCG@50 | Sample SD |
|---|---:|---:|---:|
| recvae | 5 | 0.234617 | 0.000470 |
| gers_base | 5 | 0.177215 | 0.000639 |
| tears_base | 5 | 0.184980 | 0.000516 |
| tears_recvae | 5 | 0.216634 | 0.000399 |
| gers_recvae | 5 | 0.216633 | 0.000382 |

## Run ledger

| Model | Seed | Epochs | Best selection NDCG@50 | W&B |
|---|---:|---:|---:|---|
| recvae | 2020 | 200 | 0.234597 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/62eb61407aed1ead) |
| recvae | 2021 | 200 | 0.235087 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/31fed35925901603) |
| recvae | 2022 | 200 | 0.235083 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/e7a7e560dc42d738) |
| recvae | 2023 | 200 | 0.234267 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/864c04c2241cf39b) |
| recvae | 2024 | 200 | 0.234049 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/561c3325321eb76e) |
| gers_base | 2020 | 200 | 0.176542 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/5245703b1e9bdba8) |
| gers_base | 2021 | 200 | 0.177186 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/312fe6237855c86e) |
| gers_base | 2022 | 200 | 0.176713 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/e55cc8175af57324) |
| gers_base | 2023 | 200 | 0.178135 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/de84ec8f4a172786) |
| gers_base | 2024 | 200 | 0.177498 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/1f22a69bfbe593b4) |
| tears_base | 2020 | 200 | 0.185237 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/5d0c999e296b2efd) |
| tears_base | 2021 | 200 | 0.185150 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/0c03ec9782d1d6da) |
| tears_base | 2022 | 200 | 0.185604 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/10d017146b22bc3a) |
| tears_base | 2023 | 200 | 0.184337 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/e16952923adb21fd) |
| tears_base | 2024 | 200 | 0.184573 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/a438c1462d32cdb8) |
| tears_recvae | 2020 | 200 | 0.216659 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/4aa46d0e8e9e3ef6) |
| tears_recvae | 2021 | 200 | 0.216946 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/46fbbc08b2564a97) |
| tears_recvae | 2022 | 200 | 0.216560 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/23de799096a51caa) |
| tears_recvae | 2023 | 200 | 0.217002 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/ab9b394064674024) |
| tears_recvae | 2024 | 200 | 0.216002 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/891e37eacb8e4d54) |
| gers_recvae | 2020 | 200 | 0.216612 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/daef7a64343076e6) |
| gers_recvae | 2021 | 200 | 0.216592 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/ffd6d9f397fb1c10) |
| gers_recvae | 2022 | 200 | 0.217148 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/d6335f29527f030b) |
| gers_recvae | 2023 | 200 | 0.216735 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/06bae270c7bf1e87) |
| gers_recvae | 2024 | 200 | 0.216080 | [finished](https://wandb.ai/niita-mila/tears-ml32m/runs/4d7e72d9dd04b98d) |

## Decision

Phase 6 is complete. All five model families have five valid, hashed,
200-epoch runs synchronized to W&B. Phase 7 may begin with validation-only
alpha selection; the test split remains closed until that selection is frozen.
