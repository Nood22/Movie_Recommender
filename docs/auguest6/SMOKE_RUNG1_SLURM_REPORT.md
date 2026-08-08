# ML-32M Smoke Rung-1 Slurm Submission Report

> Historical submission snapshot. All 61 jobs later completed with exit
> `0:0`; see the audited
> [August 7 status report](../august7/AUGUST_7_TRAINING_STATUS.md) and its
> [machine-readable results](../august7/SMOKE_RUNG1_RESULTS.jsonl).

**Submitted:** 2026-08-06 (America/Toronto)  
**Dataset:** deterministic 10,000-user / top-4,000-item smoke profile  
**Seed:** 2024  
**Rung:** 5 epochs  
**Scheduler profile:** one A100L, 16 CPUs, 256 GB RAM, 96-hour limit  
**Plan:** at most two training jobs can run concurrently

## Current status — 2026-08-06 20:43 EDT

| Category | Count | Current state | Reason/details |
|---|---:|---|---|
| Data preparation | 1 | Pending | `Resources`; no dependency or failure |
| RecVAE rung-1 | 54 | Pending | Intentional dependency chain behind preparation |
| GERS Base rung-1 | 6 | Pending | Intentional dependency chain behind preparation and RecVAE lanes |
| Running | 0 | — | No node allocated yet |
| Completed | 0 | — | Training has not begun |
| Failed/cancelled/timeout | 0 | — | No failure state reported by `sacct` |

Slurm estimates preparation job `10306324` will start at
**2026-08-06 22:53:06 EDT** on `cn-h001` in the automatically selected
`long-cpu` partition. This is a scheduler estimate and can change as resource
availability changes. The job requests 16 CPUs, 256 GB RAM, and an eight-hour
limit; it has no dependency and is eligible to run.

No smoke matrix files, preparation log, training checkpoint, training metric,
or per-training W&B run exists yet. This is consistent with the preparation
job still being pending rather than evidence of a failure. The two W&B runs
currently associated with this phase are submission-manifest runs, not model
training runs.

## W&B run ledger — 2026-08-06 20:43 EDT

**Running Slurm training jobs: 0. Running W&B training runs: 0.** W&B cannot
create a model-training run until a dependent Slurm job starts and executes
`wandb.init`. The following are the only verified smoke-related W&B runs:

| W&B run ID | Name | Job type | State | Slurm job | Purpose |
|---|---|---|---|---|---|
| [`submit-7a35d3848c143214`](https://wandb.ai/niita-mila/tears-ml32m/runs/submit-7a35d3848c143214) | `smoke-rung1-all-submissions-7a35d384` | `slurm-submission` | Terminated after upload | None | Combined RecVAE/GERS submission manifest |
| [`submit-9044407bc05070ed`](https://wandb.ai/niita-mila/tears-ml32m/runs/submit-9044407bc05070ed) | `smoke-rung1-submission-9044407b` | `slurm-submission` | Terminated after upload | None | Initial RecVAE submission manifest |

These orchestration runs were intentionally ended after artifact logging; the
combined run ended at **2026-08-06 20:03:13 EDT**. They were never intended to
remain active or monitor Slurm. Their immutable artifacts remain available.
These are orchestration records, not recommendation-model results. When jobs
`10306331` onward begin, their exact W&B run IDs, Slurm IDs, URLs, state, model,
seed, and candidate hyperparameters will be appended to this ledger.

## Submission summary

- Data preparation: **1 job**, Slurm ID `10306324`.
- RecVAE hyperparameter grid: **54 jobs**, Slurm IDs `10306331`–`10306384`.
- GERS Base hyperparameter grid: **6 jobs**, Slurm IDs `10306389`–`10306394`.
- Total submitted in this phase: **61 jobs** (one preparation plus 60 training).
- State immediately after submission: preparation pending resources; all training jobs pending dependencies.
- Representative dependencies were verified with `scontrol show job` after
  submission; Slurm reports the intended `afterok` and two-lane `afterany`
  relationships.
- RecVAE plan fingerprint: `9044407bc05070ed1101e8051e717fbf5a6f2cdd7fb3d115bc82d91f9007bbba`.
- GERS Base plan fingerprint: `13618d0dfb34475d70fcfd6c8dbf5866d1363d1e953e48d6ae2f8914eb6355b0`.

## W&B submission artifact

The complete machine-readable RecVAE and GERS Base submission manifests were uploaded as one W&B artifact.

- Artifact: [`niita-mila/tears-ml32m/smoke-rung1-submissions-7a35d3848c14:v0`](https://wandb.ai/niita-mila/tears-ml32m/artifacts/slurm-submission/smoke-rung1-submissions-7a35d3848c14/v0)
- Submission run: [`submit-7a35d3848c143214`](https://wandb.ai/niita-mila/tears-ml32m/runs/submit-7a35d3848c143214)
- Combined submission fingerprint: `7a35d3848c143214650674cd060a0f6a980effb8a6f6805916535b763625bd74`.
- Contents: `SMOKE_RUNG1_SUBMISSION.json` and `SMOKE_GERS_RUNG1_SUBMISSION.json`.
- The submission W&B run being terminated does not delete or invalidate this
  artifact; artifact version `v0` remains the durable submission record.
- The earlier printed initial-artifact reference ending in `:v0:v0` was a
  display-string construction error. Its canonical reference has one version
  suffix: `niita-mila/tears-ml32m/smoke-rung1-submission-9044407bc050:v0`.

Each training job starts its own online W&B run under `niita-mila/tears-ml32m`. Its exact run ID is derived from the prepared matrix fingerprint and training fingerprint, so those per-training-run IDs are created only after job `10306324` completes and the dependent job starts.

## Submitted jobs

| Slurm ID | Model/run | Candidate | Hyperparameters | Dependency | Initial state |
|---:|---|---|---|---|---|
| 10306324 | Smoke data preparation | — | Split seed 2024; 10k users; top 4k items; sparse/mmap artifacts | None | Pending resources |
| 10306331 | RecVAE, seed 2024, 5 epochs | `7c2de45b7048` | dropout 0.1; lr 0.001; gamma 0.0035; wd 0 | afterok:10306324 | Pending dependency |
| 10306332 | RecVAE, seed 2024, 5 epochs | `02d05d493b71` | dropout 0.1; lr 0.001; gamma 0.0035; wd 0.00001 | afterok:10306324 | Pending dependency |
| 10306333 | RecVAE, seed 2024, 5 epochs | `f5f2e6959825` | dropout 0.1; lr 0.001; gamma 0.004; wd 0 | afterok:10306324; afterany:10306331 | Pending dependency |
| 10306334 | RecVAE, seed 2024, 5 epochs | `9b78419f69c6` | dropout 0.1; lr 0.001; gamma 0.004; wd 0.00001 | afterok:10306324; afterany:10306332 | Pending dependency |
| 10306335 | RecVAE, seed 2024, 5 epochs | `deeb21b1d9bc` | dropout 0.1; lr 0.001; gamma 0.005; wd 0 | afterok:10306324; afterany:10306333 | Pending dependency |
| 10306336 | RecVAE, seed 2024, 5 epochs | `7dfa696e1e46` | dropout 0.1; lr 0.001; gamma 0.005; wd 0.00001 | afterok:10306324; afterany:10306334 | Pending dependency |
| 10306337 | RecVAE, seed 2024, 5 epochs | `d54a9e8f409b` | dropout 0.1; lr 0.0001; gamma 0.0035; wd 0 | afterok:10306324; afterany:10306335 | Pending dependency |
| 10306338 | RecVAE, seed 2024, 5 epochs | `3260b1183cb2` | dropout 0.1; lr 0.0001; gamma 0.0035; wd 0.00001 | afterok:10306324; afterany:10306336 | Pending dependency |
| 10306339 | RecVAE, seed 2024, 5 epochs | `88ab46de7ad7` | dropout 0.1; lr 0.0001; gamma 0.004; wd 0 | afterok:10306324; afterany:10306337 | Pending dependency |
| 10306340 | RecVAE, seed 2024, 5 epochs | `07dc545c1c99` | dropout 0.1; lr 0.0001; gamma 0.004; wd 0.00001 | afterok:10306324; afterany:10306338 | Pending dependency |
| 10306341 | RecVAE, seed 2024, 5 epochs | `ced17dfd7cae` | dropout 0.1; lr 0.0001; gamma 0.005; wd 0 | afterok:10306324; afterany:10306339 | Pending dependency |
| 10306342 | RecVAE, seed 2024, 5 epochs | `cf8e37b38646` | dropout 0.1; lr 0.0001; gamma 0.005; wd 0.00001 | afterok:10306324; afterany:10306340 | Pending dependency |
| 10306343 | RecVAE, seed 2024, 5 epochs | `caef8f278640` | dropout 0.1; lr 0.00001; gamma 0.0035; wd 0 | afterok:10306324; afterany:10306341 | Pending dependency |
| 10306344 | RecVAE, seed 2024, 5 epochs | `79ef673672ca` | dropout 0.1; lr 0.00001; gamma 0.0035; wd 0.00001 | afterok:10306324; afterany:10306342 | Pending dependency |
| 10306345 | RecVAE, seed 2024, 5 epochs | `4a5bcb68a4e2` | dropout 0.1; lr 0.00001; gamma 0.004; wd 0 | afterok:10306324; afterany:10306343 | Pending dependency |
| 10306346 | RecVAE, seed 2024, 5 epochs | `ae19bae520d7` | dropout 0.1; lr 0.00001; gamma 0.004; wd 0.00001 | afterok:10306324; afterany:10306344 | Pending dependency |
| 10306347 | RecVAE, seed 2024, 5 epochs | `af5ea5cdb41b` | dropout 0.1; lr 0.00001; gamma 0.005; wd 0 | afterok:10306324; afterany:10306345 | Pending dependency |
| 10306348 | RecVAE, seed 2024, 5 epochs | `22a9e271ce19` | dropout 0.1; lr 0.00001; gamma 0.005; wd 0.00001 | afterok:10306324; afterany:10306346 | Pending dependency |
| 10306349 | RecVAE, seed 2024, 5 epochs | `a14dc03d94f1` | dropout 0.2; lr 0.001; gamma 0.0035; wd 0 | afterok:10306324; afterany:10306347 | Pending dependency |
| 10306350 | RecVAE, seed 2024, 5 epochs | `cb782b006e55` | dropout 0.2; lr 0.001; gamma 0.0035; wd 0.00001 | afterok:10306324; afterany:10306348 | Pending dependency |
| 10306351 | RecVAE, seed 2024, 5 epochs | `7aaebfcf1b04` | dropout 0.2; lr 0.001; gamma 0.004; wd 0 | afterok:10306324; afterany:10306349 | Pending dependency |
| 10306352 | RecVAE, seed 2024, 5 epochs | `1fc99d48835c` | dropout 0.2; lr 0.001; gamma 0.004; wd 0.00001 | afterok:10306324; afterany:10306350 | Pending dependency |
| 10306353 | RecVAE, seed 2024, 5 epochs | `bee1842d9a4c` | dropout 0.2; lr 0.001; gamma 0.005; wd 0 | afterok:10306324; afterany:10306351 | Pending dependency |
| 10306354 | RecVAE, seed 2024, 5 epochs | `63aa35b9339c` | dropout 0.2; lr 0.001; gamma 0.005; wd 0.00001 | afterok:10306324; afterany:10306352 | Pending dependency |
| 10306355 | RecVAE, seed 2024, 5 epochs | `4d5fd6e066b2` | dropout 0.2; lr 0.0001; gamma 0.0035; wd 0 | afterok:10306324; afterany:10306353 | Pending dependency |
| 10306356 | RecVAE, seed 2024, 5 epochs | `47f25c196a51` | dropout 0.2; lr 0.0001; gamma 0.0035; wd 0.00001 | afterok:10306324; afterany:10306354 | Pending dependency |
| 10306357 | RecVAE, seed 2024, 5 epochs | `4082bc2bce44` | dropout 0.2; lr 0.0001; gamma 0.004; wd 0 | afterok:10306324; afterany:10306355 | Pending dependency |
| 10306358 | RecVAE, seed 2024, 5 epochs | `a78a9f3254ee` | dropout 0.2; lr 0.0001; gamma 0.004; wd 0.00001 | afterok:10306324; afterany:10306356 | Pending dependency |
| 10306359 | RecVAE, seed 2024, 5 epochs | `d98d3e236a00` | dropout 0.2; lr 0.0001; gamma 0.005; wd 0 | afterok:10306324; afterany:10306357 | Pending dependency |
| 10306360 | RecVAE, seed 2024, 5 epochs | `b44caebc4cdf` | dropout 0.2; lr 0.0001; gamma 0.005; wd 0.00001 | afterok:10306324; afterany:10306358 | Pending dependency |
| 10306361 | RecVAE, seed 2024, 5 epochs | `39ec5d094b03` | dropout 0.2; lr 0.00001; gamma 0.0035; wd 0 | afterok:10306324; afterany:10306359 | Pending dependency |
| 10306362 | RecVAE, seed 2024, 5 epochs | `6325448fcc2b` | dropout 0.2; lr 0.00001; gamma 0.0035; wd 0.00001 | afterok:10306324; afterany:10306360 | Pending dependency |
| 10306363 | RecVAE, seed 2024, 5 epochs | `012564882acf` | dropout 0.2; lr 0.00001; gamma 0.004; wd 0 | afterok:10306324; afterany:10306361 | Pending dependency |
| 10306364 | RecVAE, seed 2024, 5 epochs | `e448bac62372` | dropout 0.2; lr 0.00001; gamma 0.004; wd 0.00001 | afterok:10306324; afterany:10306362 | Pending dependency |
| 10306365 | RecVAE, seed 2024, 5 epochs | `5eca4b2e5404` | dropout 0.2; lr 0.00001; gamma 0.005; wd 0 | afterok:10306324; afterany:10306363 | Pending dependency |
| 10306366 | RecVAE, seed 2024, 5 epochs | `7fa9dca61fd9` | dropout 0.2; lr 0.00001; gamma 0.005; wd 0.00001 | afterok:10306324; afterany:10306364 | Pending dependency |
| 10306367 | RecVAE, seed 2024, 5 epochs | `4cfe1e2e65a3` | dropout 0.4; lr 0.001; gamma 0.0035; wd 0 | afterok:10306324; afterany:10306365 | Pending dependency |
| 10306368 | RecVAE, seed 2024, 5 epochs | `69c3b4deb263` | dropout 0.4; lr 0.001; gamma 0.0035; wd 0.00001 | afterok:10306324; afterany:10306366 | Pending dependency |
| 10306369 | RecVAE, seed 2024, 5 epochs | `0c8b91a7a1ed` | dropout 0.4; lr 0.001; gamma 0.004; wd 0 | afterok:10306324; afterany:10306367 | Pending dependency |
| 10306370 | RecVAE, seed 2024, 5 epochs | `4551f768cde3` | dropout 0.4; lr 0.001; gamma 0.004; wd 0.00001 | afterok:10306324; afterany:10306368 | Pending dependency |
| 10306371 | RecVAE, seed 2024, 5 epochs | `07d5f073f4e9` | dropout 0.4; lr 0.001; gamma 0.005; wd 0 | afterok:10306324; afterany:10306369 | Pending dependency |
| 10306372 | RecVAE, seed 2024, 5 epochs | `8306ee0048c3` | dropout 0.4; lr 0.001; gamma 0.005; wd 0.00001 | afterok:10306324; afterany:10306370 | Pending dependency |
| 10306373 | RecVAE, seed 2024, 5 epochs | `f367c32349f5` | dropout 0.4; lr 0.0001; gamma 0.0035; wd 0 | afterok:10306324; afterany:10306371 | Pending dependency |
| 10306374 | RecVAE, seed 2024, 5 epochs | `c3dbbfdd5b95` | dropout 0.4; lr 0.0001; gamma 0.0035; wd 0.00001 | afterok:10306324; afterany:10306372 | Pending dependency |
| 10306375 | RecVAE, seed 2024, 5 epochs | `430532bffe56` | dropout 0.4; lr 0.0001; gamma 0.004; wd 0 | afterok:10306324; afterany:10306373 | Pending dependency |
| 10306376 | RecVAE, seed 2024, 5 epochs | `f268bc3590af` | dropout 0.4; lr 0.0001; gamma 0.004; wd 0.00001 | afterok:10306324; afterany:10306374 | Pending dependency |
| 10306377 | RecVAE, seed 2024, 5 epochs | `a036acbe4445` | dropout 0.4; lr 0.0001; gamma 0.005; wd 0 | afterok:10306324; afterany:10306375 | Pending dependency |
| 10306378 | RecVAE, seed 2024, 5 epochs | `4e245c857ee2` | dropout 0.4; lr 0.0001; gamma 0.005; wd 0.00001 | afterok:10306324; afterany:10306376 | Pending dependency |
| 10306379 | RecVAE, seed 2024, 5 epochs | `b26a5ebb911a` | dropout 0.4; lr 0.00001; gamma 0.0035; wd 0 | afterok:10306324; afterany:10306377 | Pending dependency |
| 10306380 | RecVAE, seed 2024, 5 epochs | `95c8506aa79d` | dropout 0.4; lr 0.00001; gamma 0.0035; wd 0.00001 | afterok:10306324; afterany:10306378 | Pending dependency |
| 10306381 | RecVAE, seed 2024, 5 epochs | `419bca62c08a` | dropout 0.4; lr 0.00001; gamma 0.004; wd 0 | afterok:10306324; afterany:10306379 | Pending dependency |
| 10306382 | RecVAE, seed 2024, 5 epochs | `46a429a37506` | dropout 0.4; lr 0.00001; gamma 0.004; wd 0.00001 | afterok:10306324; afterany:10306380 | Pending dependency |
| 10306383 | RecVAE, seed 2024, 5 epochs | `2803d16df6d0` | dropout 0.4; lr 0.00001; gamma 0.005; wd 0 | afterok:10306324; afterany:10306381 | Pending dependency |
| 10306384 | RecVAE, seed 2024, 5 epochs | `74628b4e1622` | dropout 0.4; lr 0.00001; gamma 0.005; wd 0.00001 | afterok:10306324; afterany:10306382 | Pending dependency |
| 10306389 | GERS Base, seed 2024, 5 epochs | `6bdbc48868f8` | dropout 0.1; lr 0.001 | afterok:10306324,10306383,10306384 | Pending dependency |
| 10306390 | GERS Base, seed 2024, 5 epochs | `7a93d8eb695f` | dropout 0.1; lr 0.0001 | afterok:10306324,10306383,10306384 | Pending dependency |
| 10306391 | GERS Base, seed 2024, 5 epochs | `289c68677d84` | dropout 0.2; lr 0.001 | afterok:10306324,10306383,10306384; afterany:10306389 | Pending dependency |
| 10306392 | GERS Base, seed 2024, 5 epochs | `9043b7078091` | dropout 0.2; lr 0.0001 | afterok:10306324,10306383,10306384; afterany:10306390 | Pending dependency |
| 10306393 | GERS Base, seed 2024, 5 epochs | `d5e1bd9c9037` | dropout 0.4; lr 0.001 | afterok:10306324,10306383,10306384; afterany:10306391 | Pending dependency |
| 10306394 | GERS Base, seed 2024, 5 epochs | `b60cb03a1938` | dropout 0.4; lr 0.0001 | afterok:10306324,10306383,10306384; afterany:10306392 | Pending dependency |

## Dependency and concurrency design

Jobs `10306331` and `10306332` are the heads of two RecVAE lanes and start only after data-preparation job `10306324` succeeds. Every later RecVAE job waits for the preparation job and for the job two positions earlier, limiting the grid to two concurrent GPUs.

The GERS Base lane starts after preparation and both final RecVAE lane jobs (`10306383` and `10306384`) succeed. Its six jobs use the same two-lane pattern. This prevents the combined submission from exceeding the approved two-job smoke concurrency.

## Jobs not submitted in this action

- TEARS Base: waiting for validated cached summaries. No paid Batch request was authorized or submitted.
- TEARS-RecVAE and GERS-RecVAE: waiting for rung selection and the selected seed-2024 RecVAE checkpoint.
- Rungs 20 and 60: waiting for rung-5 validation NDCG@50 and GPU-hour results.
- Full five-seed phase: still behind the scientific-pilot promotion gate.
- Test evaluation: not run and not inspected.

## Machine-readable records and logs

- [RecVAE submission manifest](SMOKE_RUNG1_SUBMISSION.json)
- [GERS Base submission manifest](SMOKE_GERS_RUNG1_SUBMISSION.json)
- Slurm logs: `/network/scratch/a/adls/FullTrainingTEARS/logs/%x-%j.out`
- Checkpoints: `/network/scratch/a/adls/FullTrainingTEARS/checkpoints/smoke/`
- W&B project: [`niita-mila/tears-ml32m`](https://wandb.ai/niita-mila/tears-ml32m)
