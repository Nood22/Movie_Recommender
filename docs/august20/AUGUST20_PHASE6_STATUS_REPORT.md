# August 20, 2026 - Phase 6 run status and comparison

**Recorded:** 2026-08-20 13:23 EDT  
**W&B:** [niita-mila/tears-ml32m](https://wandb.ai/niita-mila/tears-ml32m/workspace?nw=nwuserniita)  
**Scope:** retraining on the frozen V10 summary corpus and the 25-job, five-seed full-training launch  
**Test isolation:** the 10,000-user test split remains untouched; every number below is execution status or validation evidence.

Machine-readable companion: [`phase6_run_snapshot.json`](phase6_run_snapshot.json).

> **Deployment track (16:55 EDT):** the completed 200-epoch pilot
> TEARS-RecVAE and GERS-RecVAE checkpoints are live at
> <https://tearsgersab.tail9d6ed0.ts.net/pilot/> on Slurm job `10314035`.
> Public health, website, TEARS inference, and GERS inference smoke tests pass.
> This starts deployment from a promoted single seed without opening the
> reserved-test or five-seed comparison gate. See
> [`AUGUST20_DEPLOYMENT_REPORT.md`](AUGUST20_DEPLOYMENT_REPORT.md).

> **Slurm update (16:41 EDT):** the three GERS Base replacements completed,
> bringing the canonical matrix to 15 / 25 completed runs. Four TEARS jobs are
> running and six are dependency-pending. Automatic resume exposed a CPU/CUDA
> RNG-state restore error in three preempted jobs; the error is fixed, focused
> tests pass, and replacement jobs `10429224`, `10429231`, and `10429232` are
> inserted into the existing two-job lanes. See
> [`AUGUST20_PHASE6_SLURM_REPORT.md`](AUGUST20_PHASE6_SLURM_REPORT.md) for the
> current queue, repaired dependency graph, and end-time estimate.

> **Completion continuation (13:45 EDT):** GERS Base seeds 2021-2023 were
> resubmitted as jobs `10426276`, `10426278`, and `10426292` with `cn-g002`
> excluded. The TEARS-RecVAE jobs were rebalanced into two dependency lanes,
> preserving the original two-job concurrency cap. Finalizer job `10426384`
> will verify all 25 local results, checkpoint hashes, frozen-summary provenance,
> and W&B final states before writing `PHASE6_FINALIZATION_REPORT.md`. The test
> split remains closed.
>
> **Resume repair (14:04 EDT):** requeued jobs previously restarted at epoch 0
> because the launcher did not pass `--resume`. Training now auto-loads the
> matching run's `resume.pt` on restart, including optimizer, RNG, best score,
> and epoch state. Explicit `--resume` still takes precedence. Focused training
> and finalization coverage passes (21 tests).

## Executive status

The generated-summary corpus is complete and in active use by the TEARS jobs:
200,948 / 200,948 users are validated, and the training hardlink still has
SHA-256 `763605a2a22dfa77e32c0226fb1f2318b8889b89fa9517e341fb2ac008e90b07`.

Phase 6 is **not complete**. Of the 25 submitted jobs, 12 completed, 3 failed
before Python started, 3 are running, and 7 are waiting on dependencies.
RecVAE and GERS-RecVAE have complete five-seed results. GERS Base has only two
valid seeds. TEARS Base and TEARS-RecVAE are still training on the new summaries.

| Gate | August 19 snapshot | August 20 snapshot |
|---|---:|---:|
| Completed | 0 | **12** |
| Failed | 0 | **3** |
| Running | 4 | **3** |
| Pending | 21 | **7** |
| W&B training records | 0 | **15** |
| Five-seed RecVAE complete | no | **yes** |
| Five-seed GERS-RecVAE complete | no | **yes** |
| Five-seed GERS Base complete | no | no, 2 / 5 |
| Any summary-driven family complete | no | no |
| Phase 7 ready | no | **no** |

## Status by family

| Model | Completed | Failed | Running | Pending | Current disposition |
|---|---:|---:|---:|---:|---|
| RecVAE | **5** | 0 | 0 | 0 | Final five-seed checkpoints available |
| GERS Base | 2 | **3** | 0 | 0 | Incomplete; seeds 2021-2023 need clean reruns |
| GERS-RecVAE | **5** | 0 | 0 | 0 | Final five-seed checkpoints available |
| TEARS Base | 0 | 0 | **2** | 3 | Training on frozen V10 summaries |
| TEARS-RecVAE | 0 | 0 | **1** | 4 | Training on frozen V10 summaries |

The 12 successful jobs all reached epoch 200, wrote `result.json`, and recorded
SHA-256 hashes for `best.pt`, `final.pt`, and `resume.pt`. Their W&B runs are
all `finished`.

## Completed-run comparison

There are two different validation views and they should not be conflated:

- `best_validation_selection_ndcg@50` comes from each local `result.json` and
  selects the best checkpoint. For a hybrid, the selection metric is the mean
  NDCG@50 across alpha 0, 0.5, and 1.
- The W&B alpha-0.5 values below are final-epoch raw NDCG@50 at the paper's
  fixed blend. These are directly useful for comparing the hybrid at alpha 0.5
  with RecVAE's final-epoch NDCG@50.

### Best-checkpoint selection metric

| Model | Seeds | Mean | Sample SD | Difference from RecVAE |
|---|---:|---:|---:|---:|
| RecVAE | 5 | **0.234617** | 0.000470 | - |
| GERS-RecVAE | 5 | 0.216633 | 0.000382 | -0.017983 (-7.66%) |
| GERS Base | 2 only | 0.177020 | 0.000676 | not a final five-seed comparison |

RecVAE wins the checkpoint-selection objective on every paired seed. That does
not mean the GERS hybrid loses at the paper blend: the hybrid objective averages
the strong alpha-0.5 midpoint with weaker alpha-0 and alpha-1 endpoints.

### Final-epoch W&B metric

| Validation metric | Seeds | Mean | Sample SD |
|---|---:|---:|---:|
| RecVAE NDCG@50 | 5 | 0.232473 | 0.001183 |
| GERS-RecVAE alpha-0.5 NDCG@50 | 5 | **0.237644** | 0.000775 |
| GERS-RecVAE alpha-0 NDCG@50 | 5 | 0.220407 | 0.000921 |
| GERS-RecVAE alpha-1 NDCG@50 | 5 | 0.175291 | 0.000280 |

At alpha 0.5, GERS-RecVAE is higher than RecVAE on all five paired seeds. The
mean paired gain is **+0.005171 NDCG@50 (+2.22%)**. This is validation evidence,
not a test-set result or a final paper claim.

| Seed | RecVAE best selection | GERS-RecVAE best selection | RecVAE final NDCG@50 | GERS-RecVAE final alpha-0.5 |
|---:|---:|---:|---:|---:|
| 2020 | 0.234597 | 0.216612 | 0.233615 | **0.237252** |
| 2021 | 0.235087 | 0.216592 | 0.233192 | **0.237271** |
| 2022 | 0.235083 | 0.217148 | 0.231620 | **0.238614** |
| 2023 | 0.234267 | 0.216735 | 0.233095 | **0.238300** |
| 2024 | 0.234049 | 0.216080 | 0.230842 | **0.236784** |

## Summary-driven runs in progress

The local checkpoints are fresher than W&B for two preempted jobs, so Slurm and
the local `metrics.jsonl` files are authoritative for current execution state.

| Job | Model / seed | Slurm | Preemptions | Best validation so far | W&B state |
|---|---|---|---:|---|---|
| `10416420` | TEARS Base / 2020 | running on `cn-g014` | 3 | 0.182839 selection NDCG@50, epoch 40 | [`running`](https://wandb.ai/niita-mila/tears-ml32m/runs/5d0c999e296b2efd) |
| `10416421` | TEARS Base / 2021 | running on `cn-g003` | 5 | 0.180481 selection NDCG@50, epoch 30 | [`crashed`, stale](https://wandb.ai/niita-mila/tears-ml32m/runs/0c03ec9782d1d6da) |
| `10416425` | TEARS-RecVAE / 2020 | running on `cn-g008` | 2 | 0.215323 selection; 0.240155 at alpha 0.5, epoch 19 | [`crashed`, stale](https://wandb.ai/niita-mila/tears-ml32m/runs/4aa46d0e8e9e3ef6) |

At the snapshot, all three had updated `resume.pt` and `metrics.jsonl` within
the preceding 15 minutes. The TEARS-RecVAE seed-2020 alpha-0.5 value is
promising, but it is one incomplete seed and must remain provisional.

Pending on the intended dependency/concurrency chains:

| Model | Seeds | Job IDs |
|---|---|---|
| TEARS Base | 2022-2024 | `10416422`-`10416424` |
| TEARS-RecVAE | 2021-2024 | `10416427`, `10416429`, `10416431`, `10416433` |

## Failed GERS Base jobs

Jobs `10416416`, `10416417`, and `10416418` (seeds 2021-2023) all landed on
`cn-g002` together and failed in two seconds with exit `127:0`:

```text
.venv/bin/torchrun: cannot execute: required file not found
```

They failed before Python, W&B initialization, manifest creation, or GPU work.
The `torchrun` shebang resolves through the virtual environment to
`/usr/bin/python3.10`; the failure is therefore a launcher/node environment
problem, not a GERS model result. The successful seeds 2020 and 2024 ran on
other nodes. A clean rerun should first exclude or validate `cn-g002`.

## W&B reconciliation

W&B contains 15 Phase 6 training records created since August 19:

- 12 `finished`, matching the 12 successful Slurm jobs.
- 1 `running`, matching TEARS Base seed 2020.
- 2 `crashed` even though Slurm and local checkpoints show active resumed jobs.
- No records for the 3 launcher failures because Python never started.
- No records for the 7 dependency-pending jobs because `wandb.init` has not run.

Repeated scheduler preemption explains the two stale `crashed` states. Those
records should not be counted as failed training runs while their Slurm jobs
remain active and checkpoints continue to advance.

## Finalization decision

This August 20 snapshot and the completed RecVAE/GERS-RecVAE result ledgers are
finalized. Phase 6 itself is **not finalized**:

1. Complete replacement GERS Base jobs `10426276`, `10426278`, and `10426292`.
2. Let the three summary-driven jobs finish and allow their dependency chains to drain.
3. Require five successful seeds and hashed `result.json` artifacts for every model family.
4. Reconcile resumed W&B records so final states agree with Slurm.
5. Keep Phase 7 and the reserved test split closed until those gates pass.

No new summary-generation API calls or costs were incurred for this status
update.
