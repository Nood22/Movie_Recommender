# August 20, 2026 - Phase 6 Slurm report

**Recorded:** 2026-08-20 16:41 EDT  
**Scope:** frozen-V10 Phase 6 training, repaired dependency lanes, and finalizer  
**Test isolation:** no test-split metrics were read or used.

Machine-readable companion: [`phase6_slurm_snapshot.json`](phase6_slurm_snapshot.json).

## Current decision

Phase 6 is **not complete** and Phase 7 remains closed. The canonical 25-run
matrix is 15 complete, 4 running, and 6 dependency-pending. All missing slots
have a live Slurm path; there are no canonical seeds left without a queued job.

The single-seed deployment track is independently live using completed pilot
checkpoints. “Phase 7 remains closed” here refers to the reserved-test and final
five-seed comparison gate, not the website deployment.

| View | Complete | Running | Pending | Failed / superseded |
|---|---:|---:|---:|---:|
| Canonical model/seed slots (25) | **15** | **4** | **6** | 0 unresolved |
| Slurm training attempts (31) | 15 | 4 | 6 | 6 historical |
| Finalizer (1) | 0 | 0 | **1** | 0 |

The six historical failures remain visible for auditability. Three original
GERS Base attempts failed on `cn-g002` and now have completed replacements.
Three resumed TEARS attempts hit the RNG restore defect described below and now
have dependency-queued replacements.

## Running jobs

| Job | Model / seed | Node | Started EDT | Progress at snapshot | Recent epoch time |
|---|---|---|---|---:|---:|
| `10416422` | TEARS Base / 2022 | `cn-g003` | Aug 20 16:01 | latest epoch 1 (2 / 200) | 18.8 min |
| `10416423` | TEARS Base / 2023 | `cn-g014` | Aug 20 16:10 | latest epoch 0 (1 / 200) | 19.0 min |
| `10416427` | TEARS-RecVAE / 2021 | `cn-g003` | Aug 20 13:41 | latest epoch 7 (8 / 200) | 21.3 min |
| `10416429` | TEARS-RecVAE / 2022 | `cn-g029` | Aug 20 14:42 | latest epoch 4 (5 / 200) | 21.3 min |

All four jobs had fresh `metrics.jsonl` updates between 16:30 and 16:41 EDT.
The `EndTime` shown by Slurm is the 96-hour allocation limit, not the estimated
training completion time.

## Dependency lanes

The repaired graph preserves the original limit of two concurrent jobs per
summary-driven family.

| Lane | Ordered jobs | Compute-only lane finish |
|---|---|---|
| TEARS Base A | `10416422` seed 2022 -> `10429224` resume seed 2020 from epoch 50 -> `10416424` seed 2024 | Aug 27 around 21:00 EDT |
| TEARS Base B | `10416423` seed 2023 -> `10429231` resume seed 2021 from epoch 16 | Aug 25 around 17:00 EDT |
| TEARS-RecVAE A | `10416429` seed 2022 -> `10416433` seed 2024 | Aug 26 around 13:00 EDT |
| TEARS-RecVAE B | `10416427` seed 2021 -> `10429232` resume seed 2020 from epoch 29 -> `10416431` seed 2023 | **Aug 29 around 02:00 EDT** |

Finalizer `10426384` waits on the four terminal jobs: `10416424`, `10429231`,
`10416431`, and `10416433`. It then checks all 25 results, checkpoint hashes,
frozen-summary provenance, and W&B terminal states before opening Phase 7.

## Resume failure and repair

Jobs `10416420`, `10416421`, and `10416425` failed after their next starts with:

```text
TypeError: RNG state must be a torch.ByteTensor
```

The checkpoint was loaded with `map_location` set to the GPU, which also moved
the CPU torch RNG state to CUDA. Training now loads resume checkpoints on CPU
and normalizes every torch/CUDA RNG state to a CPU `uint8` tensor before
restoring it. The fix passes 22 focused tests and restored a real affected
checkpoint successfully. Replacement jobs are:

| Replacement | Model / seed | Waits for | Resume point |
|---|---|---|---:|
| `10429224` | TEARS Base / 2020 | `10416422` | epoch 50 |
| `10429231` | TEARS Base / 2021 | `10416423` | epoch 16 |
| `10429232` | TEARS-RecVAE / 2020 | `10416427` | epoch 29 |

The replacement commands retain the original model fingerprints and W&B run
identities, so they continue the existing checkpoints rather than creating
new canonical runs.

## End-time estimate

The observed median pace is approximately 18.9 minutes per TEARS Base epoch
and 21.3 minutes per TEARS-RecVAE epoch. Applying those rates to each remaining
epoch and the dependency graph gives a **compute-only Phase 6 gate estimate of
August 29 at about 02:30 EDT**, including the finalizer.

Repeated preemption and queue reacquisition have already added substantial wall
time. The operational planning estimate is therefore:

- **Point estimate:** September 1, 2026 at 12:00 EDT.
- **Planning window:** August 31 through September 2 EDT.
- **Confidence:** low-to-moderate; another resume defect would invalidate the
  estimate, while an uninterrupted queue would finish near the compute-only
  date.

Phase 7 may start only after `10426384` reports PASS. No test evaluation should
be scheduled from this ETA alone.
