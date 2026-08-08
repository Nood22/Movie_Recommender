# ML-32M TEARS/GERS Training Runbook

This runbook operates the implementation specified by
[`ML32M_TEARS_GERS_SCALING_PLAN.md`](ML32M_TEARS_GERS_SCALING_PLAN.md). The
commands are resumable and fingerprinted. Planning and dry runs are safe; paid
OpenAI Batch requests and Slurm jobs require separate explicit confirmation.

The August 7 amendment authorizes the current scientific pilot to use all
9,763 validated-summary users as a matched 8,787/491/485 cohort. Do not insert
unvalidated summaries or silently synthesize replacements. Carry the remaining
237 users into the Phase 6 summary-completion gate.

## 1. Environment and preflight

Run from the repository root inside the existing virtual environment:

```bash
export SCRATCH=/network/scratch/a/adls
. .venv/bin/activate
python -m tears_training.data --config configs/ml32m.toml verify \
  --initialize-layout --dry-run
python -m tears_training.orchestrate --config configs/ml32m.toml \
  --phase smoke \
  --matrix-dir "$SCRATCH/FullTrainingTEARS/datasets/smoke/matrix" \
  --summaries "$SCRATCH/FullTrainingTEARS/summaries/smoke/all.jsonl" \
  --models recvae --rung 5 --dry-run
```

Inspect `preflight.workspace_secret_findings`. The scanner reports only file and
line locations, never matching secret values. Revoke any exposed key before
continuing. W&B must be authenticated for online runs under entity
`niita-mila`, project `tears-ml32m`.

Initialize the scratch layout only after the dry run is clean:

```bash
python -m tears_training.data --config configs/ml32m.toml verify \
  --initialize-layout
```

This verifies ML-32M checksums and creates the dataset symlink and artifact
directories. It does not duplicate the raw dataset.

## 2. Prepare deterministic data

```bash
python -m tears_training.data --config configs/ml32m.toml prepare \
  --profile smoke --dry-run
python -m tears_training.data --config configs/ml32m.toml prepare \
  --profile smoke
```

For catalog selection, prepare each training-support candidate separately:

```bash
python -m tears_training.data --config configs/ml32m.toml prepare \
  --profile full --item-support 20
python -m tears_training.data --config configs/ml32m.toml prepare \
  --profile full --item-support 50
python -m tears_training.data --config configs/ml32m.toml prepare \
  --profile full --item-support 200
```

Each matrix directory contains compressed CSR matrices, memory-mappable CSR
components, user/item mappings, eligibility denominators, and an immutable
fingerprint. Validation/test text and interaction histories share the same
chronological 70/30 boundary.

## 3. Plan and validate cached summaries

Start with 100 users, then 1,000, then the 10,000-user pilot cohort:

```bash
python -m tears_training.summaries --config configs/ml32m.toml plan \
  --matrix-dir "$SCRATCH/FullTrainingTEARS/datasets/smoke/matrix" \
  --split train --limit 100 --dry-run
python -m tears_training.summaries --config configs/ml32m.toml plan \
  --matrix-dir "$SCRATCH/FullTrainingTEARS/datasets/smoke/matrix" \
  --split train --limit 100
```

Before submission, obtain the current Batch input/output prices from the
official OpenAI pricing page and the organization queue limit from the Limits
page. Test the cost gate without making an external request:

```bash
python -m tears_training.summaries --config configs/ml32m.toml submit \
  --plan "$SCRATCH/FullTrainingTEARS/summaries/smoke/train/request_plan.json" \
  --input-price-per-million INPUT_BATCH_PRICE \
  --output-price-per-million OUTPUT_BATCH_PRICE \
  --actual-spend-usd 0 --dry-run
```

Paid submission additionally requires `OPENAI_API_KEY` and
`--confirm-spend`. The request plan and JSONL are mirrored to the append-only
home backup before upload:

```bash
python -m tears_training.summaries --config configs/ml32m.toml submit \
  --plan "$SCRATCH/FullTrainingTEARS/summaries/smoke/train/request_plan.json" \
  --input-price-per-million INPUT_BATCH_PRICE \
  --output-price-per-million OUTPUT_BATCH_PRICE \
  --actual-spend-usd ACTUAL_SPEND --confirm-spend
```

Polling is one-shot and resumable; it never loops indefinitely:

```bash
python -m tears_training.summaries --config configs/ml32m.toml poll \
  --submission "$SCRATCH/FullTrainingTEARS/summaries/smoke/train/submission.json"
python -m tears_training.summaries --config configs/ml32m.toml validate \
  --plan "$SCRATCH/FullTrainingTEARS/summaries/smoke/train/request_plan.json"
```

Validation enforces structured JSON, word bounds, title/year/rating privacy,
and cross-user duplication. Invalid outputs create a retry plan, capped at two
attempts and still protected by the same explicit spend gate. No fallback text
is synthesized.

Stop after the 10,000-user pilot. Remaining paid generation requires the
manual promotion artifact described below.

## 4. Local training dry run

```bash
python -m tears_training.train --config configs/ml32m.toml \
  --profile smoke \
  --matrix-dir "$SCRATCH/FullTrainingTEARS/datasets/smoke/matrix" \
  --model recvae --seed 2024 --epochs 5 --dry-run
```

TEARS models require `--summaries`. Hybrid models additionally require the
same-seed RecVAE `best.pt` through `--recvae-checkpoint`. Checkpoints with a
different matrix or hyperparameter fingerprint are rejected on resume.

## 5. Successive-halving and Slurm planning

The first smoke rung generates 54 RecVAE, 6-per-Base, or 18-per-hybrid
candidates from the approved grids. Start RecVAE and Base searches first:

```bash
python -m tears_training.orchestrate --config configs/ml32m.toml \
  --phase smoke \
  --matrix-dir "$SCRATCH/FullTrainingTEARS/datasets/smoke/matrix" \
  --summaries "$SCRATCH/FullTrainingTEARS/summaries/smoke/validated/all.jsonl" \
  --models recvae tears_base gers_base --rung 5 --dry-run
```

Remove `--dry-run` to write a plan. This still does not submit anything. Slurm
submission requires both `--submit` and `--confirm-submit`. Pilot jobs use one
A100L; full profiles use two or four A100Ls with concurrency capped at eight or
four jobs respectively.

Later rungs use a JSONL result file containing `model`, `hyperparameters`,
`validation_selection_ndcg@50`, and `gpu_hours`:

```bash
python -m tears_training.orchestrate --config configs/ml32m.toml \
  --phase smoke --matrix-dir MATRIX --summaries SUMMARIES \
  --models recvae tears_base gers_base --rung 20 \
  --previous-results rung-5-results.jsonl --dry-run
```

Run hybrids only after selecting a RecVAE checkpoint:

```bash
python -m tears_training.orchestrate --config configs/ml32m.toml \
  --phase smoke --matrix-dir MATRIX --summaries SUMMARIES \
  --models tears_recvae gers_recvae --rung 5 \
  --recvae-checkpoint-template '/absolute/checkpoints/seed-{seed}/best.pt' \
  --dry-run
```

Rungs 5, 20, and 60 retain the best third by the required validation score.
The 200/300 comparison uses the lower GPU-hour run when scores differ by less
than 1% relative.

## 6. Evaluation and test isolation

Validation can sweep hybrid alpha from 0.0 through 1.0:

```bash
python -m tears_training.evaluate \
  --run-manifest RUN_DIR/manifest.json --split validation \
  --output RUN_DIR/validation.json
```

Test evaluation is deliberately blocked unless the caller supplies both a
frozen selection artifact and explicit confirmation:

```bash
python -m tears_training.evaluate \
  --run-manifest RUN_DIR/manifest.json --split test --alpha SELECTED_ALPHA \
  --frozen-selection frozen-selection.json --confirm-test \
  --output RUN_DIR/test.json
```

## 7. Scientific pilot and full promotion

For the amended pilot, prepare the selected-catalog matrix using the persisted
9,763-user validated cohort file. Use that same matrix for TEARS Base, GERS
Base, TEARS-RecVAE, and GERS-RecVAE. The full-training-user RecVAE and the
separately labeled GERS ceiling continue to use all 180,948 training users.

Pilot/full planning requires a selection JSON mapping each model name to its
frozen hyperparameters. Full planning defaults to the four-GPU profile and
seeds 2020–2024.

Generate the promotion report:

```bash
python -m tears_training.report promotion \
  --execution-manifest EXECUTION.json --results-jsonl RESULTS.jsonl \
  --summary-poll POLL.json --output promotion-report.json
```

After reviewing a passing scientific-pilot report, create a manual approval:

```json
{
  "approved": true,
  "plan_fingerprint": "COPY_FROM_THE_FULL_EXECUTION_PLAN"
}
```

Full Slurm submission refuses approval files for any other fingerprint. No
command in this runbook automatically creates that approval.

## 8. Final aggregation

Prepare one JSONL row per model/seed with flattened numeric metrics, then run:

```bash
python -m tears_training.report aggregate \
  --metrics-jsonl final-seed-metrics.jsonl --baseline recvae \
  --output final-report.json
```

The output records mean, sample standard deviation, denominators supplied in
the input, and paired t/Wilcoxon tests against the baseline. W&B remains the
online experiment record; local reports and checkpoint hashes are the
reproduction record.
