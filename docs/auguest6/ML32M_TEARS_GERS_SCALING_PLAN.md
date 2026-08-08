# ML-32M TEARS and GERS Scaling Plan

**Approved:** August 2026  
**Dataset:** MovieLens 32M  
**Purpose:** Research/publication with clear documentation and bounded completion time

## August 7 pilot-cohort amendment

The current scientific-pilot training is authorized to use the complete set of
9,763 summaries that passed the existing validator. This produces one matched
TEARS/GERS cohort with 8,787 train, 491 validation, and 485 test users. The 237
validation failures (213 train, 9 validation, and 15 test) are excluded from
this pilot only and move to the Phase 6 summary-completion work. They must be
regenerated, validated, and restored before five-seed full training.

This amendment does not change the fixed 180,948/10,000/10,000 global split,
catalog selection, model definitions, validation-only selection rules, test
isolation, or the requirement that TEARS and GERS use exactly the same users
within a training stage.

## 1. Objective and fixed decisions

Build a clean, reproducible training system for RecVAE, TEARS Base, GERS Base,
TEARS-RecVAE, and GERS-RecVAE. The system must support a quick run, a
scientific pilot, and a manually approved five-seed full run. Application/API
deployment is outside this phase.

- Full user split: 180,948 train, 10,000 validation, 10,000 test.
- Original pilot text cohort: 9,000 train, 500 validation, 500 test. For the
  amended current pilot, use the validated subset: 8,787/491/485.
- Pilot users are sampled deterministically and proportionally from activity
  bands 20–49, 50–99, 100–199, 200–499, and 500+.
- Validation/test history is chronological: earliest 70% observed, latest 30%
  held out.
- Original 0.5–5.0 ratings are RecVAE inputs; ratings >=4.0 are positive GERS
  and evaluation targets.
- Fixed users with no positive catalog target remain in the split but are
  omitted from undefined ranking rows; every denominator is reported.
- T5 backbone: pinned `google-t5/t5-base`; LoRA rank 64, alpha 16; latent size
  400.
- Paper alpha convention: alpha 0 is RecVAE-only and alpha 1 is
  text/genre-only.
- Seeds: 2020–2024; pilot and split seed: 2024.
- Checkpoint selection: NDCG@50 for Base models and mean NDCG@50 across alpha
  0, 0.5, and 1 for hybrids.
- W&B is required online under entity `niita-mila`, project `tears-ml32m`.
- Training completion, artifact integrity, and reproducibility are formal
  acceptance gates. Quality improvements are measured outcomes, not required
  pass conditions.

## 2. Storage and reproducibility contract

Use `$SCRATCH/datasets/ml-32m` as the immutable raw source. Create
`$SCRATCH/new_dataset/ml-32m` as a symlink to it; do not duplicate the existing
912 MB dataset and do not use SCP.

Store working artifacts below `$SCRATCH/FullTrainingTEARS`:

```text
manifests/                 immutable run/data/prompt manifests
datasets/                  smoke_10k_4k, pilot_10k, and full prepared data
summaries/requests/        uploaded or planned Batch JSONL
summaries/responses/       raw paid responses
summaries/errors/          raw provider errors
summaries/validated/       canonical validated summaries
summaries/controllability_edits/
checkpoints/<model>/       best, final, and one rolling resume checkpoint
metrics/                   per-epoch and final machine-readable metrics
logs/                      Slurm and local logs
wandb/                     W&B metadata and run identifiers
reports/                   quick, promotion, and final reports
```

Mirror all paid requests, responses, errors, validated summaries, and edits to
`~/FullTrainingTEARS_paid_artifacts`. Paid artifacts are append-only and keyed
by user, ordered-history hash, prompt hash, exact model snapshot, and generation
configuration.

Every run records the Git revision/dirty state, `uv.lock` hash, dataset and
split fingerprints, item mapping, prompt/model versions, hyperparameters,
seed, Slurm/GPU metadata, W&B run ID, token use/cost, metric denominators, and
checkpoint hashes.

## 3. Implementation interfaces

Implement a new `tears_training` package rather than repairing the incomplete
legacy research snapshot. It must provide idempotent, configurable commands:

```bash
python -m tears_training.data verify
python -m tears_training.data prepare --profile smoke|pilot|full
python -m tears_training.summaries plan --cohort pilot|full
python -m tears_training.summaries submit
python -m tears_training.summaries poll
python -m tears_training.summaries validate
python -m tears_training.train --model MODEL --profile PROFILE --seed SEED
python -m tears_training.evaluate --run-manifest MANIFEST
python -m tears_training.orchestrate --phase smoke|pilot|full
```

Every command supports configuration files and `--dry-run`, uses atomic writes,
refuses incompatible fingerprints, and resumes safely. Paths and credentials
must never be hardcoded.

## 4. Atomic execution phases

### Phase 0 — secure preflight

1. Revoke every OpenAI key exposed in notebooks or chat.
2. Supply a new project-scoped key only through `OPENAI_API_KEY`.
3. Add secret scanning and ensure keys never enter logs, W&B, manifests,
   checkpoints, notebooks, or Slurm scripts.
4. Extend the existing `.venv` with `uv`; preserve and verify W&B 0.28.1.
5. Verify W&B online access before GPU work.
6. Verify ML-32M against `checksums.txt` and persist schemas, counts, and hashes.
7. Create scratch/home artifact directories and the raw-data symlink.
8. Pre-cache and hash T5-base/tokenizer artifacts.
9. Pass unit/integration tests before the first GPU submission.

### Phase 1 — deterministic data preparation

1. Stream ratings in chunks; never materialize the full dense user-item matrix.
2. Stable-sort histories by `(timestamp, movieId)`.
3. Create the fixed activity-stratified user splits with seed 2024.
4. Split validation/test histories chronologically 70/30.
5. Build relevance targets only from held-out ratings >=4.0.
6. Count item support from training users only.
7. Create candidate catalogs with minimum train support 20, 50, and 200.
8. Convert MovieLens IDs to contiguous model IDs and preserve both mappings.
9. Treat `(no genres listed)` as `Unknown`; do not use tags.
10. Persist sparse CSR/memory-mapped inputs and immutable manifests.
11. Produce the activity-stratified 9,000/500/500 pilot cohort.
12. Produce a top-4,000 training-supported catalog view for the quick run.
13. Assert no user overlap, target leakage, or catalog leakage.

### Phase 2 — cached GPT-5 mini summaries

Use pinned `gpt-5-mini-2025-08-07` through Batch API with structured output and
minimal reasoning. For each user, prompt from at most the 50 most recent
observed movies, including private title/rating/genre evidence, and request an
approximately 200-word `Summary:` describing liked/disliked genres and themes
without movie titles, actors, years, or numeric rating commentary.

1. Write request JSONL before external submission.
2. Save raw response/error JSONL immediately after retrieval.
3. Validate JSON, nonempty output, 120–260 words, title/rating leakage, and
   exact cross-user duplication.
4. Retry validation failures at most twice; never synthesize silent fallbacks.
5. Progress through 100, 1,000, then 10,000 users.
6. Stop after the pilot cohort; the remaining summaries require manual
   promotion approval.
7. Before full generation, record the organization Batch queue limit from the
   Limits page.
8. Chunk waves to at most 50,000 requests, 200 MB, and the account token-queue
   limit.
9. Submit resumable sequential waves.
10. Refuse any wave whose actual spend plus projection and 15% reserve exceeds
    the hard $250 ceiling.

Expected GPT-5 mini Batch cost is about $76 for medium canonical summaries plus
$11–$25 for full controllability generation; the heavy scenario with reserve is
about $145–$175. The Batch API currently advertises a 50% discount and a
24-hour window, while GPT-5 mini standard text pricing is $0.25/M input and
$2/M output tokens.

### Phase 3 — quick 10k-user/4k-item run

1. Use the pilot cohort, top 4,000 items, BF16, and seed 2024.
2. Run one GPU per job; run two independent jobs concurrently when two GPUs are
   available.
3. Train RecVAE and all four Base/hybrid variants.
4. Run successive halving at total epochs 5, 20, and 60, retaining the best
   third at each rung.
5. RecVAE grid: dropout 0.1/0.2/0.4; LR 1e-3/1e-4/1e-5; gamma
   0.0035/0.004/0.005; weight decay 0/1e-5.
6. Base grid: dropout 0.1/0.2/0.4 and LR 1e-3/1e-4.
7. Hybrid grid: Base grid times OT weight 0.1/0.5/1.
8. Compare each finalist under fixed 200 epochs and maximum 300 epochs with a
   minimum 30, patience 20 early stopping.
9. Select schedules independently per variant. If validation scores differ by
   less than 1% relative, choose lower GPU-hours; otherwise choose higher
   NDCG@50.
10. Never use test metrics for tuning.
11. Publish chosen configurations, training curves, ranking/alpha metrics,
    resource measurements, and full-run duration projection.

### Phase 4 — full-catalog selection

1. With the quick-run RecVAE configuration, train 60-epoch full-training-user
   profiles for support thresholds 20, 50, and 200.
2. Stream validation scoring and record NDCG@20, coverage, throughput, epoch
   duration, and peak GPU memory.
3. Require peak memory below 72 GB/GPU.
4. Select the largest catalog within 5% relative of the best validation
   NDCG@20 that meets the memory gate.
5. Freeze its fingerprint before scientific-pilot or test work.

Raw reference counts are approximately 23,350/99.14% of ratings at support 20,
16,034/98.43% at 50, and 9,322/96.32% at 200. Final counts must use training
users only.

### Phase 5 — scientific pilot

1. Train seed-2024 RecVAE on all 180,948 training users and the selected
   catalog.
2. Freeze its encoder and initialize hybrid shared decoders from it.
3. Train TEARS Base, GERS Base, TEARS-RecVAE, and GERS-RecVAE on the amended
   matched 8,787-user training cohort; select on its 491 validation users and
   retain its 485 test users for the one post-selection pilot evaluation.
4. Train a separately labeled full-training-user GERS ceiling.
5. Select on the 500-user validation cohort and evaluate once on the untouched
   500-user test cohort.
6. Run pilot ranking and controllability evaluation.
7. Produce exact spent/remaining API cost, GPU throughput, projected five-seed
   duration, artifacts, metrics, and known failures.
8. Stop for explicit manual approval before remaining API calls or full jobs.

### Phase 6 — five-seed full training

After approval, first regenerate and validate the 237 pilot-summary failures,
then generate all other missing summaries and train every model for seeds
2020–2024 with the frozen catalog/configuration. Each hybrid uses its same-seed
RecVAE. Train hybrids at alpha 0.5; freeze the RecVAE encoder; keep the shared
decoder trainable; optimize recommendation reconstruction, OT alignment, and KL
with annealing to 0.5. Validate every epoch and keep best/final/resume artifacts.
Do not inspect full test results until all selection decisions are frozen.

### Phase 7 — full paper evaluation

1. Report Recall and NDCG at 20/50 for every model and seed.
2. Sweep alpha 0.0–1.0 by 0.1; select alpha on validation only.
3. Read the full test set once after selection.
4. Report mean, standard deviation, denominators, and paired significance tests
   for RecVAE vs each hybrid and TEARS-RecVAE vs GERS-RecVAE.
5. Run large-scope preference flips on all 10,000 test users.
6. Run three fine-grained edits for eligible rank-100–500 targets.
7. Run guided More/Less-genre tests and GERS proportional/upper-bound tests.
8. Cache every paid edit and publish rank-shift, genre-NDCG, alpha/control, and
   recommendation-quality plots.

## 5. Model fidelity

- TEARS Base maps T5 summary features to a Gaussian latent and shared item
  decoder.
- GERS Base maps normalized positive genre counts to a Gaussian latent and
  item decoder.
- Hybrids align editable and frozen RecVAE latent distributions through the
  paper's OT and KL objectives.
- Dense item logits exist only within a mini-batch.
- Observed items are always masked from recommendations.
- Do not introduce tags, sampled softmax, alternate backbones, EASE blending,
  or the later experimental text-alignment head.
- The new package follows paper alpha semantics even though legacy application
  files contain reversed conventions.

## 6. Slurm strategy

Shared directives:

```bash
#SBATCH --partition=long
#SBATCH --mem=256G
#SBATCH --time=96:00:00
#SBATCH -x 'cn-d[001-004], cn-b[001-005], cn-e[002-003]'
```

- Pilot: `--cpus-per-task=16`, `--gres=gpu:a100l:1`.
- Full 2-GPU: `--cpus-per-task=32`, `--gres=gpu:a100l:2`.
- Full 4-GPU: `--cpus-per-task=64`, `--gres=gpu:a100l:4`.
- Use single-node `torchrun`, deriving world size from Slurm.
- Prefer four GPUs. If pending solely for resources for six hours, replace the
  pending request with the two-GPU profile.
- Cap concurrency at four 4-GPU or eight 2-GPU jobs.
- Use job dependencies, per-epoch rolling checkpoints, a pre-timeout signal,
  and resumable resubmission.
- W&B unavailability fails the online-only job, but local resume state remains.

## 7. Verification and acceptance

Unit tests cover splits, leakage, support filtering, deterministic sampling,
rating semantics, genres, sparse batches, alpha direction, losses, cache keys,
cost gates, and checkpoint fingerprints. Integration tests cover ML-1M shape
parity, forward/backward for every model, 1-vs-2-GPU DDP parity, checkpoint
resume, W&B creation, non-paying Batch dry-run, observed-item masking, test-set
isolation, and paid-artifact backup hashes.

The quick phase is accepted when all models produce finite training, resumable
artifacts, metrics, W&B runs, and full-run projections. The scientific pilot is
accepted when the full-population RecVAE, matched four-model cohort, full GERS
ceiling, evaluation, and promotion report complete below the cost ceiling. The
full phase is accepted when every five-seed run and full paper evaluation is
complete, checksummed, backed up, synchronized to W&B, and documented with
reproduction commands.

Target timing is 3–5 calendar days for the quick/scientific pilot and two weeks
for the full phase after manual approval, subject to measured throughput and
Mila queue time.
