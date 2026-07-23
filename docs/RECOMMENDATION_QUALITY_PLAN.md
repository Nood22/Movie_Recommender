# Recommendation Quality Improvement Plan

## Objective

Increase relevance of top movie recommendations while preserving user control,
catalog coverage, diversity, reproducibility, and acceptable online latency.

## Definition of success

An improvement is accepted only if it is evaluated on the same fixed MovieLens
split and meets all of the following:

1. Higher NDCG@10 and Recall@10 than the current baseline.
2. No material regression in catalog coverage or genre diversity.
3. Deterministic rankings for identical inputs and artifacts.
4. Explicitly disliked genres are excluded.
5. Seen/liked titles are not returned unless the caller requests repeats.
6. The online scorer has a documented, versioned feature contract.

Initial target: at least 10% relative NDCG@10 improvement over the reproducible
baseline. The target will be revisited after measuring baseline variance.

## Phase 0: measurement foundation

- Build one evaluator with fixed chronological leave-one-out splits.
- Record Recall@K, NDCG@K, MRR@K, coverage, popularity, diversity, and latency.
- Add deterministic seeds and configuration files.
- Save machine-readable result JSON and a human-readable comparison table.
- Add small tests for title mapping, genre filtering, seen-item filtering, and
  deterministic ranking.

## Phase 1: repair correctness and reproducibility

- Replace absolute paths with paths relative to the repository or environment
  overrides.
- Validate nonempty requests, `top_k`, and `alpha`.
- Parse MovieLens genres using `|`.
- Treat liked titles as profile evidence and exclude them from results.
- Remove the runtime-created random projection.
- Version every learned or fitted artifact with its training configuration.

Expected effect: reliable behavior and a trustworthy baseline, even before a
more sophisticated model is introduced.

## Phase 2: strong deterministic hybrid baseline

Implement a lightweight hybrid recommender using only training interactions:

- Collaborative signal: item-item cosine/co-occurrence similarity from positive
  MovieLens interactions.
- Content signal: normalized multi-hot movie genre vectors and genres detected
  in the user description.
- Profile signal: aggregate vectors of liked/history movies.
- Popularity prior: smoothed and capped so cold profiles remain useful without
  collapsing all recommendations to blockbusters.
- Final score: tuned weighted combination of collaborative, content, and
  popularity components.

Tune weights on a validation-user subset, then evaluate once on the fixed test
users. Do not tune against the final test result.

## Phase 3: learned text-to-recommender alignment

After the deterministic hybrid establishes a strong floor:

- Generate or use existing user taste summaries from training histories only.
- Encode summaries with a frozen sentence/T5 encoder.
- Train a small projection or two-tower objective so text representations align
  with positive item representations.
- Compare pairwise/BPR, sampled-softmax, contrastive, and optimal-transport
  alignment objectives.
- Use hard negatives and popularity-aware negative sampling.
- Save the trained projection; never create it randomly during inference.

This phase is the likely Slurm workload because text encoding and hyperparameter
search benefit from GPUs and parallel jobs.

## Phase 4: reranking and user control

- Add maximal marginal relevance or xQuAD-style reranking for diversity.
- Penalize near-duplicate franchises when appropriate.
- Calibrate `alpha` as a real text-versus-history control.
- Enforce disliked genres as hard constraints.
- Support soft negative preferences separately from hard exclusions.
- Return score contributions for debugging: collaborative, text/content,
  popularity, and reranking adjustment.

## Phase 5: production validation

- Test cold-start, sparse-history, typo, empty-context, and conflicting-feedback
  scenarios.
- Measure CPU/GPU latency and memory.
- Add artifact checksums and startup contract validation.
- Shadow the improved ranker before switching defaults.
- Once user feedback exists, add online measures: saves/clicks, completion,
  explicit satisfaction, and hide/dislike rate.

## Experiment order

| Order | Experiment | Main question |
|---:|---|---|
| 1 | Seeded current ranker | How weak/variable is the random projection? |
| 2 | Popularity baseline | What is the non-personalized floor? |
| 3 | Genre-content profile | How much does explicit taste help? |
| 4 | Item-item collaborative | How much does history similarity help? |
| 5 | Weighted hybrid | Which combination generalizes best? |
| 6 | Diversity reranking | Can coverage/diversity improve without relevance loss? |
| 7 | Learned text projection | Does semantic text add value beyond genre/history? |
| 8 | OT/contrastive variants | Which alignment objective is best? |

## Slurm policy

Use local CPU runs for data validation, tests, popularity/content baselines, and
small collaborative experiments. Submit Slurm jobs for transformer embedding,
projection training, and parameter sweeps only after a local smoke test passes.
Each job must write configuration, logs, metrics, checkpoint path, environment,
and job ID into a unique run directory.

## Result table

The evaluator will maintain this table in `results/README.md`:

| Run | Model | Recall@10 | NDCG@10 | MRR@10 | Coverage | Diversity | ms/user |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026-07-16 | Popularity | 0.0411 | 0.0201 | 0.0138 | 0.0302 | 0.7787 | 1.43* |
| 2026-07-16 | Genre/content | 0.0100 | 0.0044 | 0.0028 | 0.0753 | 0.6803 | 1.43* |
| 2026-07-16 | Item-item CF | 0.0598 | 0.0298 | 0.0208 | 0.0872 | 0.7758 | 1.43* |
| 2026-07-16 | Initial hybrid | 0.0581 | 0.0288 | 0.0200 | 0.0866 | 0.7695 | 1.43* |
| 2026-07-16 | **EASE (accepted)** | **0.1046** | **0.0510** | **0.0351** | **0.2142** | 0.7512 | 0.17* |
| 2026-07-16 | EASE + MMR (not enabled) | 0.1046 | 0.0510 | 0.0351 | 0.2142 | 0.7512 | 0.06* |
| 2026-07-16 | Learned T5 head (not enabled) | 0.0461 | 0.0227 | 0.0157 | 0.0262 | 0.7784 | offline |
| 2026-07-16 | EASE + text (weight selected: 0) | 0.1046 | 0.0510 | 0.0351 | 0.2142 | 0.7512 | offline |

`*` Batch scoring time per user; excludes one-time fitting and HTTP overhead.
