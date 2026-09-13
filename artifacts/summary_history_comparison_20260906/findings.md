# Recovered summary pipeline: findings

## What was recovered

The August 25, 2026 online pipeline is preserved under `recovered/`, from commit `1754778339dbba31926abf5cfa8250b4d50d0f95`. Its system prompt requested four sentences: liked genres, liked themes/content, disliked genres/styles, and disliked plots/content. It prohibited titles, actors, years and numeric ratings and instructed abstention for unsupported preferences.

Unlike the current backend harness, it requested **120–260 words but only checked that the response was nonempty**. It did not enforce the requested word count or four-sentence structure, classify genre evidence, or append the current fixed abstention sentences. The recovered generation settings were `gpt-5-mini-2025-08-07`, minimal reasoning, low verbosity, a 450-output-token limit, and a JSON summary field. Fresh historical-pipeline samples in this experiment actually contained 38–111 words; restoring this version is not the same as enforcing a 140–180-word summary.

An actual August 26 saved summary was also recovered for Dune (2021) 5, Soul (2020) 5, and Everything Everywhere All at Once (2022) 4.5. No pre-September Soul-only saved summary was found. The Soul-only historical comparisons below are therefore newly generated samples using recovered settings, not the exact text served two weeks ago.

## Controlled comparison

Three fresh generations per pipeline were run for Soul, Barbie, and the mixed history. All 18 generation jobs succeeded. Current generation produced separate backend and display texts. The same frozen TEARS model scored all texts using the same per-case exclusions and release-year cutoff of 2015. The genre reranker was bypassed throughout. Saved summaries were also rescored under this current evaluation setup; these are not reconstructed historical recommendation lists.

For **Soul rated five stars**, the count of Animation-tagged titles among the top 12 was:

| Model input | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Recovered August 25 pipeline | 4 | 7 | 10 |
| Current full backend summary | 3 | 2 | 2 |
| Current short display summary | 6 | 4 | 6 |
| Current backend, removing only fixed negative abstentions | 5 | 2 | 3 |

Historical run 2 ranked Zootopia, Coco, Moana, Spider-Man: Into the Spider-Verse, and Finding Dory first. Current backend run 2 ranked Star Wars: The Force Awakens, Zootopia, The Hateful Eight, Blade Runner 2049, and The Revenant first. All samples and complete top-12 lists, including less favorable historical samples, are in `comparison.md`.

The exact saved backend summary associated with the reported Soul example produced **zero** Animation-tagged titles in its raw top 12, versus **three** when its saved display text was used directly. Removing only the backend's fixed two-sentence abstention suffix increased this to **one**. This isolates a wording effect, but does not establish that the suffix alone caused the poor results.

For **Barbie**, the recovered pipeline still produced mostly blockbuster-heavy lists: just 2 Comedy-tagged titles in each top 12, versus 2, 2, and 1 for the current backend. Removing the fixed abstention suffix did not consistently improve those results. Mixed-history results were also variable: historical Animation counts were 4, 3, 1, versus 1, 4, 1 for the current backend; the actual August 26 saved text scored 4.

## Interpretation and limits

- The current full-summary pipeline performs worse on this Soul diagnostic than the recovered lighter pipeline. Short display summaries also outperform current backend summaries on this diagnostic, so length alone is not a reliable explanation or solution.
- The current saved Soul backend spends much of its text describing uncertainty and unspecified dislikes, while its display text emphasizes introspection, purpose, and identity. The model therefore receives materially different preference information from what the user reads.
- Removing the exact fixed negative-abstention suffix preserves the positive text byte-for-byte but helps inconsistently. Other wording differences and the model's underlying ranking behavior remain relevant.
- The recovered pipeline does not solve Barbie, and three samples per history are not a broad quality evaluation. Genre-tag counts are descriptive diagnostics, not human relevance judgments, NDCG, or evidence of a general improvement.
- The live genre reranker is not a clean remedy: broad Adventure/Fantasy matches can also promote Star Wars for a Soul profile. It was excluded from this experiment rather than used to manufacture favorable results.

## Scope and next step

No live rollback, serving-policy change, or model retraining was performed for this comparison. The historical source, generated samples, exact settings and source hashes, 43 scored profiles, and repeatable comparison script are preserved.

A sensible next experiment is a switchable historical-summary baseline with raw TEARS ranking, tested across a broader fixed set of histories and explicit user edits. This should be compared before replacing the default; the current evidence supports investigating the lighter pipeline, not claiming that a rollback fixes recommendation quality generally.

Files: `experiment.json` (provenance/settings), `generated/` (samples), `rankings.json` (machine-readable rankings), `comparison.md` (all texts and lists), and `../../scripts/compare_historical_summaries.py` (reproduction).
