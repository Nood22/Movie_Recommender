# Recommendation quality investigation — September 9, 2026

## Verified serving models

The public `/pilot/api/health` endpoint was read during this investigation.
It reports TEARS Base (`tears_base`), seed 2022, 200 training epochs,
checkpoint `10d017146b22/best.pt`, SHA-256
`08f2609c6dae073a1d6c2a8ba14a3121c5ff4ec807201c7f97c7d4aa3c598e2b`.
The encoder is LoRA-adapted `google-t5/t5-base`, with a learned latent projection
and item decoder. GPT `gpt-5-mini-2025-08-07` generates the taste summaries;
it does not produce the movie ranking. GERS uses `gers_recvae`, seed 2022.
The active TEARS ranking policy is `tears-explicit-genre-alignment-v1`.
The service currently runs on CPU; no evidence ties this to relevance problems.
The root README's EASE description refers to a separate older service.

## Fresh controlled reproduction

The latest logged TEARS recommendation at investigation time was recorded at
2026-09-09T19:49:56.916Z, request
`request-4e7ed11f-1c60-4976-89cd-ee98012d2fd0`.
Its evidence was Fellowship of the Ring rated five stars.
Local CPU inference reproduced the live ordered top twelve exactly.

The same frozen checkpoint was scored with two saved texts, with and without
the genre reranker, under three candidate policies (12 variants total).
No new summaries were generated. No recommendations were submitted to the API.

| Text / ranking | Candidate policy | First three results |
|---|---|---|
| Private backend / live reranker | Current | The Revenant; Rogue One; Doctor Strange |
| Displayed profile / live reranker | Current | Rogue One; Doctor Strange; Fantastic Beasts |
| Private backend / live reranker | Exclude selected titles only, all years | A New Hope; Jurassic Park; Mission: Impossible |
| Displayed profile / live reranker | Exclude selected titles only, all years | Return of the King; The Two Towers; A New Hope |
| Displayed profile / raw scores | Exclude selected titles only, all years | Return of the King; The Two Towers; A New Hope |

These examples diagnose this request; they do not measure general relevance.

## Contributing mechanisms

1. **The hidden summary loses relevant information.** The displayed LOTR profile
   describes high fantasy, epic journeys, ensemble quests and mythic elements.
   The backend retains Adventure/Fantasy but declines to infer plot/theme
   preferences and spends three sentences on unsupported preferences.
   `tears_summary_views.py` explicitly requires 140–180 words even for sparse
   histories and instructs expansion of uncertainty explanations. Across the
   seven logged TEARS recommendation events since September 7, about 73% of
   backend words occur in sentences containing “no strong preference” (a simple
   diagnostic heuristic, not a semantic quality metric). Two have no extracted
   positive or negative genre direction.

2. **The genre correction is coarse and dominates the learned scores.**
   `align_scores` takes the union of preferred genres, with a bonus larger than
   the entire raw score range. Matching Adventure alone receives the same bonus
   as matching Adventure and Fantasy together. That explains why The Revenant
   can top the LOTR list. The policy cannot distinguish high fantasy from broad
   action/adventure or enforce tonal similarity.

3. **Candidate restrictions remove relevant choices.** The frontend excludes
   all 100 onboarding titles, not just selected titles, and the service requires
   release year >=2015. Of 22,343 catalog items, 4,584 pass the year cutoff;
   4,544 remain for this request after exclusions. Both LOTR sequels are excluded
   by year and are also in the onboarding list. Removing restrictions alone
   does not repair the backend-summary ranking, as the ablation demonstrates.

4. **TEARS Base has no direct rating-history input.** In `recommend_tears`, movie
   IDs only mask candidates. Ratings influence summary generation but are not
   supplied to this model. The alpha=0.5 field is a compatibility value with no
   effect on TEARS Base. Information lost by summarization cannot be recovered
   from a collaborative branch. GERS does have such a branch.

5. **Raw rankings have a persistent blockbuster tendency.** Fresh execution of
   `scripts/probe_tears_personalization.py` reproduced blockbuster-heavy raw
   results for neutral, comedy, romance and Barbie texts. For example, the raw
   explicit-comedy profile starts with The Force Awakens and The Hateful Eight.
   Genre reranking improves tag compliance but does not establish nuanced
   preference matching. The September 6 historical-summary comparison also
   found wording sensitivity and showed that reverting summaries alone did not
   consistently solve Barbie.

## Model comparison caveat

All 25 full-training runs are complete. The Phase 6 finalization report gives
mean validation selection NDCG@50 of 0.184980 for TEARS Base, 0.216634 for
TEARS-RecVAE and 0.234617 for RecVAE. The hybrid selection score averages
alpha=0, 0.5 and 1; it is **not** a metric for one deployable blend. These numbers
motivate comparison but do not establish an equivalent percentage improvement
for the website, its sparse histories, reranking policy or filtered catalog.
No reserved test data were read for this investigation.

## Recommended next work

Evaluate concise, grounded model input against the current backend text on a
fixed validation set of sparse histories and edits. Compare candidate policies
explicitly, including retaining unselected onboarding movies and allowing older
movies. Evaluate TEARS-RecVAE at individual alpha settings with properly encoded
positive histories, measuring both relevance and responsiveness to edits.
Treat the current genre tier policy as a separate experimental factor.
Select changes on the combined website behavior, not genre counts alone.

No serving code, model checkpoint, study policy or deployment was changed.

## Artifacts and reproduction

- `latest_request_ablation.json`: saved texts, model identity and all 12 rankings.
- `contrasting_profiles.jsonl`: seven profiles with raw and reranked top 12.
- `replay_latest.py`: read-only model/log replay; run from the repository root
  with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python
  artifacts/recommendation_quality_20260909/replay_latest.py`. It selects the
  latest logged recommendation at execution time, so later executions may use
  a different request. The JSON artifact preserves this investigation's request.
- Related source: `pilot_recommender.py`, `tears_preference_ranking.py`,
  `tears_summary_views.py`, `movie-recommender-pilot/src/components/TearsApp.jsx`.
- Existing evidence: `artifacts/summary_history_comparison_20260906/findings.md`
  and `docs/august20/PHASE6_FINALIZATION_REPORT.md`.
