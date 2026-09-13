# Recommendation refresh reliability — September 10, 2026

## Diagnosis

The exact Soul requests `request-0dfcbc16-7e48-457c-8e3f-236afb0c8f1a` and
`request-22a77bba-5da4-4378-b62a-f2d4590e4836` reproduce the two reported lists.
Both selected Soul (2020), MovieLens ID 225173, at five stars, encoded the visible
summary, and extracted the same five preferred genres. Their generated wording
differed. The report also mentions Up; its corresponding logs select ID 68954.

A fresh-checkpoint A/B/A replay (Soul, unrelated crime/horror, identical Soul)
produced bitwise-identical logits and recommendations: maximum absolute logit
difference 0.0. Every model module was in evaluation mode. A warm-up request does
not improve the ranking; regenerating the summary changes a wording-sensitive
model input.

The v2 policy gave full preferred-tier credit for any one matching genre.
Adventure or Fantasy alone let highly scored mainstream movies compete equally
with family animation. There is no title-specific boost requiring Lord of the
Rings, Star Wars, or any other franchise.

## Implemented repair

`tears-compound-genre-alignment-v3` preserves explicit combinations such as
animated family films and romantic comedy. Complete matches precede partial
matches; raw TEARS logits still order candidates within a tier. Explicit dislikes
remain stronger than positive combinations. Independently stated genre interests
outside a combination's scope disable that constraint. Withdrawn, conflicting,
uncertain, and alternative claims do not create it. The parser also recognizes
“respond to” and “open to,” which were missed in some saved requests.

Validated generation is reused for identical evidence within one participant,
including after refresh or another request. The persistent cache key includes
the full evidence, ratings, dislikes, context, model, prompt, and API source hash.
Participants do not share entries. Edits remain direct model input and are never
replaced with the cached original. Reuse creates a new session-owned summary
event and records the original generation request. Concurrent generations
converge on the first saved validated summary.

Private generation evidence retains the release year to distinguish titles;
the output validator still rejects title/year leakage. The sparse fallback
expresses Animation plus Children as “animated, family-friendly films,” retaining
their combined scope. It does not invent narrative details. Checkpoints, latent
representations, and the TEARS text-only architecture are unchanged.

The v3 policy is recorded in deployment metadata and immutable trials. A v2
baseline cannot silently cross into a v3 edit trial.

## Catalog findings

Candidates remain all 22,343 trained MovieLens items, minus selected or explicitly
excluded movies. There is no restriction or bonus for membership in the 100-card
onboarding list. The cards were themselves selected for popularity, so overlap
with a popularity-biased model is unsurprising. Two and ten movies in the
originally reported lists were already outside those cards.

The decoder cannot score movies outside its trained item vocabulary. Supporting
newer/out-of-vocabulary movies requires additional retrieval or retraining; this
repair does not claim that capability or arbitrarily exclude unselected cards.

## Measurements and limits

Both exact Soul summaries now return twelve Animation-and-Children-tagged films,
with neither Lord of the Rings nor Star Wars. The prior counts were nine and two.
Four and six recommendations, respectively, are outside the onboarding cards.
The refreshed example begins with Toy Story, The Lion King, Monsters, Inc., Shrek,
Aladdin, and Finding Nemo. There is still variation between distinct summaries.

| Same 128 validation users and candidates | v2 | v3 |
|---|---:|---:|
| NDCG@12 | 0.132932 | 0.134067 |
| Recall@12 | 0.093059 | 0.093939 |
| NDCG@50 | 0.165038 | 0.165354 |
| Recall@50 | 0.220243 | 0.220334 |

Paired bootstrap 95% intervals for these changes include zero. This supports the
targeted alignment repair, not a statistically established overall improvement.
Global genre-count, majority-match, and soft-coverage alternatives were rejected
after worsening validation relevance. No reserved-test targets were used.

The first prompt trial produced three generic fallbacks in four cases and was
revised. The final four fresh Soul/Up checks produced three generated profiles
and one validated fallback. All four preserve the combined preference and were
replayed through the actual checkpoint. Narrative specificity still varies;
the conservative parser and coarse genre metadata do not fully model themes,
tone, maturity, or franchise diversity. No blanket franchise ban was introduced.

107 focused backend tests, six frontend contract files, both React component
tests, the production build, and diff whitespace checks passed.

## Evidence and reproduction

All measurements are under `artifacts/recommendation_reliability_20260910/`.
`coverage_comparison.json` includes complete replay rankings, catalog overlap,
validation metrics, paired confidence intervals, and source hashes. Rejected
generation trials and their fallback use are retained alongside the final run.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/check_tears_request_independence.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python scripts/evaluate_tears_genre_coverage.py
```

The coverage script verifies cached-logit keys against the exact texts and model
identity. `scripts/smoke_tears_refresh.py --api BACKEND_URL` additionally checks
deployed replay, generation reuse, and GERS. Its synthetic participants start
with `synthetic-refresh-reliability-` and must be excluded from study analyses.

## Deployment verification

Backend job `10687616` was restarted on `cn-m003`. Its startup API source hash
matches the tested source, and its health endpoint reports v3. Live replay of
both exact reported summaries passed the twelve-family-animation check. A live
Soul generation followed by identical evidence returned the same text with zero
generation attempts on reuse and a fresh summary receipt. GERS also returned
twelve results. `live_smoke.json` and `live_event_audit.json` preserve the synthetic
requests and confirm that the encoded summary equals the visible summary.

No public gateway job was running when checked. The existing one-CPU/two-GB
gateway was restored as job `10745329` on `cn-h004`. Through its local proxy, the
API reports v3 and the served index matches the tested production build byte
for byte. Tailscale reports Running, online, and no health errors. However,
`tearsgersab.tail9d6ed0.ts.net` still does not resolve from this environment;
external public availability could not be verified. This remains a deployment
limitation separate from the tested backend behavior.

## Follow-up: release-year bias

A direct audit of all 28,528,213 training interactions confirms a strong
historical exposure imbalance. Pre-2015 movies are 79.17% of eligible titles but
receive 93.82% of training ratings and 93.83% of the summed rating weight.
2020–2023 movies are 5.42% of eligible titles but receive only 0.79% of training
ratings. Toy Story has 62,077 training ratings, Finding Nemo 41,553, Inside Out
19,433, and Soul 4,040. See `release_year_audit.json` in the evidence directory.

The training loader supplies the original rating vectors to a multinomial
reconstruction objective; there is no time decay or inverse-popularity weighting
in that objective. Frequent historical items therefore have much more aggregate
training weight. This is evidence for learned historical-popularity bias, not a
controlled causal estimate of each component's contribution.

The minimum-20-training-ratings catalog rule additionally excludes titles with
little exposure. The local dataset ends on October 12, 2023 and contains no
2024–2026 releases. No serving rule rewards an old release year, but the v3
genre correction preserves raw model order within preference groups and supplies
no freshness adjustment. Removing the preceding 2015 eligibility cutoff made
the existing bias more visible. Both repaired reported lists still contain only
pre-2015 titles, with median release years 2003.5 and 2000 respectively.

Finally, selecting a recent movie does not provide a release-era feature to the
text-only recommender: selected IDs exclude seen movies, while the generated
profile intentionally omits years. The refresh/genre repair does not constitute
a fix for this separate popularity and release-era issue. No ranking or training
settings were changed during this follow-up audit.
