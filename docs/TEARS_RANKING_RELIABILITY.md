# TEARS online genre-alignment correction

Current repair: [September 10 refresh reliability and compound preferences](RECOMMENDATION_REFRESH_RELIABILITY.md).
Serving uses `tears-compound-genre-alignment-v3`; the sections below record prior policies.

## September 9 quality repair

Serving now uses `tears-explicit-genre-alignment-v2`, which keeps the existing
genre tiers and TEARS logits within each tier. Its online parser additionally
recognizes “responds positively to,” “interested in,” and plural “adventures,”
and does not interpret “no longer enjoy” as a current like.

Two proposed soft rerankers were evaluated and rejected because they did not
consistently improve held-out relevance and recall. Their implementations live
in `scripts/tears_quality_policies.py` for offline reproduction only.

The adopted changes are the shared editable summary described in
`docs/DUAL_SUMMARY_PIPELINE.md` and candidate policy
`all-years-selected-only-v2`: older movies and unselected onboarding titles are
available to both TEARS and GERS. Checkpoints, model architecture, latent vectors,
and raw logits are unchanged. New candidate and ranking versions require new
study baselines; old trials cannot silently cross policies.

Reproducible validation, generation failures and successful repairs are recorded
in `artifacts/recommendation_quality_fix_20260909/`. The remainder of this
document describes the preceding v1 deployment and its historical checks.

## Diagnosis

The reported Barbie request (`request-94708077-28f7-4681-86b5-9f6808cd0088`) used its full backend profile, not stale or ignored display text. The raw model reproduced the reported Star Wars / Rogue One / Logan ranking exactly. A neutral four-sentence profile also ranked many of these titles highly. Positive comedy and romance profiles continued to rank action franchises near the top.

TEARS Base is a text-only model; selected movie IDs mask seen titles, while ratings affect the generated profile. The checkpoint and catalog mapping loaded correctly. These controlled comparisons indicate a strong general item preference in the model's raw scores, not an API failure to pass the summary. Replacing the full profile with the short display did not solve the ranking mismatch. Full subtraction of neutral-profile logits was rejected after a science-fiction probe promoted unrelated romance titles.

## Online policy

`tears-explicit-genre-alignment-v1` extracts explicit positive and negative genre mentions from the current backend summary, including tentative positive wording. Unsupported abstention clauses, mixed claims, and conflicting genre directions do not create constraints. A removed or reversed preference is therefore reflected through the synchronized backend, not reintroduced from the original ratings.

Candidates are ordered in four groups: preferred without an avoided genre; neither preferred nor avoided; both preferred and avoided; avoided without a preferred genre. TEARS raw scores determine order inside each group. The implementation adds an offset larger than the finite score range per priority tier. Scores represent this composite ordering, not probabilities. Original logits are retained as `raw_model_score` on returned items. Existing year, selected-title and onboarding exclusions still apply.

No movie title is hardcoded into serving. The policy uses the existing MovieLens genre tags and may miss nuances or synonyms that its conservative parser does not recognize. It aligns broad genres, not every theme or tonal preference, and does not ensure every comedy resembles Barbie. The checkpoint is unchanged; this is a new serving policy, not a retrained TEARS model or a new offline NDCG result.

## Controlled results

Using the exact reported backend profile and exclusions, the new top twelve were Zootopia, Knives Out, Deadpool 2, Moana, The Man from U.N.C.L.E., The Intern, La La Land, Once Upon a Time in Hollywood, Green Book, Finding Dory, Spy, and Now You See Me 2. All have a Comedy tag in the serving catalog; the old list had only Zootopia with that tag.

The horror probe shifted to Get Out, It, Split, The Conjuring 2, A Quiet Place, and other horror films. The science-fiction/action probe retained Star Wars and related action titles. The neutral probe's ordering was unchanged. `scripts/probe_tears_personalization.py` reproduces the comparisons using the saved reported request and synthetic contrasting profiles.

The deployment manifest, health endpoint, effective-model-input log, and immutable TEARS trial input record the policy version. Old raw-ranking baselines cannot be mixed with this policy in an edit trial; a fresh baseline is required. Earlier frozen-model evaluation metrics do not describe this reranked serving policy.

## Public verification

The exact saved Barbie backend was replayed through the public API under synthetic request `a08883cc157c4f1fb928eafd855165fe`; it returned the twelve genre-aligned titles above. Horror request `e0d748c2975f46efb205dba64fc6f5aa` returned twelve Horror-tagged titles, while science-fiction request `1b9c4b0165d34c4089eeb83a2ea393fa` retained its action/science-fiction ranking. Synthetic participant `synthetic-ranking-probe` must be excluded from participant analyses.

Fifty-nine selected backend tests, all six frontend contract test files, and the production build passed. GPU job `10687318` and CPU standby `10687616` serve the policy; the superseded standby was stopped. This verifies the reported genre mismatch and contrasting preference behavior, not overall held-out recommendation quality.
