# Recommendation quality repair — September 9, 2026

## Correction: missed frontend render filter

The subsequent user report exposed a gap in the verification below: it checked
API results and the served bundle, but not the actual recommendation cards.
`TearsApp` still passed the complete onboarding catalog into a second render-time
filter. The reported response contained twelve recommendations, while only seven
cards were displayed; ranks 1, 2, 9, 10 and 12 were hidden. The UI-render event
incorrectly recorded the pre-filter list. See
`../recommendation_render_fix_20260909/` for the exact response and follow-up fix.
The follow-up removes that second filter and tests the actual React component,
including visible card count/order, ranks, slider changes and persisted records.

## Adopted changes

The TEARS Base checkpoint, T5/LoRA encoder, latent projection, item decoder and
text-only inference architecture are unchanged. No retraining, alternative
recommender, title-specific boost, or LLM-generated movie ranking was introduced.

- New profiles use `shared-editable-profile-v2`: the validated visible text is
  the actual TEARS input. The server no longer generates a second, padded hidden
  summary. New-profile edits reach TEARS directly; old dual-view sources retain
  their saved backend and legacy synchronization behavior.
- The online prompt emphasizes distinctive, title-grounded content, cautious
  inference from sparse histories, and explicit dislikes. It uses low reasoning
  effort and a 2,000-token response budget (including reasoning). Word count is
  not a minimum-length gate. Frozen training prompts and summaries are unchanged.
- Online validation preserves supplied genres for a single positive film and
  explicit genre dislikes. It recognizes plural “adventures.” Neutral histories
  get a validated neutral sentence. A positive-only multi-film history can use a
  validated genre fallback after three rejected drafts; this is logged and does
  not claim detailed narrative fidelity.
- Candidate policy `all-years-selected-only-v2` makes older movies and unselected
  onboarding movies available to both TEARS and GERS. Selected movies and explicit
  request exclusions remain masked. Frontend filtering no longer hides unselected
  onboarding recommendations after the server returns them.
- Ranking version `tears-explicit-genre-alignment-v2` retains the existing genre
  tiers. Its parser recognizes additional positive phrasing and handles withdrawal
  language. Candidate/ranking versions are frozen in trials: old baselines and
  in-progress trials cannot silently cross the policy change.

## Concrete recommendation replay

The prior investigation reproduced Fellowship of the Ring → The Revenant,
Rogue One, Doctor Strange. With each of two freshly generated LOTR summaries and
the adopted all-years policy, the first two recommendations are:

1. The Return of the King
2. The Two Towers

The same summaries and checkpoint were also scored under the old candidate
restrictions, so this is a controlled candidate-policy comparison. These titles
are outputs of the unchanged model and genre policy, not hardcoded recommendations.
See `generated_rankings/results.json` for all 19 successful generated profiles,
both candidate policies, all ranking alternatives and complete top-twelve lists.

This does not establish broad satisfaction. For example, one generated romance
profile still ranks Shawshank Redemption and Forrest Gump first, while another
ranks Before Sunset and Before Sunrise first. Barbie and animation results also
retain a strong mainstream/popularity tendency. Wording sensitivity and coarse
genre metadata remain limitations of this checkpoint and serving policy.

## Harnesses and evidence

`scripts/evaluate_tears_summary_quality.py` runs the actual online evidence
classifier, prompt, retry loop and validators on ten fixed cases: LOTR, Barbie,
horror, romance, animation, explicit dislike, unwind context, neutral history,
weak negative evidence and a richer fantasy history. It writes exact generated
texts, validation attempts, source hashes and inputs for the recommendation
replay. Probe genre coverage is a separate diagnostic, not a relevance score.

The harness exposed a contradictory positive-only instruction that dropped an
explicit horror dislike, failures on neutral histories, loss of sparse genre
information, and a rich-history retry failure. These were repaired and tested.
The last full repeated run (`prompt_final/`) returned 19 valid profiles out of
20; the rich-history failure is preserved. After adding the validated fallback,
both fresh rich-history retests passed (`prompt_rich_recheck/`). The final frozen
corpus contains 20 profiles passing the final validators, with source lineage in
`verified_summary_corpus.json`. This is not a claim that a single fresh final
20-case run passed without retries, nor a guarantee of semantic quality.

`scripts/evaluate_tears_quality.py` evaluates 15 frozen diagnostic profiles and
128 randomly selected validation users (seed 20260909). Validation profiles use
their existing frozen summaries; relevance comes from held-out positive ratings.
Observed items are excluded. No test targets are read. The validation set is a
development sample, not an untouched final evaluation or a sparse-user A/B test.

Under the old 2015/onboarding filter, 103 of 128 sampled users have no eligible
held-out target. Under the all-years policy all 128 have eligible targets. Metrics
from the two candidate universes therefore have different denominators and must
not be presented as a percentage improvement from opening the catalog.

### Ranking experiments on the same all-years validation candidates

| Ranking | NDCG@12 | Recall@12 | NDCG@50 | Recall@50 |
|---|---:|---:|---:|---:|
| Raw TEARS | 0.153303 | 0.106753 | 0.187503 | 0.255461 |
| Original hard genre tiers | 0.131635 | 0.090766 | 0.165278 | 0.220923 |
| Adopted tiers with parser fixes | 0.132932 | 0.093059 | 0.165038 | 0.220243 |
| Experimental bounded coverage | 0.135279 | 0.088433 | 0.172069 | 0.236866 |
| Experimental bounded similarity | 0.132084 | 0.088112 | 0.166515 | 0.225712 |

Neither experimental soft correction consistently improved relevance and recall;
neither is used in serving. Raw scores do better on this validation sample but
still violate explicit preferences in several sparse probes. The adopted patch
does not claim that the genre tiers improve general held-out relevance. The
experiments remain in `scripts/tears_quality_policies.py`, separate from serving.

Catalog metadata itself is coarse: Rogue One has Adventure and Fantasy tags,
while Fantastic Beasts has only Fantasy. Counting matching tags cannot establish
high-fantasy or tonal similarity. Genre diagnostics must not replace relevance
or human review.

## Verification

- 94 selected backend tests passed, including shared-summary edits, explicit
  dislikes, sparse/neutral/fallback behavior, selected-item masking, and candidate
  policy changes blocking both new trials from old baselines and existing edits.
- All six frontend contract test files passed; the production build succeeded.
- Direct API-handler smoke tests using the actual TEARS and GERS checkpoints each
  returned twelve recommendations with canonical metadata and selected-item
  exclusions under the new candidate policy. They used a temporary study database.
- Syntax and diff-whitespace checks passed.

## Activated and publicly verified

Service job `10687616` was restarted onto `cn-m004`. `health_after.json` reports
the new summary, candidate and ranking policies and the exact local API source
hash. Its TEARS and GERS model identities equal `health_before.json` exactly.
The public index matches the tested frontend build byte for byte.

A synthetic public Fellowship of the Ring session generated a 52-word shared
summary on its first attempt. Its baseline top three were Return of the King,
The Two Towers, and The Matrix. An added horror dislike and a removal of fantasy
preferences each returned twelve recommendations. The durable-event audit
confirmed that display and encoded text match, edit patches are empty, the
selected movie remains excluded, the edited list has no Horror-tagged items,
and removing the fantasy text does not reintroduce it from ratings.

- Generation: `609967d466f7440580832522794aa541`
- Baseline: `29b11c118a934d6c81819c8195fa22ba`
- Added dislike: `606348cd8a8f468f94606890ccf7aa63`
- Removed preferences: `0eb98d5a62e04469b23649e84763e804`

`live_smoke.jsonl` and `live_audit.json` preserve these checks. These requests use
participant `synthetic-dual-summary-smoke` and must be excluded from participant
analyses. Existing browser tabs should reload and regenerate a summary to use
the new shared-profile flow; saved legacy summaries retain legacy behavior.

## Reproduction

From the repository root (model artifacts and cached tokenizer must be available):

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/evaluate_tears_quality.py --output /tmp/tears-quality --validation-users 128
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/evaluate_tears_summary_quality.py --output /tmp/tears-summaries --repeat 2
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/evaluate_tears_quality.py --output /tmp/tears-generated-rankings --validation-users 0 --probes /tmp/tears-summaries/generated_probes.json
```

The summary command requires the configured OpenAI API key and makes bounded
generation calls. It does not submit recommendations or write participant events.
The ranking command is local and caches recomputable logits; cache files are
excluded from version control. Generation failures cause a nonzero exit and are
preserved for inspection rather than silently removed from the success rate.
