# TEARS summary views

September 10 update: identical evidence now reuses validated generation within
the participant, including across refreshes. Edits still reach the model verbatim.
See [the refresh reliability report](RECOMMENDATION_REFRESH_RELIABILITY.md) for
cache scope, prompt/fallback changes, ranking measurements, and limitations.

## September 9 quality repair — shared editable profile

New summaries now use `shared-editable-profile-v2`: one evidence-proportional,
validated profile is both displayed and encoded by TEARS. The prompt puts
distinctive grounded content first, retains explicit dislikes, and permits
tentative title-grounded detail for a sparse history. It does not pad unsupported
slots. The frozen offline training prompt and summary artifacts are unchanged.

The server still stores `backend_summary` for log compatibility, but it equals
`representation` for new sources. Edits to a v2 source are passed to TEARS
verbatim after the existing transport checks, without a second model rewriting
them. Session ownership checks still apply. Existing dual-view sources continue
using their saved backend and the legacy patch synchronization path below.

Explicit genre dislikes are added to the online evidence contract as required
negative coverage; a positive-only rendering rule cannot suppress them. Entirely
neutral histories without explicit preferences or context get a deterministic,
validated neutral sentence. Neither case requires invented preference content.

The health endpoint and summary events record the new policy and actual prompt
hash. Local quality evidence and reproduction commands are in
`artifacts/recommendation_quality_fix_20260909/`. The remainder of this document
records the previous dual-view implementation and its historical deployments.

The website uses two representations (September 2026):

- The recommendation profile restores the system prompt from `tears_training/summaries.py`, with four sentences and a 140–180 word validation gate. Unsupported slots explicitly abstain. This profile is stored in the server's summary-result event.
- The editable display uses the existing `ONLINE_TASK1A_SYSTEM_PROMPT`: evidence-proportional length, silent omission of unsupported slots, and grounded positive and negative content. It is generated from separated evidence, without the backend's abstention prose in its prompt. Supported dislikes are not intentionally removed.
- The browser submits the display and a signed source request ID. The server retrieves the full profile belonging to that participant/session; the browser does not author the original full profile.

## Historical harness audit

The quoted original system prompt is present verbatim in `tears_training/summaries.py`. Word bounds come from configuration, not that system prompt. Its validator checks sentence count, prefix, third person, titles, years and some rating language; semantic ordering and theme grounding are prompt instructions, not comprehensive machine-verifiable guarantees.

The later V11 evidence harness (`tears_training/evidence_gated_summary_harness_v11.py`) uses ratings >=4 as positive and <=2.5 as negative, exclusive positive/negative/mixed/insufficient genre classifications, and NONE/WEAK/STRONG negative evidence. Its own prompt explicitly does **not** force four sentences or exact length. Thus the proposed harness list combines different historical versions.

Online serving consistently uses >=4 / <=2, normalizing intermediate and missing ratings to neutral before calling the V11 evidence classifier. The restored backend combines the original four-slot format with that online grounding contract, required positive/negative genre coverage, and stronger numeric/rating leakage validation. It does not change frozen training artifacts. For sparse histories, the full profile explains uncertainty to meet the requested length; the UI omits these abstentions.

When negative evidence is NONE and there are no explicit dislikes, the backend model returns only the two positive slots under a structured schema. The server appends two fixed abstention sentences and validates the complete four-sentence profile. WEAK, STRONG, and explicit-dislike cases continue generating their negative sections. This prevents fabricated-dislike checks from rejecting elaborate model-generated explanations of missing dislikes.

If three drafts fail for a single positive movie without negative evidence or viewing context, a conservative genre-only fallback is available for each view. It uses the supplied genres with tentative wording, makes no title-derived content claims, and passes the same applicable validators before returning. The backend fallback still has four sentences and 140–180 words. These fallbacks are logged as `validated_sparse_fallback`; validator failures are not waived.

## Editing

Each recommendation compares the entire current display against its saved original display. A model proposes exact, unique, non-overlapping substring patches to the original backend profile. Deletions withdraw matching claims without inferring their opposite; replacements update corresponding claims; explicit new preferences can append. Unrelated backend text is preserved byte for byte. The full cumulative edit is used, so successive changes do not restore deleted preferences from ratings.

The server validates patch addresses and size, records the patches and actual tokenized model input, and reuses completed synchronization results for identical display edits within the same session and source. Invalid patches receive up to three attempts with repair feedback, then fail rather than silently falling back to the unedited profile. Unicode is supplied directly instead of as escaped character sequences. Edit mapping uses medium reasoning, and full-profile generation uses low reasoning. Participant edits are not forced into generated-profile word/sentence bounds. Semantic matching remains model based and can make mistakes; this is not a deterministic proof of semantic equivalence.

Old independently authored profiles without a source ID remain usable verbatim. Source IDs are included in request signatures and frozen trial inputs; changing the hidden baseline mid-trial is disallowed. Existing offline study clients must carry the source ID to opt into dual views.

## Serving and verification

`scripts/run_pilot_api.sh` handles compute nodes missing the system Python used by `.venv`. `slurm/tears_pilot_api_cpu.sbatch` runs the same trained models on CPU under the gateway discovery name, providing an alternative to the GPU allocation. Both jobs retain the existing timeout/requeue behavior.

`scripts/smoke_dual_summary.py --api URL/api` explicitly runs a live synthetic Barbie generation, baseline recommendation, and edited recommendation. It records these under `synthetic-dual-summary-smoke`, which must be excluded from participant analyses. `--generation-only` tests sparse generation without running recommendations.

## Verified deployment, September 6, 2026

The final public smoke test generated a 169-word, four-sentence backend and a 53-word display. Baseline, added-preference and removed-preference requests each returned twelve recommendations. The audit confirmed that the baseline encoded the full backend verbatim and that removing the comedy preference removed comedy/comedic/humor claims from the effective backend.

- Generation request: `871b04942e294dc591707950ac40c0c8`
- Baseline: `59c3b6f681484d8a8807e220728b973e`
- Add horror dislike and friendship preference: `8c879da9d2d849baa4386333b54ae680`
- Replace visible preferences with friendship: `9baac98947b84ea0bf2ce3dd5a958ad2`

GPU job `10687318` and CPU standby `10687427` served matching final source/module hashes. Superseded CPU jobs were stopped. Verification passed: 53 selected Python tests, all six frontend contract test files, the production frontend build, syntax checks, and diff whitespace checks. These checks establish working generation and synchronization; they do not measure a recommendation-quality improvement.

### Follow-up reliability fix

Subsequent participant Barbie requests exhausted retries on fabricated-dislike and rating-language checks. The NONE-evidence structured slots, fixed abstentions, separated display input, and validated sparse fallbacks described above address this failure path. Forty-eight relevant tests passed, including forced exhaustion of all three backend and display drafts and protection of WEAK/STRONG/explicit dislikes.

Three independent public Barbie generations then succeeded with backend word counts 155, 154, and 157, each four sentences and each passing backend validation on the first attempt. Request IDs: `ecae25c9ed544f1983dd1f06bcbf77f3`, `b587a59d4e4b4625a123d7f30dc838d0`, `6b40497539074c3cb491f4ea8a31d185`. Baseline (`480d04d21a9d46f69f30fa716f199735`), addition (`45d1cde588544b1697b06fa02d485ffc`), and removal (`986a01015ea84dea81dbdc7c03595541`) each returned twelve recommendations. GPU `10687318` and replacement CPU standby `10687449` were verified against the fixed module hash; the old CPU standby was stopped.
