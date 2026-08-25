# Canonical TEARS/GERS Human-Study Task Contract

| Field | Value |
| --- | --- |
| Protocol ID | `tears-gers-human-study-tasks` |
| Version | `1.0.0` |
| Effective date | `2026-08-25` |
| Status | Canonical |
| Applies to | TEARS and GERS participant-facing study flows |

## Authority and scope

This document is the semantic source of truth for the meanings of Tasks 1a, 1b,
2, 3, and 4. UI screen order, route names, implementation comments, and legacy
task labels do not redefine these tasks.

This version defines task boundaries, participant actions, rank semantics, and
the evidence that a conforming implementation must preserve. Exact questionnaire
items, response scales, randomization, and participant-facing instructions must
be supplied by a separately versioned study-instrument manifest and must refer to
this protocol ID and version.

Adopting this contract does not authorize changes to model checkpoints, model
scoring, prompts, candidate filtering, ranking, context handling, UI, or logging.

## Shared terms and invariants

- **Preference evidence** is the participant's selected movies and ratings.
- **Preference representation** is the participant-inspectable profile derived
  from or constructed from that evidence: textual summary for TEARS and genre
  profile for GERS.
- **Baseline recommendations** are the ranked recommendations produced after the
  initial preference representation is established and before a Task 2 or Task 3
  profile edit.
- **Target recommendation** is one item selected by the participant from the
  baseline recommendations for a controllability task. Its canonical MovieLens
  movie ID must remain fixed throughout that task trial.
- **Profile edit** is one participant-submitted change to the inspectable
  preference representation followed by a new recommendation request. TEARS
  profile edits change the textual summary. GERS profile edits add or remove
  genres from the genre profile. Automatic summary generation, typing that is
  not submitted, poster/metadata loading, and changes to movie evidence are not
  profile-edit attempts.
- **Rank** is one-based; a smaller number is a higher rank. Baseline rank is
  recorded before the first edit. Every submitted edit records the target's new
  rank in the same ranking domain. A target absent from the observable ranking is
  recorded as `not_returned`; no numeric rank is imputed unless a later protocol
  version explicitly defines that rule.
- Within one Task 2 or Task 3 trial, the target ID, preference evidence, model and
  checkpoint, candidate universe, exclusions, filters, alpha or blend setting,
  top-K/measurement depth, and context must remain fixed. Only the preference
  representation may change.
- Task identity is not inferred from route or screen. Every study event must carry
  the protocol ID/version, system (`TEARS` or `GERS`), canonical task ID, session,
  participant, trial, and—where applicable—edit-attempt number and target ID.

## Canonical tasks

### Task 1a — Faithfulness

The participant provides movie preferences and ratings, inspects the generated
preference representation, and evaluates whether that representation faithfully
reflects their taste.

The object of evaluation is the preference representation, not rank movement.
Required evidence includes the submitted movie IDs and ratings, the exact
representation shown to the participant, its generation status/version, and the
participant's faithfulness responses from the study-instrument manifest.

### Task 1b — Scrutability and recommendation inspection

The participant inspects the recommendations and evaluates the system, the
preference representation, and the recommendations as specified by the
study-instrument manifest.

This task begins from an established preference representation and a completed
ranked recommendation response. It has no raise-rank or lower-rank objective.
Required evidence includes the exact representation shown, the ordered
recommendation IDs/scores/ranks shown, model and serving provenance, and all
Task 1b questionnaire responses.

### Task 2 — Raise-rank controllability

The participant selects one target recommendation from the baseline
recommendations and may submit at most five profile edits intended to move that
same target upward in rank.

The baseline rank is captured before editing. After every submitted edit, the
system reruns recommendations and records the exact edited representation,
attempt number, target rank or `not_returned`, scores, and complete observed
ordering. An upward movement occurs when the target's numeric rank becomes
smaller than its baseline rank. The trial ends when the participant stops, the
study instrument's success condition is met, or five submitted edits have been
evaluated. The implementation must not silently count more than five attempts.

### Task 3 — Lower-rank controllability

The participant selects one target recommendation from the baseline
recommendations and may submit at most five profile edits intended to move that
same target downward in rank.

The baseline rank is captured before editing. After every submitted edit, the
system reruns recommendations and records the exact edited representation,
attempt number, target rank or `not_returned`, scores, and complete observed
ordering. A downward movement occurs when the target's numeric rank becomes
larger than its baseline rank. The trial ends when the participant stops, the
study instrument's success condition is met, or five submitted edits have been
evaluated. `not_returned` remains a censored observation under version 1.0.0 and
is not automatically classified as successful lowering.

### Task 4 — Context

The participant expresses the exact viewing context **“I want to unwind”** and
the study evaluates the resulting recommendation behavior as specified by the
study-instrument manifest.

The exact context string must be preserved as study evidence and must reach the
system's effective recommendation input; displaying or accepting a context that
is discarded before inference is non-conforming. Required evidence includes the
pre-context representation and ranking, the exact submitted context, the
effective model input after context handling, the context-applied representation
and ranking, and the Task 4 evaluation responses. Task 4 is a context task, not a
profile-edit attempt in Task 2 or Task 3.

## System-specific representation mapping

| Contract concept | TEARS | GERS |
| --- | --- | --- |
| Preference evidence | Selected MovieLens movies and participant ratings | Selected MovieLens movies and participant ratings |
| Inspectable preference representation | Generated and participant-editable textual summary | Movie-derived and participant-selected editable genre profile |
| Task 2/3 profile edit | Submitted text-summary edit | Submitted genre-profile add/remove edit |
| Recommendation input | Tokenized textual summary; selected IDs also participate in serving exclusions | RecVAE interaction vector from selected movie IDs plus normalized genre vector |
| Task 4 context requirement | Exact context must survive into the effective text/model input | Exact context must reach an effective GERS input; accepting and discarding it is non-conforming |

## Minimum auditable event sequence

1. Record the initial evidence snapshot.
2. Record the generated preference representation displayed for Task 1a and its
   participant evaluation.
3. Record the baseline request, effective model input, complete observed ranked
   response, rendered ordering, and Task 1b evaluation.
4. For Task 2 or Task 3, record target selection and baseline rank, then one event
   per submitted edit/recommendation response, with attempt numbers 1 through 5
   maximum.
5. For Task 4, record the pre-context baseline, the exact context “I want to
   unwind”, the effective context-bearing input, the resulting ranking, and the
   participant evaluation.

Each event must be attributable to one immutable request/result pair. Model and
data fingerprints, timestamps, status, errors, scores, model-returned ranks, and
participant-rendered ranks must be retained so that UI, serving, and model effects
can be distinguished after the study.

## Current implementation mapping (diagnostic snapshot)

This table describes the implementation as inspected on `2026-08-25`; it is not
a declaration of conformance.

| Task | TEARS current entry point | GERS current entry point |
| --- | --- | --- |
| 1a | Movie selection and ratings trigger `/api/summarize`; generated text appears in the editable summary field | Movie selection derives a visible genre profile; no participant rating control or faithfulness instrument is implemented |
| 1b | `/api/recommend` produces ranked cards from the current summary | `/api/gers` produces ranked cards from selected IDs and active genres |
| 2 | Repeated manual summary edits plus recommendation reruns are technically possible, but there is no target selector, attempt limit, or trial state | Repeated genre add/remove edits plus reruns are technically possible, but there is no target selector, attempt limit, or trial state |
| 3 | Same undifferentiated edit/rerun mechanism as Task 2; lowering intent is not represented | Same undifferentiated edit/rerun mechanism as Task 2; lowering intent is not represented |
| 4 | Fixed context dropdown regenerates the summary; the required exact context is absent from the dropdown | Fixed context dropdown sends a context field that the backend currently discards; the required exact context is absent from the dropdown |

## Versioning rules

- Any change to task meaning, the five-edit limit, target/rank semantics, or the
  exact Task 4 context requires a protocol-version increment.
- Questionnaire-only changes require a new study-instrument manifest version and
  must state whether the canonical task-contract version remains unchanged.
- Implementation fixes that preserve these meanings do not by themselves change
  this contract version, but their deployment version and fingerprints must be
  recorded with every study event.
