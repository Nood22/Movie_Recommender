# Canonical TEARS/GERS Study-Readiness Audit

| Field | Value |
| --- | --- |
| Audit date | `2026-08-27` |
| Rollback baseline | `1754778339dbba31926abf5cfa8250b4d50d0f95` |
| Contract | `tears-gers-human-study-tasks` version `1.0.0` |
| Scope | Current approved website implementation: participant-facing Tasks 1a, 1b, 2, and 3 |
| External administration | Questionnaires/instruments are administered outside the website |
| Explicit exclusion | Task 4 is out of scope for this implementation |
| Overall decision | **Ready to field for the current approved scope** |

## Model and deployment provenance result

The study-protocol hardening does not change TEARS/GERS scoring, latent
representations, ranking logic, penalties, or candidate masking. The deployment
was, however, deliberately promoted after the rollback baseline from the pilot
RecVAE/GERS checkpoints to the frozen full-cohort seed-2022 checkpoints. The
current `pilot_recommender.py` SHA-256 is
`c18f33fa5152eb9b35e0ab66c2f8fda41c6aa5c2063136e0b5d41085f55ec36f`;
the rollback-baseline SHA-256 was
`fec72dac7b4b8e8539fe6e79796cac3b68d2408174bfcc7186e9518392032027`.
The serving manifest identifies the deployment as
`full-tears-and-gers-200948` and pins the promoted GERS checkpoint SHA-256 to
`f4755f87b133b95ee6e9919c498011efd2096ee24da3e13895f48e3d30669dea`.

The shared `release_year >= 2015` serving policy remains fixed, is rejected if
a request attempts to change it, and is recorded in the deployment manifest,
trial immutable snapshot, result payload, and event provenance. TEARS alpha is
fixed at the compatibility value `0.5`, labeled `not_applicable`, and rejected
if a request attempts to treat it as a variable input.

## Task-by-task result

| System | Task | Result | Evidence |
| --- | --- | --- | --- |
| TEARS | 1a | **Ready** | Preference evidence, exact generated representation, revision, frozen-validator status, latency, and errors are persistable. External instruments are intentionally outside the website. |
| GERS | 1a | **Ready** | The exact counted genre representation is persistable. The frozen GERS interaction input retains its declared fixed `5.0` semantics; the website logs participant-rating evidence as `null` rather than inventing ratings. |
| TEARS | 1b | **Ready** | Recommendation persistence is bound to the exact durably rendered request, representation, ordering, ranks, scores, and provenance. |
| GERS | 1b | **Ready** | The same response/render binding is implemented, and duplicate genre occurrences reach the frozen model input. |
| TEARS | 2 | **Ready** | Participant edits use only minimal inference-safety validation, are sent to TEARS without semantic rewriting, and are logged verbatim. Stable rendered-baseline target, explicit raise intent, immutable evidence/settings, changed-representation enforcement, atomic attempt reservation, five-attempt maximum, failed-attempt accounting, and rank/score history are enforced. |
| GERS | 2 | **Ready** | The same trial orchestration is enforced and genre-frequency edits remain count-preserving. |
| TEARS | 3 | **Ready** | Participant edits have the same verbatim handling as Task 2. Stable target, explicit lower intent, immutable baseline, censored `not_returned`, complete observed ordering, and five-attempt enforcement are implemented. |
| GERS | 3 | **Ready** | The same orchestration is enforced with count-preserving genre edits. |

Task 4 is intentionally excluded from this audit and from the current approved
website implementation. Its absence is not a field-readiness blocker.

## Safeguard verification

- Every recommendation response carries an independently recomputed immutable
  input signature. The UI renders only the current request/signature and only
  after the server durably accepts the exact model-returned ordering.
- Stale receipts, altered render orderings, unrendered baselines, changed target
  IDs, changed immutable trial inputs, repeated representations, invalid
  generated TEARS summaries, unsafe/empty participant edits, sixth edits,
  changed 2020 policy values, and changed TEARS alpha values are rejected.
- SQLite events retain participant, session, system, task, trial, attempt,
  representation revision, request/input signature, target, timestamps, status,
  latency, errors, protocol/deployment/model/data fingerprints, effective model
  inputs, evidence, context, recommendations, ranks, scores, and rendered
  ordering. External questionnaire administration is not a website dependency.
- GERS genre duplicates are preserved in the browser payload, API validation,
  durable effective-input record, and frozen `_genre_vector` frequency
  normalization.
- Participant-facing screens intentionally contain no questionnaire or Task 4
  workflow.

## Verification run

- Python: `171 passed`.
- Study-readiness and unchanged Task 1a reliability subset: `26 passed`.
- Frontend contract tests: `5 passed`.
- React component tests: `1 passed`.
- Optimized frontend production build: compiled successfully.

The current approved Tasks 1a/1b/2/3 implementation is field-ready. External
questionnaires and the intentionally excluded Task 4 are not website readiness
criteria.
