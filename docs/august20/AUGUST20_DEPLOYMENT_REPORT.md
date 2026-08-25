# August 20, 2026 - deployment status

**Recorded:** 2026-08-20 16:55 EDT
**Status:** LIVE
**Website:** <https://tearsgersab.tail9d6ed0.ts.net/pilot/>
**Public health:** <https://tearsgersab.tail9d6ed0.ts.net/pilot/api/health>

## Deployment decision

Deployment no longer waits for the five-seed Phase 6 comparison. The website
uses the already completed pilot seed-2024 models, while the full-corpus seeds
continue in Slurm for research comparison and final reporting.

This is a valid deployment promotion, not a claim that the full Phase 6 matrix
or reserved-test evaluation is complete.

## Promoted models

| Endpoint | Model | Training | Validation selection NDCG@50 | Checkpoint SHA-256 |
|---|---|---:|---:|---|
| `/pilot/api/recommend` | TEARS-RecVAE seed 2024 | 200 epochs | 0.205005 | `06e8129e78ee986a6b05eafc61ebb6c7d97e53c7b2ee84fba2ea0acdff926ca8` |
| `/pilot/api/gers` | GERS-RecVAE seed 2024 | 200 epochs | 0.199714 | `0065587925f80edc50d41e8c32c2b92d34261515c8a392a5bea0455baf18ec46` |

Both hybrids use the promoted pilot RecVAE checkpoint with SHA-256
`0fe50f12011d058cf211cb3098c62f9411a17e4d0457a9fa90cf06f3b6da848f`.
The serving layer verifies matrix, catalog, run, and checkpoint fingerprints at
startup before loading either model.

## Serving scope

- Deployment dataset: frozen support-20 scientific pilot.
- Training profiles: 9,763.
- Catalog items: 22,343.
- Inference device: `cuda:0`.
- Slurm allocation: `10314035` on `cn-l016`.
- Current allocation end: 2026-08-27 16:50 EDT.
- Summary credential: configured; no paid summary request was made during this
  deployment check.

The website labels this scope as a scientific pilot. It does not silently claim
to serve the still-training full 180,948-profile TEARS checkpoint.

## Verification

| Check | Result |
|---|---|
| Public website `/` | HTTP 200 |
| Public pilot website `/pilot/` | HTTP 200 |
| Public health endpoint | `running`, `cuda:0`, expected fingerprints |
| TEARS recommendation request | PASS, three ranked results |
| GERS recommendation request | PASS, three ranked results |
| Focused pilot recommender tests | 8 passed |
| Frontend contract tests | 4 passed |
| React application tests | 1 passed |
| OpenAI credential presence | configured, value not exposed |
| Test-split evaluation | not performed |

The TEARS smoke request returned *The Lord of the Rings: The Return of the
King*, *The Two Towers*, and *The Matrix* for a fantasy/adventure profile. The
GERS smoke request returned three matching Harry Potter titles. The selected
onboarding movie was excluded from both result sets.

## Operational risk

The public hostname is stable through Tailscale Funnel, but the application
runs inside a preemptible Slurm allocation. Job `10314035` has already been
requeued after prior preemptions. The deployment automatically returns on the
same hostname after allocation restart, but availability is not equivalent to
a persistent production host.

The immediate deployment phase is active. The remaining work is operational
hardening and eventual promotion of a full-corpus checkpoint after a suitable
single run completes; neither requires waiting for all comparison seeds before
serving users.
