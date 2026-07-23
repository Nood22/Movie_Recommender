# Current Recommendation Pipeline

## Scope

This document describes the root-level FastAPI application. The larger
`TEARS_Project/Code4Neda` tree is research code and is not currently connected
to the production request path.

## Online request flow

```text
User feedback
  liked movie titles
  disliked genres/titles
  optional free-text context
        |
        | POST /summarize (optional)
        v
OpenAI gpt-4o-mini
  produces a 120-180 word taste summary
        |
        | POST /recommend
        v
T5-small SummaryEncoder
  summary -> normalized 512-D mean-pooled vector
  context -> normalized 512-D mean-pooled vector
        |
        v
Weighted fusion
  alpha * summary + (1-alpha) * context
        |
        v
Untrained Linear(512, 200) projection
        |
        v
Cosine/dot-product scoring against 3,706 normalized 200-D movie vectors
        |
        v
Post-processing
  remove disliked genres
  add 0.15 to explicitly liked titles
  sort descending
        |
        v
Top-K titles, scores, and ranks
```

`POST /gers` is a second entry path. It asks OpenAI to convert selected genres
into a taste summary and then uses the same recommendation path.

## Runtime components

| Component | File | Responsibility |
|---|---|---|
| HTTP/API composition | `api.py` | Request schemas, OpenAI calls, artifact loading, endpoints |
| Text encoder | `summary_encoder.py` | Tokenization, T5 encoding, mean pooling, vector fusion |
| Ranker | `hybrid_recommender.py` | Vector projection, similarity, filtering, boosts, top-K |
| Movie vectors | `model/saved_models/movie_embeddings.pt` | 3,706 x 200 float32 item matrix |
| Catalog | `movie_titles_fixed.pkl`, `movie_genres_fixed.pkl` | Titles and genres aligned with the item matrix |
| Precomputed summaries | `TEARS_Project/Code4Neda/saved_user_summary/...json` | 6,037 MovieLens user summaries |

## Data and dimensional contracts

- The fixed title and genre lists each contain 3,706 entries.
- The movie embedding tensor contains 3,706 rows and 200 columns.
- `movie_titles[i]`, `movie_genres[i]`, and `movie_embeddings[i]` refer to the
  same item.
- T5-small produces a 512-dimensional vector.
- The current 512-to-200 projection is created at request time and is not
  trained or loaded from a checkpoint.

## Endpoint contracts

### `POST /summarize`

Input:

```json
{
  "liked": ["Alien (1979)", "Blade Runner (1982)"],
  "disliked": ["Romance"],
  "context": "I want something atmospheric and intellectually challenging."
}
```

Output: `{ "summary": "Summary: ..." }`.

### `POST /recommend`

Input:

```json
{
  "summary": "Summary: The user enjoys atmospheric science fiction...",
  "context": "Prefer a slower, thoughtful film tonight.",
  "liked": ["Alien (1979)", "Blade Runner (1982)"],
  "disliked": ["Romance"],
  "top_k": 12,
  "alpha": 0.75
}
```

Output: `{ "items": [{ "title": ..., "score": ..., "rank": ... }] }`.

## Current quality risks

1. **Untrained cross-space projection:** T5 vectors and recommender vectors do
   not share a learned coordinate system. A new random projection is created
   on every process start, so rankings are neither meaningful nor reproducible.
2. **Genre delimiter mismatch:** catalog genres use `|`, while the filter splits
   on commas. Most single-genre exclusions therefore do not work.
3. **Seen/liked item handling:** liked movies receive a boost and can be returned
   instead of being treated as evidence for finding new movies.
4. **Summary lookup mismatch:** `/summary_from_ml1m` accepts movie IDs, but the
   loaded JSON contains user-level summaries keyed by user ID.
5. **No validation:** empty text, invalid `top_k`, and `alpha` values outside
   `[0, 1]` can lead to crashes or unintended scoring.
6. **Hardcoded paths:** the service only works at one filesystem location.
7. **No stable offline evaluator:** recommendation changes cannot currently be
   compared on the same users, splits, and metrics.
8. **Incomplete dependency declaration:** the application imports packages not
   listed in `pyproject.toml`.

## Research pipeline relationship

The older `TEARS_Project/Code4Neda/model/MF.py` implementation has the stronger
intended architecture: a text encoder and collaborative VAE are trained into
compatible latent spaces and blended with `alpha` before decoding movie logits.
Its training utilities include VAE, optimal-transport, KL, JS, and contrastive
alignment losses. That code is not a drop-in runtime dependency: it has stale
imports, multiple overlapping model variants, old checkpoints, and environment
assumptions. Useful ideas should be migrated into a small tested pipeline rather
than reconnecting the entire research tree directly.

