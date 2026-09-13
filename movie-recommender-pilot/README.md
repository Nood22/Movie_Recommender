# Full-corpus TEARS / GERS participant website

This is an isolated copy of the existing interface configured for
`pilot_api.py`. It does not replace or modify the original deployment.

```bash
npm run build
uvicorn pilot_api:app --host 0.0.0.0 --port 8010
```

The public deployment mounts the application at `/pilot`, which is recorded as
the package homepage so a normal production build uses the correct React router
and static bundle base. The frontend derives `/pilot/api` from that base
automatically. For an intentional local root build, use `PUBLIC_URL=/ npm run
build`; `REACT_APP_QUALITY_API_URL` remains available as an API override.

The API loads the promoted full-corpus TEARS Base seed-2022 and GERS-RecVAE
seed-2022 checkpoints. Both use the same support-20 matrix with 180,948
training, 10,000 validation, and 10,000 reserved-test users.

The visible 100-title onboarding catalog is generated from the frozen pilot
matrix by `scripts/build_pilot_onboarding_catalog.py`. It uses training-only
ratings to select 13 popular titles per year for 2022–2023, 12 per year for
2020–2021, 5 per year for 2015–2019, and 25 total from before 2015. Popularity is
the count of training ratings of at least four stars, with total training rating
count as the first tie-breaker. Years follow the canonical MovieLens title year.
Both screens spread these groups proportionally across the full list, with no
adjacent cards from the same year. The landing poster wall follows that order.
The local MovieLens 32M snapshot ends in October 2023 and has no 2024 titles;
the approved split distributes those ten slots over 2020–2023.
The artifact freezes popularity and display order; it never reads
validation/test targets or live TMDB popularity.

Card posters, overviews, and ratings are resolved from the authoritative
MovieLens 32M `links.csv` TMDB IDs, with exact-year title search as a verified
fallback if a linked lookup is temporarily unavailable.
