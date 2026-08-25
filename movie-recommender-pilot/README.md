# Full-corpus TEARS / GERS scientific-pilot website

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

The API loads the promoted full-corpus TEARS Base seed-2022 checkpoint trained
on 180,948 users from the 200,948-profile matrix. GERS continues to use the
9,763-user support-20 GERS-RecVAE checkpoint recorded in the August pilot audit.

The visible 50-title onboarding catalog is generated from the frozen pilot
matrix by `scripts/build_pilot_onboarding_catalog.py`. It uses training-only
ratings and selects the ten most positively supported titles in each release
band from 2000 through 2023; it never reads validation/test targets.

Card posters, overviews, and ratings are resolved from the authoritative
MovieLens 32M `links.csv` TMDB IDs, with exact-year title search only as a
fallback for the small number of catalog records without a TMDB link.
