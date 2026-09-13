from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from pilot_api import DEFAULT_MIN_RECOMMENDATION_YEAR
from pilot_recommender import MATRIX_DIR, ML32M_LINKS, PilotHybridRecommender
from study_runtime import YEAR_FILTER_POLICY


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONBOARDING_PATH = (
    PROJECT_ROOT
    / "movie-recommender-pilot"
    / "src"
    / "data"
    / "pilot_support20_onboarding.json"
)
DEPLOYMENT_PATH = ONBOARDING_PATH.with_name("serving_deployment.json")


def _release_years(catalog: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(
        catalog.title.str.extract(r"\((\d{4})\)\s*$", expand=False),
        errors="coerce",
    ).fillna(0).astype("int64")


def test_all_years_policy_retains_unselected_onboarding_and_older_candidates() -> None:
    deployment = json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    assert YEAR_FILTER_POLICY == {
        "enabled": False,
        "field": "release_year",
        "operator": ">=",
        "value": None,
        "scope": "candidate_ranking",
    }
    assert DEFAULT_MIN_RECOMMENDATION_YEAR is None
    assert deployment["candidate_filter"] == YEAR_FILTER_POLICY

    catalog = pd.read_csv(MATRIX_DIR / "catalog.csv")
    years = _release_years(catalog)
    eligible_ids = set(catalog["movieId"].astype(int))
    onboarding_ids = {
        int(movie["movieId"])
        for movie in json.loads(ONBOARDING_PATH.read_text(encoding="utf-8"))
    }

    assert len(eligible_ids) == 22_343
    assert len(eligible_ids & onboarding_ids) == 100
    assert {4993, 5952, 7153} <= eligible_ids
    # Selecting Fellowship excludes only Fellowship; both sequels remain eligible.
    assert {5952, 7153} <= eligible_ids - {4993}


def test_enlarged_pool_retains_canonical_metadata_for_poster_resolution() -> None:
    catalog = pd.read_csv(MATRIX_DIR / "catalog.csv")
    catalog = catalog.loc[_release_years(catalog) >= 2015]
    links = pd.read_csv(
        ML32M_LINKS,
        dtype={"movieId": "int64", "imdbId": "string", "tmdbId": "Int64"},
    )
    eligible = catalog.merge(links, on="movieId", how="left", validate="one_to_one")

    assert len(eligible) == 4_584
    assert eligible.imdbId.notna().all()
    assert eligible.title.str.contains(r"\(\d{4}\)\s*$", regex=True).all()
    # One title has no linked TMDB ID and intentionally uses the verified
    # exact-title/year fallback exercised by the frontend metadata tests.
    assert int(eligible.tmdbId.isna().sum()) == 1


def test_larger_year_pool_does_not_change_existing_score_order() -> None:
    recommender = PilotHybridRecommender.__new__(PilotHybridRecommender)
    recommender.device = torch.device("cpu")
    recommender.item_count = 6
    recommender.movie_to_item = {movie_id: index for index, movie_id in enumerate(range(1, 7))}
    recommender.catalog = pd.DataFrame(
        [
            {"movieId": 1, "modelItemId": 0, "imdbId": "1", "tmdbId": 1, "title": "A (2014)", "genres": "Drama"},
            {"movieId": 2, "modelItemId": 1, "imdbId": "2", "tmdbId": 2, "title": "B (2015)", "genres": "Drama"},
            {"movieId": 3, "modelItemId": 2, "imdbId": "3", "tmdbId": 3, "title": "C (2019)", "genres": "Drama"},
            {"movieId": 4, "modelItemId": 3, "imdbId": "4", "tmdbId": 4, "title": "D (2020)", "genres": "Drama"},
            {"movieId": 5, "modelItemId": 4, "imdbId": "5", "tmdbId": 5, "title": "E (2021)", "genres": "Drama"},
            {"movieId": 6, "modelItemId": 5, "imdbId": "6", "tmdbId": 6, "title": "F (2022)", "genres": "Drama"},
        ]
    )
    logits = torch.tensor([[100.0, 60.0, 80.0, 70.0, 90.0, 50.0]])
    excluded = [5]

    old = recommender._ranked_items(logits, excluded, top_k=6, min_release_year=2020)
    enlarged = recommender._ranked_items(logits, excluded, top_k=6, min_release_year=2015)
    old_ids = [item["movie_id"] for item in old]
    enlarged_old_ids = [
        item["movie_id"] for item in enlarged if item["release_year"] >= 2020
    ]

    assert old_ids == [4, 6]
    assert enlarged_old_ids == old_ids
    assert [item["movie_id"] for item in enlarged] == [3, 4, 2, 6]
    assert [item["score"] for item in enlarged] == [80.0, 70.0, 60.0, 50.0]
