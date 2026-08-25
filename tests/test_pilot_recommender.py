from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from fastapi import HTTPException

from pilot_api import EXPECTED_ONBOARDING_FINGERPRINT, verify_request_provenance
from pilot_recommender import (
    EXPECTED_MATRIX_FINGERPRINT,
    EXPECTED_PILOT_MATRIX_FINGERPRINT,
    ML32M_LINKS,
    MATRIX_DIR,
    PILOT_MATRIX_DIR,
    PilotHybridRecommender,
)
from scripts.build_pilot_onboarding_catalog import select_movies


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_ONBOARDING = (
    PROJECT_ROOT
    / "movie-recommender-pilot"
    / "src"
    / "data"
    / "pilot_support20_onboarding.json"
)


def lightweight_recommender() -> PilotHybridRecommender:
    recommender = PilotHybridRecommender.__new__(PilotHybridRecommender)
    recommender.device = torch.device("cpu")
    recommender.item_count = 3
    recommender.movie_to_item = {10: 0, 20: 1, 30: 2}
    recommender.genre_names = ["Children", "Drama", "Musical", "Sci-Fi"]
    recommender.genre_to_index = {
        name: index for index, name in enumerate(recommender.genre_names)
    }
    recommender.catalog = pd.DataFrame(
        [
            {
                "movieId": 10,
                "modelItemId": 0,
                "imdbId": "0000010",
                "tmdbId": 101,
                "title": "First (2001)",
                "genres": "Drama",
            },
            {
                "movieId": 20,
                "modelItemId": 1,
                "imdbId": "0000020",
                "tmdbId": 202,
                "title": "Second (2002)",
                "genres": "Sci-Fi|Drama",
            },
            {
                "movieId": 30,
                "modelItemId": 2,
                "imdbId": "0000030",
                "tmdbId": pd.NA,
                "title": "Third (2003)",
                "genres": "Musical",
            },
        ]
    )
    return recommender


def test_interaction_vector_uses_support_catalog_mapping() -> None:
    recommender = lightweight_recommender()

    ratings = recommender._ratings([20, 999]).cpu()

    assert ratings.tolist() == [[0.0, 5.0, 0.0]]


def test_frontend_genre_aliases_match_training_features() -> None:
    recommender = lightweight_recommender()

    genres = recommender._genre_vector(
        ["Family", "Music", "Science Fiction"]
    ).cpu()

    assert genres[0].tolist() == pytest.approx([1 / 3, 0.0, 1 / 3, 1 / 3])


def test_ranking_masks_onboarding_and_selected_movie_ids() -> None:
    recommender = lightweight_recommender()

    items = recommender._ranked_items(
        torch.tensor([[10.0, 30.0, 20.0]]),
        excluded_movie_ids=[20],
        top_k=2,
    )

    assert [item["movie_id"] for item in items] == [30, 10]
    assert [item["model_item_id"] for item in items] == [2, 0]
    assert [item["rank"] for item in items] == [1, 2]
    assert items[0]["imdb_id"] == "0000030"
    assert items[0]["tmdb_id"] is None
    assert items[1]["tmdb_id"] == 101


def test_ranking_can_limit_candidates_to_recent_releases() -> None:
    recommender = lightweight_recommender()

    items = recommender._ranked_items(
        torch.tensor([[30.0, 20.0, 10.0]]),
        excluded_movie_ids=[],
        top_k=3,
        min_release_year=2002,
    )

    assert [item["movie_id"] for item in items] == [20, 30]
    assert [item["release_year"] for item in items] == [2002, 2003]
    assert [item["rank"] for item in items] == [1, 2]


def test_request_ids_outside_frozen_catalog_are_rejected() -> None:
    recommender = lightweight_recommender()

    assert recommender.validate_movie_ids([10, "20"], "liked_movie_ids") == [
        10,
        20,
    ]
    with pytest.raises(ValueError, match="outside the frozen ML-32M pilot catalog"):
        recommender.validate_movie_ids([10, 999], "excluded_movie_ids")


def test_full_tears_uses_summary_only_model_and_masks_selected_movies() -> None:
    recommender = lightweight_recommender()
    recommender._lock = RLock()
    recommender.config = SimpleNamespace(model=SimpleNamespace(max_text_tokens=4))
    recommender.tokenizer = lambda *args, **kwargs: SimpleNamespace(
        input_ids=torch.tensor([[1, 2, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 0, 0]]),
    )

    class FakeTears:
        def __init__(self) -> None:
            self.inputs: tuple[torch.Tensor, ...] | None = None

        def __call__(self, *inputs: torch.Tensor):
            self.inputs = inputs
            return torch.tensor([[30.0, 20.0, 10.0]]), None

    recommender.tears = FakeTears()

    items = recommender.recommend_tears(
        "Likes atmospheric science fiction.",
        liked_movie_ids=[10],
        excluded_movie_ids=[20],
        top_k=3,
    )

    assert recommender.tears.inputs is not None
    assert len(recommender.tears.inputs) == 2
    assert [item["movie_id"] for item in items] == [30]


def test_cached_pilot_bundle_fingerprint_remains_compatible() -> None:
    verify_request_provenance(
        EXPECTED_MATRIX_FINGERPRINT, EXPECTED_ONBOARDING_FINGERPRINT
    )
    verify_request_provenance(
        EXPECTED_PILOT_MATRIX_FINGERPRINT, EXPECTED_ONBOARDING_FINGERPRINT
    )
    with pytest.raises(HTTPException) as error:
        verify_request_provenance("0" * 64, EXPECTED_ONBOARDING_FINGERPRINT)
    assert error.value.status_code == 409


def test_frozen_pilot_manifests_match_serving_contract() -> None:
    recommender = PilotHybridRecommender.__new__(PilotHybridRecommender)

    recommender._validate_artifacts(verify_hashes=False)


def test_pilot_onboarding_catalog_is_training_derived_and_model_aligned() -> None:
    records = json.loads(PILOT_ONBOARDING.read_text(encoding="utf-8"))
    manifest = json.loads(
        PILOT_ONBOARDING.with_suffix(".manifest.json").read_text(encoding="utf-8")
    )
    model_catalog = pd.read_csv(MATRIX_DIR / "catalog.csv").set_index("movieId")
    links = pd.read_csv(
        ML32M_LINKS,
        dtype={"movieId": "int64", "imdbId": "string", "tmdbId": "Int64"},
    ).set_index("movieId")

    assert records == select_movies(PILOT_MATRIX_DIR)
    assert len(records) == 50
    assert len({record["movieId"] for record in records}) == 50
    assert Counter(record["releaseBand"] for record in records) == {
        "2000-2004": 10,
        "2005-2009": 10,
        "2010-2014": 10,
        "2015-2019": 10,
        "2020-2023": 10,
    }
    assert manifest["dataset"] == "MovieLens 32M"
    assert manifest["selection_data"] == "train_observed only"
    assert manifest["metadata_source"] == "MovieLens 32M links.csv"
    assert manifest["links_sha256"] == (
        "ef17da7710be76f7d510d5768d1b61826e3af4bf57812b9ca377e4c912123b22"
    )
    assert manifest["matrix_fingerprint"].startswith("a2cd89f043c87")
    for record in records:
        model_row = model_catalog.loc[record["movieId"]]
        assert record["modelItemId"] == int(model_row.modelItemId)
        assert record["title"] == model_row.title
        assert record["genres"] == str(model_row.genres).split("|")
        link_row = links.loc[record["movieId"]]
        assert record["imdbId"] == str(link_row.imdbId)
        assert record["tmdbId"] == int(link_row.tmdbId)


def test_pilot_active_code_has_no_legacy_ml1m_dependency() -> None:
    active_paths = [
        PROJECT_ROOT / "movie-recommender-pilot" / "src",
        PROJECT_ROOT / "pilot_api.py",
        PROJECT_ROOT / "pilot_recommender.py",
        PROJECT_ROOT / "scripts" / "build_pilot_onboarding_catalog.py",
    ]
    forbidden = ("ml-1m", "ml1m", "fixed_50_movies")

    files: list[Path] = []
    for path in active_paths:
        files.extend(path.rglob("*")) if path.is_dir() else files.append(path)
    for path in files:
        if not path.is_file() or path.suffix not in {".py", ".js", ".jsx", ".mjs", ".json"}:
            continue
        normalized_name = path.name.casefold()
        contents = path.read_text(encoding="utf-8").casefold()
        assert not any(value in normalized_name for value in forbidden), path
        assert not any(value in contents for value in forbidden), path
