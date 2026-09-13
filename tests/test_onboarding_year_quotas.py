from collections import Counter

import pandas as pd
import pytest

from scripts.build_pilot_onboarding_catalog import (
    select_year_quota_rows,
    spread_release_groups,
)


def test_quota_selection_uses_popularity_and_spreads_older_movies():
    rows = []
    for year in range(1990, 2024):
        for popularity in range(1, 31):
            movie_id = year * 100 + popularity
            rows.append({
                "movieId": movie_id,
                "title": f"Movie {movie_id} ({year})",
                "releaseYear": year,
                "pilotTrainPositiveCount": popularity,
                "pilotTrainRatingCount": popularity + 5,
                "imdbId": str(movie_id),
                "tmdbId": movie_id,
            })
    quotas = {2023: 13, 2022: 13, 2021: 12, 2020: 12, **dict.fromkeys(range(2015, 2020), 5)}
    result = select_year_quota_rows(pd.DataFrame(rows), quotas)
    assert len(result) == 100
    assert len({row["movieId"] for row in result}) == 100
    assert all(a["releaseYear"] != b["releaseYear"] for a, b in zip(result, result[1:]))
    counts = Counter(row["releaseYear"] for row in result)
    for year, count in quotas.items():
        assert counts[year] == count
        assert {row["pilotTrainPositiveCount"] for row in result if row["releaseYear"] == year} == set(range(31 - count, 31))
    assert sum(row["releaseYear"] < 2015 for row in result) == 25
    for offset in range(0, 100, 20):
        assert 4 <= sum(row["releaseYear"] < 2015 for row in result[offset:offset + 20]) <= 6
    assert result == select_year_quota_rows(pd.DataFrame(rows).sample(frac=1, random_state=7), quotas)


def test_missing_year_is_reported_instead_of_silently_reallocated():
    catalog = pd.DataFrame([{
        "movieId": 1, "title": "Movie (2023)", "releaseYear": 2023,
        "pilotTrainPositiveCount": 10, "pilotTrainRatingCount": 15,
        "imdbId": "1", "tmdbId": 1,
    }])
    with pytest.raises(RuntimeError, match="2024: requested 10, available 0"):
        select_year_quota_rows(catalog, {2024: 10}, before_count=0)


def test_spacing_handles_a_dominant_year_and_rejects_impossible_counts():
    groups = [[{"releaseYear": year, "movieId": f"{year}-{i}"} for i in range(count)]
              for year, count in [(2000, 4), (2020, 2), (2021, 1)]]
    result = spread_release_groups(groups)
    assert len(result) == 7
    assert all(a["releaseYear"] != b["releaseYear"] for a, b in zip(result, result[1:]))
    with pytest.raises(ValueError, match="Cannot separate"):
        spread_release_groups([[{"releaseYear": 2000}] * 4, [{"releaseYear": 2020}] * 2])


def test_overlapping_year_buckets_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        select_year_quota_rows(pd.DataFrame(), {2014: 5})
