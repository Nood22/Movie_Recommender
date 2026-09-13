"""Build the pilot website catalog from training-only support-20 evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tears_training.artifacts import atomic_write_json, sha256_file, stable_hash


DEFAULT_MATRIX_DIR = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/"
    "datasets/pilot/support_20/matrix"
)
DEFAULT_OUTPUT = Path(
    "movie-recommender-pilot/src/data/pilot_support20_onboarding.json"
)
DEFAULT_LINKS = PROJECT_ROOT / "ml-32m" / "links.csv"
YEAR_QUOTAS = {
    2023: 13, 2022: 13, 2021: 12, 2020: 12,
    **dict.fromkeys(range(2015, 2020), 5),
}
OLDER_YEAR_EXCLUSIVE = 2015
OLDER_MOVIE_COUNT = 25
CATALOG_SIZE = 100


def spread_release_groups(groups: list[list[dict]]) -> list[dict]:
    """Spread quota groups across the list without adjacent equal release years."""
    pending = [list(group) for group in groups]
    sizes = [len(group) for group in groups]
    total = sum(sizes)
    used = [0] * len(groups)
    remaining_years = Counter(
        int(row["releaseYear"]) for group in groups for row in group
    )
    if remaining_years and max(remaining_years.values()) > (total + 1) // 2:
        raise ValueError("Cannot separate every movie from others of the same year")

    ordered = []
    previous_year = None
    for position in range(total):
        # Favor the bucket furthest behind its proportional share of the list.
        priority = sorted(
            range(len(groups)),
            key=lambda index: (-(position + 1) * sizes[index] + used[index] * total, index),
        )
        selected = None
        for index in priority:
            for offset, row in enumerate(pending[index]):
                year = int(row["releaseYear"])
                if year == previous_year:
                    continue
                remaining = total - position - 1
                # Leave a feasible suffix whose first year differs from this one.
                if any(
                    count - (other_year == year)
                    > (remaining // 2 if other_year == year else (remaining + 1) // 2)
                    for other_year, count in remaining_years.items()
                ):
                    continue
                selected = (index, offset, year)
                break
            if selected is not None:
                break
        if selected is None:
            raise ValueError("Cannot separate every movie from others of the same year")
        index, offset, year = selected
        ordered.append(pending[index].pop(offset))
        used[index] += 1
        remaining_years[year] -= 1
        previous_year = year
    return ordered


def select_year_quota_rows(
    catalog: pd.DataFrame,
    year_counts: dict[int, int],
    *,
    before_year: int = 2015,
    before_count: int = 25,
) -> list[dict]:
    """Select the most popular supported titles in each requested year bucket."""
    if before_count < 0 or any(count < 0 for count in year_counts.values()):
        raise ValueError("Movie quotas must be nonnegative")
    if any(year < before_year for year in year_counts):
        raise ValueError("Year quotas overlap the older-movie bucket")
    eligible = catalog.loc[
        (catalog.releaseYear > 0)
        & (catalog.pilotTrainPositiveCount > 0)
        & catalog.imdbId.notna()
        & catalog.tmdbId.notna()
    ].sort_values(
        ["pilotTrainPositiveCount", "pilotTrainRatingCount", "movieId", "title"],
        ascending=[False, False, True, True],
    ).drop_duplicates("movieId").drop_duplicates("title")
    groups = []
    buckets = [
        (str(year), eligible.releaseYear == year, count)
        for year, count in sorted(year_counts.items(), reverse=True)
    ]
    buckets.append((f"before-{before_year}", eligible.releaseYear < before_year, before_count))
    for label, mask, count in buckets:
        rows = eligible.loc[mask].head(count).to_dict("records")
        if len(rows) != count:
            raise RuntimeError(
                f"Not enough supported titles in {label}: requested {count}, available {len(rows)}"
            )
        groups.append([{**row, "releaseBand": label} for row in rows])
    return spread_release_groups(groups)


def select_movies(
    matrix_dir: Path,
    links_path: Path = DEFAULT_LINKS,
) -> list[dict[str, object]]:
    catalog = pd.read_csv(matrix_dir / "catalog.csv")
    links = pd.read_csv(
        links_path,
        dtype={"movieId": "int64", "imdbId": "string", "tmdbId": "Int64"},
    )
    catalog = catalog.merge(
        links,
        on="movieId",
        how="left",
        validate="one_to_one",
    )
    if catalog.imdbId.isna().any():
        raise RuntimeError("support-20 catalog contains IDs absent from ML-32M links.csv")
    observed = sparse.load_npz(matrix_dir / "train_observed.npz").tocsr()
    if observed.shape[1] != len(catalog):
        raise RuntimeError("Training matrix and support-20 catalog are misaligned")
    if not np.array_equal(
        catalog.modelItemId.to_numpy(np.int64),
        np.arange(len(catalog), dtype=np.int64),
    ):
        raise RuntimeError("support-20 modelItemId values are not contiguous")

    positive = observed.copy()
    positive.data = (positive.data >= 4).astype(np.float32)
    positive.eliminate_zeros()
    catalog["pilotTrainPositiveCount"] = np.asarray(
        positive.getnnz(axis=0)
    ).ravel()
    catalog["pilotTrainRatingCount"] = np.asarray(
        observed.getnnz(axis=0)
    ).ravel()
    years = catalog.title.str.extract(r"\((\d{4})\)\s*$")[0]
    catalog["releaseYear"] = (
        pd.to_numeric(years, errors="coerce").fillna(0).astype(int)
    )

    selected_rows = select_year_quota_rows(
        catalog, YEAR_QUOTAS,
        before_year=OLDER_YEAR_EXCLUSIVE, before_count=OLDER_MOVIE_COUNT,
    )
    selected: list[dict[str, object]] = []
    for row in selected_rows:
        year = int(row["releaseYear"])
        selected.append(
            {
                "movieId": int(row["movieId"]),
                "imdbId": str(row["imdbId"]),
                "tmdbId": int(row["tmdbId"]),
                "title": str(row["title"]),
                "genres": str(row["genres"]).split("|"),
                "modelItemId": int(row["modelItemId"]),
                "releaseYear": year,
                "frozenPopularity": int(row["pilotTrainPositiveCount"]),
                "pilotTrainPositiveCount": int(row["pilotTrainPositiveCount"]),
                "pilotTrainRatingCount": int(row["pilotTrainRatingCount"]),
                "releaseBand": row["releaseBand"],
            }
        )

    if len(selected) != CATALOG_SIZE:
        raise AssertionError(f"Expected {CATALOG_SIZE} onboarding titles")
    movie_ids = [int(record["movieId"]) for record in selected]
    if len(movie_ids) != len(set(movie_ids)):
        raise RuntimeError("Pilot onboarding catalog contains duplicate MovieLens IDs")
    titles = [str(record["title"]) for record in selected]
    if len(titles) != len(set(titles)):
        raise RuntimeError("Pilot onboarding catalog contains duplicate titles")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--links", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    matrix_dir = args.matrix_dir.expanduser().resolve()
    links_path = args.links.expanduser().resolve()
    output = args.output.expanduser().resolve()
    records = select_movies(matrix_dir, links_path)
    matrix_manifest = json.loads(
        (matrix_dir / "manifest.json").read_text(encoding="utf-8")
    )
    manifest = {
        "dataset": "MovieLens 32M",
        "pilot_users": 9_763,
        "catalog_support": 20,
        "model_items": int(matrix_manifest["items"]),
        "matrix_fingerprint": matrix_manifest["fingerprint"],
        "metadata_source": "MovieLens 32M links.csv",
        "links_sha256": sha256_file(links_path),
        "selection_data": "train_observed only",
        "positive_rating_threshold": 4.0,
        "selection_policy": "popular-year-quotas-v1",
        "release_year_source": "Canonical MovieLens 32M title year",
        "year_quotas": {str(year): count for year, count in YEAR_QUOTAS.items()},
        "older_movies": {"before_year": OLDER_YEAR_EXCLUSIVE, "count": OLDER_MOVIE_COUNT},
        "catalog_size": CATALOG_SIZE,
        "popularity_field": "frozenPopularity",
        "popularity_source": (
            "Frozen pilot train_observed count of ratings >= 4; no live TMDB "
            "popularity is used"
        ),
        "selection_rule": (
            "Select 13 titles per year for 2022-2023, 12 per year for 2020-2021, "
            "5 per year for 2015-2019, and 25 before 2015. Within each bucket rank "
            "supported titles with pinned IMDb/TMDB links by positive count DESC, "
            "rating count DESC, MovieLens ID ASC, title ASC"
        ),
        "display_order": (
            "Proportionally interleave release-year buckets across the full list; "
            "never place equal release years next to each other. Prefer popularity "
            "order within each bucket subject to feasible year spacing"
        ),
        "movie_ids": [record["movieId"] for record in records],
        "records_fingerprint": stable_hash(records),
    }
    manifest["fingerprint"] = stable_hash(manifest)
    atomic_write_json(output, records)
    atomic_write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
