"""Build the pilot website catalog from training-only support-20 evidence."""

from __future__ import annotations

import argparse
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
RELEASE_BANDS = (
    (2000, 2004),
    (2005, 2009),
    (2010, 2014),
    (2015, 2019),
    (2020, 2023),
)
TITLES_PER_BAND = 10


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

    selected: list[dict[str, object]] = []
    for start, end in RELEASE_BANDS:
        candidates = catalog.loc[
            catalog.releaseYear.between(start, end)
            & (catalog.pilotTrainPositiveCount > 0)
        ].sort_values(
            ["pilotTrainPositiveCount", "pilotTrainRatingCount", "movieId"],
            ascending=[False, False, True],
        )
        if len(candidates) < TITLES_PER_BAND:
            raise RuntimeError(f"Not enough supported titles in {start}-{end}")
        for row in candidates.head(TITLES_PER_BAND).itertuples(index=False):
            selected.append(
                {
                    "movieId": int(row.movieId),
                    "imdbId": str(row.imdbId),
                    "tmdbId": None if pd.isna(row.tmdbId) else int(row.tmdbId),
                    "title": str(row.title),
                    "genres": str(row.genres).split("|"),
                    "modelItemId": int(row.modelItemId),
                    "pilotTrainPositiveCount": int(row.pilotTrainPositiveCount),
                    "pilotTrainRatingCount": int(row.pilotTrainRatingCount),
                    "releaseBand": f"{start}-{end}",
                }
            )

    expected = len(RELEASE_BANDS) * TITLES_PER_BAND
    if len(selected) != expected:
        raise AssertionError(f"Expected {expected} onboarding titles")
    movie_ids = [int(record["movieId"]) for record in selected]
    if len(movie_ids) != len(set(movie_ids)):
        raise RuntimeError("Pilot onboarding catalog contains duplicate MovieLens IDs")
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
        "release_bands": [list(band) for band in RELEASE_BANDS],
        "titles_per_band": TITLES_PER_BAND,
        "selection_rule": (
            "Within each release band, rank by pilot training-user positive "
            "count, then rating count, then MovieLens ID"
        ),
        "movie_ids": [record["movieId"] for record in records],
    }
    manifest["fingerprint"] = stable_hash(manifest)
    atomic_write_json(output, records)
    atomic_write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
