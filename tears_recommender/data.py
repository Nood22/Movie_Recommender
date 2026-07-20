from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATINGS_PATH = (
    ROOT / "TEARS_Project" / "Code4Neda" / "data" / "ml-1m" / "ratings.dat"
)
DEFAULT_MOVIES_PATH = (
    ROOT / "TEARS_Project" / "Code4Neda" / "data" / "ml-1m" / "movies.dat"
)
DEFAULT_ARTIFACT_DIR = ROOT / "model" / "saved_models"


@dataclass(frozen=True)
class Catalog:
    movie_ids: np.ndarray
    titles: tuple[str, ...]
    genres: tuple[tuple[str, ...], ...]
    genre_names: tuple[str, ...]
    genre_matrix: np.ndarray
    movie_id_to_index: dict[int, int]
    normalized_title_to_index: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.titles)


def normalize_title(title: str) -> str:
    value = title.lower().strip()
    if value.endswith(")") and "(" in value:
        value = value[: value.rfind("(")].strip()
    for suffix in (", the", ", an", ", a"):
        if value.endswith(suffix):
            value = f"{value[:-len(suffix)]} {suffix[2:]}"
            break
    return " ".join(value.split())


def _read_movies(path: Path) -> dict[str, tuple[int, tuple[str, ...]]]:
    by_title: dict[str, tuple[int, tuple[str, ...]]] = {}
    with path.open(encoding="latin-1") as handle:
        for line in handle:
            movie_id, title, raw_genres = line.rstrip("\n").split("::")
            by_title[title] = (int(movie_id), tuple(raw_genres.split("|")))
    return by_title


def load_catalog(
    movies_path: Path = DEFAULT_MOVIES_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> Catalog:
    """Load the exact 3,706-item catalog aligned with saved model artifacts."""
    with (artifact_dir / "movie_titles_fixed.pkl").open("rb") as handle:
        titles = tuple(pickle.load(handle))
    with (artifact_dir / "movie_genres_fixed.pkl").open("rb") as handle:
        saved_genres = tuple(pickle.load(handle))

    if len(titles) != len(saved_genres):
        raise ValueError("Fixed title and genre artifacts are not aligned")

    movie_rows = _read_movies(movies_path)
    missing = [title for title in titles if title not in movie_rows]
    if missing:
        raise ValueError(f"{len(missing)} catalog titles are absent from movies.dat")

    movie_ids = np.asarray([movie_rows[title][0] for title in titles], dtype=np.int32)
    genres = tuple(tuple(raw.split("|")) for raw in saved_genres)
    genre_names = tuple(sorted({genre for row in genres for genre in row}))
    genre_to_column = {genre: index for index, genre in enumerate(genre_names)}
    genre_matrix = np.zeros((len(titles), len(genre_names)), dtype=np.float32)
    for item_index, item_genres in enumerate(genres):
        for genre in item_genres:
            genre_matrix[item_index, genre_to_column[genre]] = 1.0

    return Catalog(
        movie_ids=movie_ids,
        titles=titles,
        genres=genres,
        genre_names=genre_names,
        genre_matrix=genre_matrix,
        movie_id_to_index={int(mid): i for i, mid in enumerate(movie_ids)},
        normalized_title_to_index={normalize_title(title): i for i, title in enumerate(titles)},
    )


def load_ratings(path: Path = DEFAULT_RATINGS_PATH) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={
            "user_id": np.int32,
            "movie_id": np.int32,
            "rating": np.float32,
            "timestamp": np.int64,
        },
    )

