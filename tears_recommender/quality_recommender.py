from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.linalg import inv

from .data import Catalog, normalize_title


@dataclass(frozen=True)
class ScoreWeights:
    collaborative: float = 0.80
    content: float = 0.15
    popularity: float = 0.05

    def normalized(self) -> "ScoreWeights":
        total = self.collaborative + self.content + self.popularity
        if total <= 0:
            raise ValueError("At least one score weight must be positive")
        return ScoreWeights(
            self.collaborative / total,
            self.content / total,
            self.popularity / total,
        )


def row_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        minimum = float(values.min())
        scale = float(values.max() - minimum)
        return (values - minimum) / scale if scale > 1e-12 else np.zeros_like(values)
    minimum = values.min(axis=1, keepdims=True)
    scale = values.max(axis=1, keepdims=True) - minimum
    return np.divide(
        values - minimum,
        scale,
        out=np.zeros_like(values),
        where=scale > 1e-12,
    )


class QualityRecommender:
    """Deterministic item-CF, genre-content, and popularity hybrid ranker."""

    def __init__(self, catalog: Catalog, weights: ScoreWeights = ScoreWeights()):
        self.catalog = catalog
        self.weights = weights.normalized()
        self.item_similarity: sp.csr_matrix | None = None
        self.popularity: np.ndarray | None = None

    def fit(
        self,
        interactions: sp.csr_matrix,
        similarity_alpha: float = 0.5,
        shrinkage: float = 0.0,
    ) -> "QualityRecommender":
        if interactions.shape[1] != self.catalog.size:
            raise ValueError("Interaction columns must align with the catalog")
        if not 0.0 <= similarity_alpha <= 1.0:
            raise ValueError("similarity_alpha must be between 0 and 1")
        if shrinkage < 0:
            raise ValueError("shrinkage cannot be negative")

        binary = interactions.astype(np.float32).tocsr()
        binary.data[:] = 1.0
        item_counts = np.asarray(binary.sum(axis=0)).ravel()
        cooccurrence = (binary.T @ binary).astype(np.float32).tocsr()
        cooccurrence.setdiag(0)
        cooccurrence.eliminate_zeros()

        left_norm = np.zeros_like(item_counts, dtype=np.float32)
        right_norm = np.zeros_like(item_counts, dtype=np.float32)
        nonzero = item_counts > 0
        left_norm[nonzero] = item_counts[nonzero] ** (-similarity_alpha)
        right_norm[nonzero] = item_counts[nonzero] ** (-(1.0 - similarity_alpha))
        similarity = sp.diags(left_norm) @ cooccurrence @ sp.diags(right_norm)
        if shrinkage:
            similarity.data *= cooccurrence.data / (cooccurrence.data + shrinkage)
        self.item_similarity = similarity.tocsr()
        self.popularity = row_minmax(np.log1p(item_counts))
        return self

    def _require_fitted(self) -> None:
        if self.item_similarity is None or self.popularity is None:
            raise RuntimeError("Call fit() before scoring")

    def content_scores(self, profiles: sp.csr_matrix) -> np.ndarray:
        genre_profiles = np.asarray(profiles @ self.catalog.genre_matrix)
        genre_profiles = (genre_profiles > 0).astype(np.float32)
        raw = genre_profiles @ self.catalog.genre_matrix.T
        denominator = np.maximum(genre_profiles.sum(axis=1, keepdims=True), 1.0)
        return raw / denominator

    def score_batch(
        self,
        profiles: sp.csr_matrix,
        weights: ScoreWeights | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        self._require_fitted()
        selected = (weights or self.weights).normalized()
        collaborative = row_minmax((profiles @ self.item_similarity).toarray())
        content = row_minmax(self.content_scores(profiles))
        popularity = np.broadcast_to(self.popularity, collaborative.shape)
        total = (
            selected.collaborative * collaborative
            + selected.content * content
            + selected.popularity * popularity
        )
        return total, {
            "collaborative": collaborative,
            "content": content,
            "popularity": popularity,
        }

    def _genres_from_text(self, text: str) -> set[str]:
        lowered = text.lower()
        found: set[str] = set()
        aliases = {"science fiction": "Sci-Fi", "scifi": "Sci-Fi", "sci fi": "Sci-Fi"}
        for phrase, canonical in aliases.items():
            if phrase in lowered:
                found.add(canonical)
        for genre in self.catalog.genre_names:
            pattern = rf"(?<!\w){re.escape(genre.lower())}(?!\w)"
            if re.search(pattern, lowered):
                found.add(genre)
        return found

    def recommend(
        self,
        description: str,
        liked_titles: Iterable[str] = (),
        disliked_genres: Iterable[str] = (),
        exclude_titles: Iterable[str] = (),
        top_k: int = 10,
        weights: ScoreWeights | None = None,
    ) -> list[dict[str, object]]:
        self._require_fitted()
        if top_k < 1:
            raise ValueError("top_k must be positive")

        liked_indices = {
            self.catalog.normalized_title_to_index[normalize_title(title)]
            for title in liked_titles
            if normalize_title(title) in self.catalog.normalized_title_to_index
        }
        explicit_genres = self._genres_from_text(description)
        profile = sp.csr_matrix(
            (
                np.ones(len(liked_indices), dtype=np.float32),
                ([0] * len(liked_indices), list(liked_indices)),
            ),
            shape=(1, self.catalog.size),
        )
        scores, components = self.score_batch(profile, weights)

        if explicit_genres:
            columns = [self.catalog.genre_names.index(g) for g in explicit_genres]
            text_content = self.catalog.genre_matrix[:, columns].mean(axis=1)
            scores[0] = 0.8 * scores[0] + 0.2 * row_minmax(text_content)

        excluded_genres = {genre.lower() for genre in disliked_genres}
        excluded = set(liked_indices)
        excluded_normalized_titles = {
            normalize_title(title) for title in exclude_titles
        }
        excluded.update(
            index
            for index, title in enumerate(self.catalog.titles)
            if normalize_title(title) in excluded_normalized_titles
        )
        excluded.update(
            i
            for i, genres in enumerate(self.catalog.genres)
            if excluded_genres.intersection(genre.lower() for genre in genres)
        )
        if excluded:
            scores[0, list(excluded)] = -np.inf

        count = min(top_k, self.catalog.size - len(excluded))
        candidate = np.argpartition(-scores[0], count - 1)[:count]
        ranked = candidate[np.argsort(-scores[0, candidate])]
        return [
            {
                "title": self.catalog.titles[i],
                "movie_id": int(self.catalog.movie_ids[i]),
                "genres": list(self.catalog.genres[i]),
                "score": float(scores[0, i]),
                "score_components": {
                    name: float(values[0, i]) for name, values in components.items()
                },
                "rank": rank,
            }
            for rank, i in enumerate(ranked, start=1)
        ]


class EASERecommender:
    """Embarrassingly Shallow Autoencoder for implicit-feedback ranking."""

    def __init__(self, regularization: float = 500.0):
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        self.regularization = regularization
        self.item_weights: np.ndarray | None = None

    def fit(self, interactions: sp.csr_matrix) -> "EASERecommender":
        matrix = interactions.astype(np.float32).tocsr()
        gram = (matrix.T @ matrix).toarray().astype(np.float32)
        diagonal = np.diag_indices_from(gram)
        gram[diagonal] += self.regularization
        precision = inv(gram, overwrite_a=True, check_finite=False)
        weights = precision / (-np.diag(precision)[None, :])
        weights[diagonal] = 0.0
        self.item_weights = weights.astype(np.float32, copy=False)
        return self

    def score_batch(self, profiles: sp.csr_matrix) -> np.ndarray:
        if self.item_weights is None:
            raise RuntimeError("Call fit() before scoring")
        return np.asarray(profiles @ self.item_weights, dtype=np.float32)


class EnhancedRecommender:
    """Online EASE ranker with text genre control and hard exclusions."""

    def __init__(
        self,
        catalog: Catalog,
        regularization: float = 500.0,
        explicit_content_weight: float = 0.15,
        popularity_weight: float = 0.02,
    ):
        if not 0.0 <= explicit_content_weight <= 1.0:
            raise ValueError("explicit_content_weight must be between 0 and 1")
        if not 0.0 <= popularity_weight <= 1.0:
            raise ValueError("popularity_weight must be between 0 and 1")
        self.catalog = catalog
        self.ease = EASERecommender(regularization)
        self.explicit_content_weight = explicit_content_weight
        self.popularity_weight = popularity_weight
        self.popularity: np.ndarray | None = None
        self._genre_parser = QualityRecommender(catalog)

    def fit(self, interactions: sp.csr_matrix) -> "EnhancedRecommender":
        self.ease.fit(interactions)
        counts = np.asarray(interactions.astype(bool).sum(axis=0)).ravel()
        self.popularity = row_minmax(np.log1p(counts))
        return self

    def recommend(
        self,
        description: str,
        liked_titles: Iterable[str] = (),
        disliked_genres: Iterable[str] = (),
        exclude_titles: Iterable[str] = (),
        top_k: int = 10,
    ) -> list[dict[str, object]]:
        liked_titles = tuple(liked_titles)
        if self.popularity is None:
            raise RuntimeError("Call fit() before recommending")
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        if not description.strip() and not liked_titles:
            raise ValueError("Provide a description or at least one liked title")

        liked_indices = {
            self.catalog.normalized_title_to_index[normalize_title(title)]
            for title in liked_titles
            if normalize_title(title) in self.catalog.normalized_title_to_index
        }
        profile = sp.csr_matrix(
            (
                np.ones(len(liked_indices), dtype=np.float32),
                ([0] * len(liked_indices), list(liked_indices)),
            ),
            shape=(1, self.catalog.size),
        )
        ease_scores = row_minmax(self.ease.score_batch(profile)[0])
        explicit_genres = self._genre_parser._genres_from_text(description)
        content_scores = np.zeros(self.catalog.size, dtype=np.float32)
        if explicit_genres:
            columns = [self.catalog.genre_names.index(genre) for genre in explicit_genres]
            content_scores = row_minmax(self.catalog.genre_matrix[:, columns].mean(axis=1))

        if liked_indices:
            content_weight = self.explicit_content_weight if explicit_genres else 0.0
            popularity_weight = self.popularity_weight
            ease_weight = 1.0 - content_weight - popularity_weight
        else:
            content_weight = 0.85 if explicit_genres else 0.0
            popularity_weight = 1.0 - content_weight
            ease_weight = 0.0
        scores = (
            ease_weight * ease_scores
            + content_weight * content_scores
            + popularity_weight * self.popularity
        )

        disliked = {genre.lower() for genre in disliked_genres}
        excluded = set(liked_indices)
        excluded_normalized_titles = {
            normalize_title(title) for title in exclude_titles
        }
        excluded.update(
            index
            for index, title in enumerate(self.catalog.titles)
            if normalize_title(title) in excluded_normalized_titles
        )
        excluded.update(
            index
            for index, genres in enumerate(self.catalog.genres)
            if disliked.intersection(genre.lower() for genre in genres)
        )
        if excluded:
            scores[list(excluded)] = -np.inf

        ranked = []
        used_normalized_titles: set[str] = set()
        for index in np.argsort(-scores):
            if not np.isfinite(scores[index]):
                continue
            normalized = normalize_title(self.catalog.titles[index])
            if normalized in used_normalized_titles:
                continue
            used_normalized_titles.add(normalized)
            ranked.append(index)
            if len(ranked) == top_k:
                break
        return [
            {
                "title": self.catalog.titles[index],
                "movie_id": int(self.catalog.movie_ids[index]),
                "genres": list(self.catalog.genres[index]),
                "score": float(scores[index]),
                "score_components": {
                    "ease": float(ease_scores[index]),
                    "explicit_content": float(content_scores[index]),
                    "popularity": float(self.popularity[index]),
                },
                "rank": rank,
            }
            for rank, index in enumerate(ranked, start=1)
        ]
