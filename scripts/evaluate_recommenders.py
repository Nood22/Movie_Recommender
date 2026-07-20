#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tears_recommender import (
    EASERecommender,
    QualityRecommender,
    ScoreWeights,
    load_catalog,
    load_ratings,
)
from tears_recommender.data import DEFAULT_RATINGS_PATH


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_split(max_users: int | None, seed: int):
    catalog = load_catalog()
    ratings = load_ratings()
    ratings = ratings[
        (ratings.rating >= 4.0) & ratings.movie_id.isin(catalog.movie_id_to_index)
    ].copy()
    ratings["item_index"] = ratings.movie_id.map(catalog.movie_id_to_index)
    ratings.sort_values(["user_id", "timestamp", "movie_id"], inplace=True)
    counts = ratings.groupby("user_id").size()
    eligible_ids = counts[counts >= 5].index.to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(eligible_ids)
    if max_users is not None:
        eligible_ids = eligible_ids[:max_users]
    ratings = ratings[ratings.user_id.isin(eligible_ids)]

    users = {int(user_id): row for row, user_id in enumerate(eligible_ids)}
    targets = np.empty(len(users), dtype=np.int32)
    rows: list[int] = []
    columns: list[int] = []
    for user_id, group in ratings.groupby("user_id", sort=False):
        row = users[int(user_id)]
        item_indices = group.item_index.to_numpy(dtype=np.int32)
        targets[row] = item_indices[-1]
        rows.extend([row] * (len(item_indices) - 1))
        columns.extend(item_indices[:-1])
    profiles = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(users), catalog.size),
    )
    profiles.data[:] = 1.0
    profiles.eliminate_zeros()
    validation_mask = np.zeros(len(users), dtype=bool)
    validation_mask[: max(1, len(users) // 5)] = True
    return catalog, profiles, targets, validation_mask


def top_k_indices(scores: np.ndarray, profiles: sp.csr_matrix, k: int) -> np.ndarray:
    scores = scores.copy()
    seen_rows, seen_columns = profiles.nonzero()
    scores[seen_rows, seen_columns] = -np.inf
    partial = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    values = np.take_along_axis(scores, partial, axis=1)
    order = np.argsort(-values, axis=1)
    return np.take_along_axis(partial, order, axis=1)


def mmr_top_k_indices(
    scores: np.ndarray,
    profiles: sp.csr_matrix,
    catalog,
    k: int,
    diversity_penalty: float,
    candidate_k: int = 50,
) -> np.ndarray:
    """Greedy genre-aware maximal marginal relevance reranking."""
    candidates = top_k_indices(scores, profiles, min(candidate_k, scores.shape[1]))
    output = np.empty((scores.shape[0], k), dtype=np.int32)
    genre_matrix = catalog.genre_matrix.astype(np.float32)
    for row_index, row_candidates in enumerate(candidates):
        relevance = scores[row_index, row_candidates].astype(np.float32).copy()
        minimum = float(relevance.min())
        scale = float(relevance.max() - minimum)
        relevance = (relevance - minimum) / scale if scale > 1e-12 else np.zeros_like(relevance)
        available = np.ones(len(row_candidates), dtype=bool)
        selected_positions: list[int] = []
        for rank in range(k):
            objective = relevance.copy()
            if selected_positions and diversity_penalty:
                candidate_genres = genre_matrix[row_candidates]
                selected_genres = candidate_genres[selected_positions]
                intersection = candidate_genres @ selected_genres.T
                union = (
                    candidate_genres.sum(axis=1, keepdims=True)
                    + selected_genres.sum(axis=1)[None, :]
                    - intersection
                )
                similarity = np.divide(
                    intersection,
                    union,
                    out=np.zeros_like(intersection),
                    where=union > 0,
                ).max(axis=1)
                objective -= diversity_penalty * similarity
            objective[~available] = -np.inf
            selected = int(np.argmax(objective))
            selected_positions.append(selected)
            available[selected] = False
            output[row_index, rank] = row_candidates[selected]
    return output


def metrics(
    ranked: np.ndarray,
    targets: np.ndarray,
    catalog,
    popularity: np.ndarray,
) -> dict[str, float]:
    hits = ranked == targets[:, None]
    found = hits.any(axis=1)
    ranks = np.argmax(hits, axis=1) + 1
    reciprocal = np.where(found, 1.0 / ranks, 0.0)
    ndcg = np.where(found, 1.0 / np.log2(ranks + 1), 0.0)
    unique_items = np.unique(ranked).size

    diversity_values = []
    genres = catalog.genre_matrix.astype(np.int16)
    for row in ranked:
        matrix = genres[row]
        intersection = matrix @ matrix.T
        sizes = matrix.sum(axis=1)
        union = sizes[:, None] + sizes[None, :] - intersection
        similarity = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0,
        )
        upper = similarity[np.triu_indices(len(row), k=1)]
        diversity_values.append(float(1.0 - upper.mean()))

    return {
        f"recall@{ranked.shape[1]}": float(found.mean()),
        f"ndcg@{ranked.shape[1]}": float(ndcg.mean()),
        f"mrr@{ranked.shape[1]}": float(reciprocal.mean()),
        "catalog_coverage": float(unique_items / catalog.size),
        "intra_list_genre_diversity": float(np.mean(diversity_values)),
        "mean_recommendation_popularity": float(popularity[ranked].mean()),
    }


def evaluate_scores(name, scores, profiles, targets, mask, catalog, popularity, k, elapsed):
    ranked = top_k_indices(scores[mask], profiles[mask], k)
    result = metrics(ranked, targets[mask], catalog, popularity)
    result["users"] = int(mask.sum())
    result["milliseconds_per_user"] = float(1000 * elapsed / max(1, mask.sum()))
    return {"name": name, "metrics": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ease-regularization", type=float, default=500.0)
    parser.add_argument("--skip-ease", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    started = time.perf_counter()
    catalog, profiles, targets, validation = make_split(args.max_users, args.seed)
    ranker = QualityRecommender(catalog).fit(profiles)
    fit_seconds = time.perf_counter() - started
    test = ~validation
    if not test.any():
        test = validation.copy()

    components_started = time.perf_counter()
    combined_scores, components = ranker.score_batch(profiles)
    del combined_scores
    component_seconds = time.perf_counter() - components_started

    candidate_weights = [
        ScoreWeights(cf, content, popularity).normalized()
        for cf in (0.60, 0.75, 0.90)
        for content in (0.05, 0.20, 0.35)
        for popularity in (0.0, 0.05, 0.10)
    ]
    tuning = []
    for weights in candidate_weights:
        scores = (
            weights.collaborative * components["collaborative"]
            + weights.content * components["content"]
            + weights.popularity * components["popularity"]
        )
        ranked = top_k_indices(scores[validation], profiles[validation], args.top_k)
        score = metrics(
            ranked, targets[validation], catalog, ranker.popularity
        )[f"ndcg@{args.top_k}"]
        tuning.append((score, weights))
    best_validation_ndcg, best_weights = max(
        tuning,
        key=lambda value: (value[0], -value[1].popularity, value[1].content),
    )

    model_specs = {
        "popularity": components["popularity"],
        "genre_content": components["content"],
        "item_item_cf": components["collaborative"],
        "hybrid": (
            best_weights.collaborative * components["collaborative"]
            + best_weights.content * components["content"]
            + best_weights.popularity * components["popularity"]
        ),
    }
    models = []
    for name, scores in model_specs.items():
        models.append(
            evaluate_scores(
                name,
                scores,
                profiles,
                targets,
                test,
                catalog,
                ranker.popularity,
                args.top_k,
                component_seconds,
            )
        )
    del model_specs, components, scores
    gc.collect()

    ease_fit_seconds = None
    ease_score_seconds = None
    selected_mmr_penalty = None
    if not args.skip_ease:
        ease_started = time.perf_counter()
        ease = EASERecommender(args.ease_regularization).fit(profiles)
        ease_fit_seconds = time.perf_counter() - ease_started
        ease_score_started = time.perf_counter()
        validation_profiles = profiles[validation]
        validation_scores = ease.score_batch(validation_profiles)
        mmr_candidates = []
        for penalty in (0.0, 0.02, 0.05, 0.10, 0.20):
            validation_ranked = mmr_top_k_indices(
                validation_scores,
                validation_profiles,
                catalog,
                args.top_k,
                penalty,
            )
            validation_metrics = metrics(
                validation_ranked,
                targets[validation],
                catalog,
                ranker.popularity,
            )
            mmr_candidates.append((
                validation_metrics[f"ndcg@{args.top_k}"],
                validation_metrics["intra_list_genre_diversity"],
                -penalty,
                penalty,
            ))
        _, _, _, selected_mmr_penalty = max(mmr_candidates)
        del validation_scores, validation_profiles, validation_ranked
        gc.collect()

        test_profiles = profiles[test]
        test_scores = ease.score_batch(test_profiles)
        ease_score_seconds = time.perf_counter() - ease_score_started
        ease_ranked = top_k_indices(test_scores, test_profiles, args.top_k)
        ease_result = metrics(
            ease_ranked,
            targets[test],
            catalog,
            ranker.popularity,
        )
        ease_result["users"] = int(test.sum())
        ease_result["milliseconds_per_user"] = float(
            1000 * ease_score_seconds / test.sum()
        )
        models.append({"name": "ease", "metrics": ease_result})
        del ease_ranked
        mmr_started = time.perf_counter()
        mmr_ranked = mmr_top_k_indices(
            test_scores,
            test_profiles,
            catalog,
            args.top_k,
            selected_mmr_penalty,
        )
        mmr_elapsed = time.perf_counter() - mmr_started
        mmr_result = metrics(
            mmr_ranked,
            targets[test],
            catalog,
            ranker.popularity,
        )
        mmr_result["users"] = int(test.sum())
        mmr_result["milliseconds_per_user"] = float(1000 * mmr_elapsed / test.sum())
        models.append({"name": "ease_mmr", "metrics": mmr_result})
        del test_scores, test_profiles, mmr_ranked, ease
        gc.collect()

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "positive_rating_threshold": 4.0,
            "minimum_positives": 5,
            "holdout": "latest positive interaction",
            "validation_fraction": 0.2,
            "seed": args.seed,
            "top_k": args.top_k,
            "catalog_items": catalog.size,
            "eligible_users": profiles.shape[0],
            "test_users": int(test.sum()),
            "training_interactions": int(profiles.nnz),
            "ratings_sha256": sha256(DEFAULT_RATINGS_PATH),
        },
        "timing": {
            "item_cf_fit_seconds": fit_seconds,
            "component_scoring_seconds": component_seconds,
            "ease_fit_seconds": ease_fit_seconds,
            "ease_scoring_seconds": ease_score_seconds,
        },
        "ease_regularization": None if args.skip_ease else args.ease_regularization,
        "selected_mmr_diversity_penalty": selected_mmr_penalty,
        "selected_hybrid_weights": asdict(best_weights),
        "validation_ndcg": best_validation_ndcg,
        "models": models,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"users{profiles.shape[0]}_seed{args.seed}"
    json_path = args.output_dir / f"evaluation_{suffix}.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
