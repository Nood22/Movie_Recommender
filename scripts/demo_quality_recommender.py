#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tears_recommender import EnhancedRecommender, load_catalog, load_ratings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("description")
    parser.add_argument("--liked", action="append", default=[])
    parser.add_argument("--disliked-genre", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    catalog = load_catalog()
    ratings = load_ratings()
    ratings = ratings[
        (ratings.rating >= 4.0) & ratings.movie_id.isin(catalog.movie_id_to_index)
    ]
    users = {user_id: row for row, user_id in enumerate(ratings.user_id.unique())}
    rows = ratings.user_id.map(users).to_numpy()
    columns = ratings.movie_id.map(catalog.movie_id_to_index).to_numpy()
    interactions = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(users), catalog.size),
    )
    ranker = EnhancedRecommender(catalog).fit(interactions)
    recommendations = ranker.recommend(
        args.description,
        liked_titles=args.liked,
        disliked_genres=args.disliked_genre,
        top_k=args.top_k,
    )
    print(json.dumps(recommendations, indent=2))


if __name__ == "__main__":
    main()
