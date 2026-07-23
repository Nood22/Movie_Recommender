import unittest

import numpy as np
import scipy.sparse as sp

from tears_recommender.data import Catalog, normalize_title
from tears_recommender.quality_recommender import (
    EASERecommender,
    EnhancedRecommender,
    QualityRecommender,
)


def small_catalog() -> Catalog:
    titles = ("Space One (2000)", "Space Two (2001)", "Love One (2002)", "Mix (2003)")
    genres = (("Sci-Fi",), ("Sci-Fi", "Action"), ("Romance",), ("Sci-Fi", "Romance"))
    genre_names = ("Action", "Romance", "Sci-Fi")
    matrix = np.asarray([[0, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=np.float32)
    return Catalog(
        movie_ids=np.arange(1, 5),
        titles=titles,
        genres=genres,
        genre_names=genre_names,
        genre_matrix=matrix,
        movie_id_to_index={i + 1: i for i in range(4)},
        normalized_title_to_index={normalize_title(t): i for i, t in enumerate(titles)},
    )


class QualityRecommenderTest(unittest.TestCase):
    def setUp(self):
        interactions = sp.csr_matrix(
            np.asarray([[1, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
        )
        self.ranker = QualityRecommender(small_catalog()).fit(interactions)

    def test_is_deterministic_and_excludes_liked_title(self):
        kwargs = {"description": "science fiction", "liked_titles": ["Space One (2000)"]}
        first = self.ranker.recommend(**kwargs)
        second = self.ranker.recommend(**kwargs)
        self.assertEqual(first, second)
        self.assertNotIn("Space One (2000)", [item["title"] for item in first])

    def test_disliked_genre_is_hard_exclusion(self):
        results = self.ranker.recommend(
            "science fiction", liked_titles=["Space One"], disliked_genres=["Romance"]
        )
        self.assertTrue(all("Romance" not in item["genres"] for item in results))

    def test_excluded_and_normalized_duplicate_titles_are_not_returned(self):
        results = self.ranker.recommend(
            "science fiction",
            liked_titles=["Space One (2000)"],
            exclude_titles=["Space Two (2001)"],
        )
        self.assertNotIn("Space Two (2001)", [item["title"] for item in results])

    def test_ease_scores_have_expected_shape_and_finite_values(self):
        interactions = sp.csr_matrix(
            np.asarray([[1, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
        )
        scores = EASERecommender(10.0).fit(interactions).score_batch(interactions)
        self.assertEqual(scores.shape, interactions.shape)
        self.assertTrue(np.isfinite(scores).all())

    def test_enhanced_ranker_respects_feedback(self):
        interactions = sp.csr_matrix(
            np.asarray([[1, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
        )
        ranker = EnhancedRecommender(small_catalog(), regularization=10.0).fit(interactions)
        results = ranker.recommend(
            "I like science fiction", ["Space One (2000)"], ["Romance"], top_k=1
        )
        self.assertEqual(results[0]["title"], "Space Two (2001)")
        self.assertNotIn("Romance", results[0]["genres"])

    def test_enhanced_ranker_excludes_previous_results(self):
        interactions = sp.csr_matrix(
            np.asarray([[1, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
        )
        ranker = EnhancedRecommender(small_catalog(), regularization=10.0).fit(interactions)
        first = ranker.recommend("science fiction", ["Space One (2000)"], top_k=1)
        second = ranker.recommend(
            "science fiction",
            ["Space One (2000)"],
            exclude_titles=[first[0]["title"]],
            top_k=1,
        )
        self.assertNotEqual(first[0]["title"], second[0]["title"])


if __name__ == "__main__":
    unittest.main()
