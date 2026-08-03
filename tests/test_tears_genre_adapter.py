import types
import unittest

from tears_inference_adapter import TEARSInferenceAdapter


class GenreInferenceAdapterTest(unittest.TestCase):
    def make_adapter(self):
        adapter = object.__new__(TEARSInferenceAdapter)
        adapter.movie_metadata = {
            1: {"genres": ["Action", "Sci-Fi"]},
            2: {"genres": ["Children's", "Comedy"]},
        }
        calls = []

        def recommend(instance, **kwargs):
            calls.append(kwargs)
            return [{"movie_id": 1, "title": "Example (1999)"}]

        adapter.recommend = types.MethodType(recommend, adapter)
        return adapter, calls

    def test_genres_use_otrecvae_text_side_without_a_generated_summary(self):
        adapter, calls = self.make_adapter()

        items = adapter.recommend_genres(
            ["Action", "Science Fiction", "Action"], top_k=5
        )

        self.assertEqual(items[0]["movie_id"], 1)
        self.assertEqual(
            calls,
            [
                {
                    "summary": (
                        "The user prefers movies in these genres: "
                        "Action, Sci-Fi."
                    ),
                    "liked_movie_ids": [],
                    "alpha": 0.0,
                    "top_k": 5,
                }
            ],
        )

    def test_unknown_genre_is_rejected(self):
        adapter, _ = self.make_adapter()

        with self.assertRaisesRegex(ValueError, "Unknown MovieLens genre"):
            adapter.recommend_genres(["Made Up Genre"])

    def test_selected_catalog_movies_are_masked_from_genre_results(self):
        adapter, calls = self.make_adapter()

        adapter.recommend_genres(
            ["Action"], excluded_movie_ids=[1, 2], top_k=5
        )

        self.assertEqual(calls[0]["liked_movie_ids"], [1, 2])
        self.assertEqual(calls[0]["alpha"], 0.0)


if __name__ == "__main__":
    unittest.main()
