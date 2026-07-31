import unittest

from tmdb_collections import (
    TMDBCollectionResolver,
    movie_title_variants,
    normalize_movie_title,
)


STAR_WARS_CASES = (
    (
        260,
        "Star Wars: Episode IV - A New Hope (1977)",
        "1977",
        "Star Wars",
        11,
    ),
    (
        1196,
        "Star Wars: Episode V - The Empire Strikes Back (1980)",
        "1980",
        "The Empire Strikes Back",
        1891,
    ),
    (
        1210,
        "Star Wars: Episode VI - Return of the Jedi (1983)",
        "1983",
        "Return of the Jedi",
        1892,
    ),
    (
        2628,
        "Star Wars: Episode I - The Phantom Menace (1999)",
        "1999",
        "Star Wars: Episode I - The Phantom Menace",
        1893,
    ),
)


class TMDBTitleResolutionTests(unittest.TestCase):
    def test_episode_subtitle_variants_are_conservative(self):
        variants = movie_title_variants(
            "Star Wars: Episode V \u2013 The Empire Strikes Back (1980)"
        )
        normalized = {normalize_movie_title(variant) for variant in variants}
        self.assertEqual(
            normalized,
            {
                "star wars episode v the empire strikes back",
                "the empire strikes back",
                "star wars",
            },
        )

    def test_movielens_star_wars_titles_resolve_by_exact_variant_and_year(self):
        for movie_id, title, year, tmdb_title, tmdb_id in STAR_WARS_CASES:
            with self.subTest(movie_id=movie_id):
                resolver = TMDBCollectionResolver("test-key", None)

                def fake_get_json(path, params=None):
                    if path == "/search/movie":
                        if normalize_movie_title(params["query"]) != normalize_movie_title(
                            tmdb_title
                        ):
                            return True, {"results": []}
                        return True, {
                            "results": [
                                {
                                    "id": 999999,
                                    "title": tmdb_title,
                                    "release_date": f"{int(year) + 1}-01-01",
                                },
                                {
                                    "id": tmdb_id,
                                    "title": tmdb_title,
                                    "release_date": f"{year}-01-01",
                                },
                            ]
                        }
                    self.assertEqual(path, f"/movie/{tmdb_id}")
                    return True, {
                        "belongs_to_collection": {
                            "id": 10,
                            "name": "Star Wars Collection",
                        }
                    }

                resolver._get_json = fake_get_json
                result = resolver.resolve(movie_id, title, year)
                self.assertIsNotNone(result)
                self.assertEqual(result.tmdb_movie_id, tmdb_id)
                self.assertEqual(result.collection_id, 10)
                self.assertEqual(result.collection_name, "Star Wars Collection")

    def test_unrelated_title_with_matching_year_is_rejected(self):
        resolver = TMDBCollectionResolver("test-key", None)

        def fake_get_json(path, params=None):
            return True, {
                "results": [
                    {
                        "id": 603,
                        "title": "Star Trek: The Motion Picture",
                        "release_date": "1980-01-01",
                    }
                ]
            }

        resolver._get_json = fake_get_json
        self.assertIsNone(
            resolver.resolve(
                1196,
                "Star Wars: Episode V - The Empire Strikes Back (1980)",
                "1980",
            )
        )


if __name__ == "__main__":
    unittest.main()
