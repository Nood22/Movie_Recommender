import unittest

from tears_recommender import load_catalog


class DataContractTest(unittest.TestCase):
    def test_fixed_catalog_is_aligned(self):
        catalog = load_catalog()
        self.assertEqual(catalog.size, 3706)
        self.assertEqual(catalog.genre_matrix.shape[0], catalog.size)
        self.assertEqual(len(catalog.movie_ids), catalog.size)
        self.assertEqual(len(set(catalog.movie_ids.tolist())), catalog.size)


if __name__ == "__main__":
    unittest.main()

