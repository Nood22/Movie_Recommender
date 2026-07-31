import unittest
from types import SimpleNamespace

from tears_candidate_filter import filter_tears_items


class TEARSCandidateFilterTests(unittest.TestCase):
    def test_excludes_selected_catalog_duplicates_and_collection(self):
        items = [
            {"movie_id": 356, "title": "Forrest Gump (1994)"},
            {"movie_id": 2028, "title": "Saving Private Ryan (1998)"},
            {"movie_id": 4000, "title": "Outside Movie (2001)"},
            {"movie_id": 4000, "title": "Outside Movie (2001)"},
            {"movie_id": 4001, "title": "Same Collection (2002)"},
            {"movie_id": 4002, "title": "Another Outside Movie (2003)"},
        ]
        retained, removed = filter_tears_items(
            items,
            selected_movie_ids={"356"},
            catalog_movie_ids={"356", "2028"},
            selected_collection_ids={99},
            candidate_collections={
                "4001": SimpleNamespace(collection_id=99),
            },
            top_k=2,
        )

        self.assertEqual([item["movie_id"] for item in retained], [4000, 4002])
        self.assertEqual(
            [item["reason"] for item in removed],
            ["selected", "catalog", "duplicate", "selected_collection"],
        )


if __name__ == "__main__":
    unittest.main()
