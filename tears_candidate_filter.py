"""Pure candidate-eligibility filtering for TEARS recommendations."""

from typing import Any


def normalize_movie_id(movie_id: object) -> str:
    try:
        return str(int(movie_id))
    except (TypeError, ValueError, OverflowError):
        return str(movie_id).strip()


def filter_tears_items(
    items: list[dict],
    selected_movie_ids: set[str],
    catalog_movie_ids: set[str],
    selected_collection_ids: set[int],
    candidate_collections: dict[str, Any],
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    filtered_items = []
    removed_items = []
    seen_movie_ids: set[str] = set()
    for item in items:
        movie_id = item["movie_id"]
        normalized_movie_id = normalize_movie_id(movie_id)
        if normalized_movie_id in selected_movie_ids:
            removed_items.append({"movie_id": movie_id, "reason": "selected"})
            continue
        if normalized_movie_id in catalog_movie_ids:
            removed_items.append({"movie_id": movie_id, "reason": "catalog"})
            continue
        if normalized_movie_id in seen_movie_ids:
            removed_items.append({"movie_id": movie_id, "reason": "duplicate"})
            continue
        collection = candidate_collections.get(normalized_movie_id)
        if (
            collection is not None
            and collection.collection_id is not None
            and collection.collection_id in selected_collection_ids
        ):
            removed_items.append(
                {
                    "movie_id": movie_id,
                    "reason": "selected_collection",
                    "collection_id": collection.collection_id,
                }
            )
            continue
        seen_movie_ids.add(normalized_movie_id)
        filtered_items.append(item)
        if len(filtered_items) == top_k:
            break

    # Candidate ranks describe the pre-filter model output. Once ineligible
    # movies have been removed, expose ranks for the actual returned list so
    # clients always receive a contiguous #1..#N ordering.
    for rank, item in enumerate(filtered_items, start=1):
        item["rank"] = rank
        item["rank_label"] = f"#{rank}"
    return filtered_items, removed_items
