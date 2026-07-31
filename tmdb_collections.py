from __future__ import annotations

from dataclasses import dataclass
import json
import re
from threading import RLock
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class TMDBCollection:
    tmdb_movie_id: int
    collection_id: int | None
    collection_name: str | None


def split_movie_title_year(title: str) -> tuple[str, str]:
    match = re.search(r"\s*\((\d{4})\)\s*$", title)
    year = match.group(1) if match else ""
    clean_title = title[: match.start()].strip() if match else title.strip()
    return clean_title, year


def canonical_search_title(title: str) -> str:
    clean_title, _ = split_movie_title_year(title)
    article_match = re.match(r"^(.*),\s*(The|A|An)$", clean_title, re.IGNORECASE)
    if article_match:
        return f"{article_match.group(2)} {article_match.group(1)}"
    return clean_title


def normalize_movie_title(title: str) -> str:
    clean_title = canonical_search_title(title)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", clean_title.casefold()).split())


def movie_title_variants(title: str) -> list[str]:
    full_title = canonical_search_title(title)
    variants = [full_title]
    episode_match = re.match(
        r"^(.*?):\s*Episode\s+[IVXLCDM]+\s*[-\u2013\u2014:]\s*(.+)$",
        full_title,
        re.IGNORECASE,
    )
    if episode_match:
        variants.extend(
            [
                canonical_search_title(episode_match.group(2)),
                canonical_search_title(episode_match.group(1)),
            ]
        )

    unique_variants = []
    normalized_variants = set()
    for variant in variants:
        normalized_variant = normalize_movie_title(variant)
        if normalized_variant and normalized_variant not in normalized_variants:
            normalized_variants.add(normalized_variant)
            unique_variants.append(variant)
    return unique_variants


class TMDBCollectionResolver:
    API_BASE = "https://api.themoviedb.org/3"

    def __init__(
        self,
        api_key: str | None,
        bearer_token: str | None,
        timeout_seconds: float = 4.0,
    ) -> None:
        self.api_key = api_key.strip() if api_key else None
        self.bearer_token = bearer_token.strip() if bearer_token else None
        self.timeout_seconds = timeout_seconds
        self._movielens_to_tmdb: dict[str, int | None] = {}
        self._tmdb_to_collection: dict[int, tuple[int | None, str | None]] = {}
        self._lock = RLock()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key or self.bearer_token)

    def _get_json(
        self,
        path: str,
        params: dict[str, str] | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        if not self.enabled:
            return False, None
        query = dict(params or {})
        if self.api_key:
            query["api_key"] = self.api_key
        url = f"{self.API_BASE}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        headers = {"Accept": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return True, json.load(response)
        except Exception as error:
            print(f"TMDB collection lookup failed for {path}: {error}", flush=True)
            return False, None

    def _resolve_tmdb_id(
        self,
        movielens_movie_id: object,
        title: str,
        year: str,
    ) -> int | None:
        cache_key = str(movielens_movie_id).strip()
        with self._lock:
            if cache_key in self._movielens_to_tmdb:
                return self._movielens_to_tmdb[cache_key]

        _, title_year = split_movie_title_year(title)
        expected_year = year or title_year
        search_variants = movie_title_variants(title)
        expected_titles = {
            normalize_movie_title(variant) for variant in search_variants
        }
        tmdb_movie_id = None
        completed_search = False
        for search_title in search_variants:
            ok, payload = self._get_json(
                "/search/movie",
                {
                    "query": search_title,
                    "year": expected_year,
                    "include_adult": "false",
                },
            )
            if not ok or payload is None:
                continue
            completed_search = True
            for result in payload.get("results", []):
                result_titles = {
                    normalize_movie_title(result_title)
                    for result_title in (
                        result.get("title"),
                        result.get("original_title"),
                    )
                    if result_title
                }
                result_year = (result.get("release_date") or "").split("-")[0]
                if not result_titles.intersection(expected_titles):
                    continue
                if expected_year and result_year != expected_year:
                    continue
                try:
                    tmdb_movie_id = int(result["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                break
            if tmdb_movie_id is not None:
                break

        if completed_search:
            with self._lock:
                self._movielens_to_tmdb[cache_key] = tmdb_movie_id
        return tmdb_movie_id

    def resolve(
        self,
        movielens_movie_id: object,
        title: str,
        year: str = "",
    ) -> TMDBCollection | None:
        tmdb_movie_id = self._resolve_tmdb_id(
            movielens_movie_id,
            title,
            year,
        )
        if tmdb_movie_id is None:
            return None

        with self._lock:
            cached_collection = self._tmdb_to_collection.get(tmdb_movie_id)
        if cached_collection is not None:
            return TMDBCollection(tmdb_movie_id, *cached_collection)

        ok, payload = self._get_json(f"/movie/{tmdb_movie_id}")
        if not ok or payload is None:
            return TMDBCollection(tmdb_movie_id, None, None)

        collection = payload.get("belongs_to_collection")
        collection_id = None
        collection_name = None
        if isinstance(collection, dict):
            try:
                collection_id = int(collection["id"])
            except (KeyError, TypeError, ValueError):
                collection_id = None
            collection_name = collection.get("name") or None

        with self._lock:
            self._tmdb_to_collection[tmdb_movie_id] = (
                collection_id,
                collection_name,
            )
        return TMDBCollection(tmdb_movie_id, collection_id, collection_name)
