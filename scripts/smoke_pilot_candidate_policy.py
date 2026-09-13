#!/usr/bin/env python3
"""Exercise live TEARS/GERS inference against the shared candidate policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from urllib.request import Request, urlopen
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from study_runtime import PROTOCOL_ID, PROTOCOL_VERSION, YEAR_FILTER_POLICY, signature

ONBOARDING_PATH = (
    PROJECT_ROOT
    / "movie-recommender-pilot"
    / "src"
    / "data"
    / "pilot_support20_onboarding.json"
)


def request_json(base_url: str, path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def recommendation_payload(system: str, representation: str | list[str], catalog: dict) -> dict:
    liked_movie_id = int(catalog["items"][0]["movieId"])
    excluded_movie_ids = []
    payload = {
        "liked_movie_ids": [liked_movie_id],
        "preference_evidence": [
            {"movie_id": liked_movie_id, "rating": 5.0 if system == "TEARS" else None}
        ],
        "excluded_movie_ids": excluded_movie_ids,
        "catalog_fingerprint": catalog["catalog_fingerprint"],
        "onboarding_fingerprint": catalog["onboarding_fingerprint"],
        "context": "",
        "alpha": 0.5,
        "top_k": 12,
        "min_release_year": YEAR_FILTER_POLICY["value"],
    }
    payload["summary" if system == "TEARS" else "genres"] = representation
    snapshot = {
        "system": system,
        "liked_movie_ids": payload["liked_movie_ids"],
        "preference_evidence": payload["preference_evidence"],
        "excluded_movie_ids": payload["excluded_movie_ids"],
        "catalog_fingerprint": payload["catalog_fingerprint"],
        "onboarding_fingerprint": payload["onboarding_fingerprint"],
        "context": payload["context"],
        "alpha": payload["alpha"],
        "top_k": payload["top_k"],
        "min_release_year": payload["min_release_year"],
        "representation": representation,
    }
    payload["study"] = {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "participant_id": "candidate-policy-smoke",
        "session_id": f"candidate-policy-{uuid4()}",
        "system": system,
        "task_id": "1b",
        "trial_id": None,
        "attempt": None,
        "representation_revision": 1,
        "request_id": f"candidate-policy-{system.lower()}-{uuid4()}",
        "input_signature": signature(snapshot),
        "target_movie_id": None,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18010")
    parser.add_argument("--asgi", action="store_true")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--emit-payload", choices=("TEARS", "GERS"))
    parser.add_argument("--verify-response", choices=("TEARS", "GERS"))
    args = parser.parse_args()

    if args.emit_payload:
        from pilot_api import (
            EXPECTED_MATRIX_FINGERPRINT,
            EXPECTED_ONBOARDING_FINGERPRINT,
        )

        local_catalog = {
            "catalog_fingerprint": EXPECTED_MATRIX_FINGERPRINT,
            "onboarding_fingerprint": EXPECTED_ONBOARDING_FINGERPRINT,
            "items": json.loads(ONBOARDING_PATH.read_text(encoding="utf-8")),
        }
        representation: str | list[str] = (
            "The viewer enjoys atmospheric science fiction, thoughtful drama, "
            "imaginative worlds, emotionally grounded characters, and strong "
            "visual storytelling."
            if args.emit_payload == "TEARS"
            else ["Drama", "Science Fiction", "Adventure"]
        )
        print(json.dumps(recommendation_payload(args.emit_payload, representation, local_catalog)))
        return

    if args.verify_response:
        result = json.load(sys.stdin)
        items = result["items"]
        excluded = {int(json.loads(ONBOARDING_PATH.read_text(encoding="utf-8"))[0]["movieId"])}
        assert len(items) == 12
        assert not ({int(item["movie_id"]) for item in items} & excluded)
        assert not YEAR_FILTER_POLICY["enabled"]
        assert all(item["imdb_id"] and item["title"] and item["rank"] for item in items)
        assert [int(item["rank"]) for item in items] == list(range(1, 13))
        print(
            json.dumps(
                {
                    "system": args.verify_response,
                    "status": "passed",
                    "years": [int(item["release_year"]) for item in items],
                    "movie_ids": [int(item["movie_id"]) for item in items],
                },
                sort_keys=True,
            )
        )
        return

    if args.asgi:
        from fastapi.testclient import TestClient
        from pilot_api import app

        with TestClient(app) as client:
            health = client.get("/api/health").json()
            catalog = client.get("/api/catalog").json()
            assert health["study"]["candidate_filter"] == YEAR_FILTER_POLICY
            excluded = {int(catalog["items"][0]["movieId"])}
            cases = {
                "TEARS": (
                    "/api/recommend",
                    "The viewer enjoys atmospheric science fiction, thoughtful drama, "
                    "imaginative worlds, emotionally grounded characters, and strong "
                    "visual storytelling.",
                ),
                "GERS": ("/api/gers", ["Drama", "Science Fiction", "Adventure"]),
            }
            report = {"candidate_filter": YEAR_FILTER_POLICY, "systems": {}}
            for system, (path, representation) in cases.items():
                response = client.post(
                    path,
                    json=recommendation_payload(system, representation, catalog),
                )
                assert response.status_code == 200, response.text
                items = response.json()["items"]
                assert len(items) == 12
                assert not ({int(item["movie_id"]) for item in items} & excluded)
                assert not YEAR_FILTER_POLICY["enabled"]
                assert all(item["imdb_id"] and item["title"] and item["rank"] for item in items)
                assert [int(item["rank"]) for item in items] == list(range(1, 13))
                report["systems"][system] = {
                    "status": "passed",
                    "years": [int(item["release_year"]) for item in items],
                    "movie_ids": [int(item["movie_id"]) for item in items],
                }
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.direct:
        import pilot_api
        from pilot_recommender import PilotHybridRecommender
        from study_runtime import StudyStore

        recommender = PilotHybridRecommender()
        local_catalog = {
            "catalog_fingerprint": pilot_api.EXPECTED_MATRIX_FINGERPRINT,
            "onboarding_fingerprint": pilot_api.EXPECTED_ONBOARDING_FINGERPRINT,
            "items": json.loads(ONBOARDING_PATH.read_text(encoding="utf-8")),
        }
        excluded = {int(local_catalog["items"][0]["movieId"])}
        cases = {
            "TEARS": (
                pilot_api.TEARSRequest,
                pilot_api.recommend,
                "The viewer enjoys atmospheric science fiction, thoughtful drama, "
                "imaginative worlds, emotionally grounded characters, and strong "
                "visual storytelling.",
            ),
            "GERS": (
                pilot_api.GERSRequest,
                pilot_api.gers,
                ["Drama", "Science Fiction", "Adventure"],
            ),
        }
        report = {"candidate_filter": YEAR_FILTER_POLICY, "systems": {}}
        with TemporaryDirectory(prefix="tears-candidate-smoke-") as temporary:
            store = StudyStore(Path(temporary) / "study.sqlite3", {"smoke": True})
            request = SimpleNamespace(
                app=SimpleNamespace(
                    state=SimpleNamespace(
                        recommender=recommender,
                        study_store=store,
                        study_instrument={},
                    )
                )
            )
            for system, (schema, handler, representation) in cases.items():
                payload = schema.model_validate(
                    recommendation_payload(system, representation, local_catalog)
                )
                result = handler(payload, request)
                items = result["items"]
                assert len(items) == 12
                assert not ({int(item["movie_id"]) for item in items} & excluded)
                assert not YEAR_FILTER_POLICY["enabled"]
                assert all(item["imdb_id"] and item["title"] and item["rank"] for item in items)
                assert [int(item["rank"]) for item in items] == list(range(1, 13))
                report["systems"][system] = {
                    "status": "passed",
                    "years": [int(item["release_year"]) for item in items],
                    "movie_ids": [int(item["movie_id"]) for item in items],
                }
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    health = request_json(args.base_url, "/api/health")
    catalog = request_json(args.base_url, "/api/catalog")
    assert health["study"]["candidate_filter"] == YEAR_FILTER_POLICY
    assert len(catalog["items"]) == 100
    excluded = {int(catalog["items"][0]["movieId"])}

    cases = {
        "TEARS": (
            "/api/recommend",
            "The viewer enjoys atmospheric science fiction, thoughtful drama, imaginative worlds, emotionally grounded characters, and strong visual storytelling.",
        ),
        "GERS": ("/api/gers", ["Drama", "Science Fiction", "Adventure"]),
    }
    report = {"candidate_filter": YEAR_FILTER_POLICY, "systems": {}}
    for system, (path, representation) in cases.items():
        result = request_json(
            args.base_url,
            path,
            recommendation_payload(system, representation, catalog),
        )
        items = result["items"]
        assert len(items) == 12
        assert not ({int(item["movie_id"]) for item in items} & excluded)
        assert not YEAR_FILTER_POLICY["enabled"]
        assert all(item["imdb_id"] and item["title"] and item["rank"] for item in items)
        assert [int(item["rank"]) for item in items] == list(range(1, 13))
        report["systems"][system] = {
            "status": "passed",
            "years": [int(item["release_year"]) for item in items],
            "movie_ids": [int(item["movie_id"]) for item in items],
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
