"""Explicit live smoke test, recorded under a synthetic study participant."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from uuid import uuid4


def signature(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                     sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api")
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--edit-only", action="store_true")
    parser.add_argument("--summaries-only", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--movie-id", type=int, default=288513)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.edit_only:
        import sqlite3
        import sys
        sys.path.insert(0, str(root))
        from openai import OpenAI
        from tears_summary_views import synchronize_edit
        with sqlite3.connect(f"file:{root}/logs/pilot_study.sqlite3?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT payload_json FROM events WHERE participant_id=? AND event_type='summary_result' ORDER BY rowid DESC LIMIT 1",
                ("synthetic-dual-summary-smoke",),
            ).fetchone()
        saved = json.loads(row[0])
        display = saved["representation"]
        print(json.dumps({"original_backend": saved["backend_summary"]}), flush=True)
        from types import SimpleNamespace
        client = OpenAI()
        def create(**kwargs):
            response = client.responses.create(**kwargs)
            print(response.output_text, flush=True)
            return response
        debug_client = SimpleNamespace(responses=SimpleNamespace(create=create))
        for edited in [display + " They dislike horror and prefer stories about friendship.",
                       "Summary: They prefer stories about friendship."]:
            backend, patches = synchronize_edit(debug_client, "gpt-5-mini-2025-08-07",
                saved["backend_summary"], display, edited)
            print(json.dumps({"edited": edited, "backend": backend, "patches": patches}), flush=True)
        return
    catalog = json.loads((root / "movie-recommender-pilot/src/data/pilot_support20_onboarding.json").read_text())
    manifest = json.loads((root / "movie-recommender-pilot/src/data/pilot_support20_onboarding.manifest.json").read_text())
    deployment = json.loads((root / "movie-recommender-pilot/src/data/serving_deployment.json").read_text())
    movie = next(item for item in catalog if int(item["movieId"]) == args.movie_id)
    if args.generation_only:
        import sys
        sys.path.insert(0, str(root))
        from types import SimpleNamespace
        from openai import OpenAI
        import pilot_api
        payload = pilot_api.SummaryRequest.model_construct(
            movies=[pilot_api.SummaryMovieEvidence(title=movie["title"], rating=5, genres=movie["genres"])],
            disliked=[], context="",
        )
        lines, evidence = pilot_api._online_generation_evidence(payload)
        recommender = SimpleNamespace(config=SimpleNamespace(summaries=SimpleNamespace(min_words=140, max_words=180)))
        client = OpenAI()
        display, attempts = pilot_api._generate_valid_tears_summary(client, lines,
            recommender, [movie["title"]], [], evidence)
        print(json.dumps({"display": display, "backend": display,
                          "summary_policy": pilot_api.SUMMARY_POLICY, "attempts": attempts}), flush=True)
        return
    if not args.api:
        parser.error("--api is required unless --generation-only is used")
    session = "dual-summary-smoke-" + uuid4().hex

    def post(route, payload, snapshot, task):
        payload["study"] = {
            "protocol_id": "tears-gers-human-study-tasks", "protocol_version": "1.0.0",
            "participant_id": "synthetic-dual-summary-smoke", "session_id": session,
            "system": "TEARS", "task_id": task, "representation_revision": 1,
            "request_id": uuid4().hex, "input_signature": signature(snapshot),
        }
        request = Request(args.api.rstrip("/") + route, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=240) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(error.read().decode()) from error

    payload = {"movies": [{"title": movie["title"], "rating": 5, "genres": movie["genres"]}],
               "disliked": [], "context": ""}
    for index in range(args.repeat):
        payload.pop("study", None)
        result = post("/summarize", payload, {"system": "TEARS", **payload}, "1a")
        print(json.dumps({"stage": "generated", "run": index + 1, **result}), flush=True)
    if args.summaries_only:
        return
    for stage in ("baseline", "edited", "removed"):
        display = result["summary"]
        if stage == "edited":
            display += " They dislike horror and prefer stories about friendship."
        elif stage == "removed":
            display = "Summary: They prefer stories about friendship."
        payload = {
            "summary": display, "summary_source_request_id": result["summary_source_request_id"],
            "liked_movie_ids": [int(movie["movieId"])],
            "preference_evidence": [{"movie_id": int(movie["movieId"]), "rating": 5}],
            "excluded_movie_ids": [],
            "catalog_fingerprint": deployment["matrix_fingerprint"],
            "onboarding_fingerprint": manifest["fingerprint"], "context": "",
            "alpha": 0.5, "top_k": 12, "min_release_year": deployment["candidate_filter"]["value"],
        }
        snapshot = {"system": "TEARS", **payload}
        snapshot["representation"] = snapshot.pop("summary")
        recommendations = post("/recommend", payload, snapshot, "1b")
        print(json.dumps({"stage": stage,
                          "count": len(recommendations["items"]),
                          "top3": [item["title"] for item in recommendations["items"][:3]],
                          "request_id": recommendations["study"]["request_id"]}), flush=True)


if __name__ == "__main__":
    main()
