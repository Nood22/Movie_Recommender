from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest
import torch
from fastapi import HTTPException
from pydantic import ValidationError

import pilot_api
from pilot_api import EXPECTED_ONBOARDING_FINGERPRINT
from pilot_recommender import EXPECTED_MATRIX_FINGERPRINT
from study_runtime import (
    PROTOCOL_ID,
    PROTOCOL_VERSION,
    StudyStore,
    YEAR_FILTER_POLICY,
    signature,
)


def valid_summary(marker: str = "baseline") -> str:
    sentences = []
    for index in range(4):
        words = " ".join(["atmospheric"] * 34)
        prefix = "Summary: " if index == 0 else ""
        sentences.append(
            f"{prefix}The viewer prefers {marker} {words} experiences."
        )
    return " ".join(sentences)


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        marker = len(texts[0]) % 10
        return SimpleNamespace(
            input_ids=torch.tensor([[marker, 2, 3, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 0]]),
        )


class FakeRecommender:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            summaries=SimpleNamespace(min_words=120, max_words=260),
            model=SimpleNamespace(max_text_tokens=4),
        )
        self.tokenizer = FakeTokenizer()
        self.movie_to_item = {10: 0, 101: 1, 201: 2, 202: 3}
        self.catalog = pd.DataFrame(
            [
                {"movieId": 10, "title": "Evidence Movie (2019)"},
                {"movieId": 101, "title": "Target Movie (2022)"},
                {"movieId": 201, "title": "First Movie (2021)"},
                {"movieId": 202, "title": "Third Movie (2023)"},
            ]
        )
        self.genre_to_index = {"Drama": 0, "Comedy": 1, "Sci-Fi": 2}
        self.last_gers_genres: list[str] = []
        self.last_tears_summary: str | None = None
        self.tears_calls = 0

    @staticmethod
    def _item(movie_id: int, rank: int, score: float) -> dict:
        return {
            "movie_id": movie_id,
            "model_item_id": movie_id,
            "imdb_id": str(movie_id),
            "tmdb_id": movie_id,
            "title": f"Movie {movie_id} (2022)",
            "release_year": 2022,
            "genres": ["Drama"],
            "score": score,
            "rank": rank,
            "rank_label": f"#{rank}",
        }

    def recommend_tears(self, summary: str, **kwargs):
        self.tears_calls += 1
        self.last_tears_summary = summary
        if "modelerror" in summary:
            raise RuntimeError("synthetic inference failure")
        if "notreturned" in summary:
            return [self._item(201, 1, 0.9), self._item(202, 2, 0.7)]
        if "raise" in summary:
            return [
                self._item(101, 1, 0.95),
                self._item(201, 2, 0.9),
                self._item(202, 3, 0.7),
            ]
        if "lower" in summary:
            return [
                self._item(201, 1, 0.9),
                self._item(202, 2, 0.85),
                self._item(101, 3, 0.6),
            ]
        return [
            self._item(201, 1, 0.9),
            self._item(101, 2, 0.8),
            self._item(202, 3, 0.7),
        ]

    def recommend_gers(self, genres, **kwargs):
        self.last_gers_genres = list(genres)
        return [
            self._item(201, 1, 0.9),
            self._item(101, 2, 0.8),
            self._item(202, 3, 0.7),
        ]


class DirectResponse:
    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self._body = body
        self.text = json.dumps(body)

    def json(self) -> dict:
        return self._body


class DirectStudyClient:
    """Exercise validated route handlers without this cluster's hung ASGI threadpool."""

    def __init__(self, recommender, store, instrument) -> None:
        self.request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    recommender=recommender,
                    study_store=store,
                    study_instrument=instrument,
                )
            )
        )

    def post(self, path: str, json: dict) -> DirectResponse:
        routes = {
            "/api/recommend": (pilot_api.TEARSRequest, pilot_api.recommend),
            "/api/gers": (pilot_api.GERSRequest, pilot_api.gers),
            "/api/study/trials": (pilot_api.TrialCreateRequest, pilot_api.create_trial),
            "/api/study/render": (pilot_api.RenderEventRequest, pilot_api.record_render),
            "/api/study/evaluations": (
                pilot_api.EvaluationRequest,
                pilot_api.persist_evaluation,
            ),
            "/api/summarize": (pilot_api.SummaryRequest, pilot_api.summarize),
        }
        try:
            if path.startswith("/api/study/trials/") and path.endswith("/complete"):
                trial_id = path.split("/")[-2]
                payload = pilot_api.TrialCloseRequest.model_validate(json)
                body = pilot_api.complete_trial(trial_id, payload, self.request)
            else:
                model, handler = routes[path]
                payload = model.model_validate(json)
                body = handler(payload, self.request)
            return DirectResponse(200, body)
        except HTTPException as error:
            return DirectResponse(error.status_code, {"detail": error.detail})
        except ValidationError as error:
            return DirectResponse(422, {"detail": error.errors()})

    def get(self, path: str) -> DirectResponse:
        try:
            if path.startswith("/api/study/trials/"):
                trial_id = path.rsplit("/", 1)[-1]
                body = pilot_api.get_trial(trial_id, self.request)
            elif path == "/api/study/instrument":
                body = pilot_api.study_instrument(self.request)
            else:
                raise AssertionError(f"Unsupported direct GET route: {path}")
            return DirectResponse(200, body)
        except HTTPException as error:
            return DirectResponse(error.status_code, {"detail": error.detail})


@pytest.fixture
def study_client(tmp_path: Path):
    recommender = FakeRecommender()
    provenance = {
        "protocol": {"id": PROTOCOL_ID, "version": PROTOCOL_VERSION, "sha256": "p" * 64},
        "deployment": {"id": "test", "sha256": "d" * 64},
        "models": {"tears": {"sha256": "t" * 64}, "gers": {"sha256": "g" * 64}},
        "data": {"matrix_fingerprint": EXPECTED_MATRIX_FINGERPRINT},
        "candidate_filter": YEAR_FILTER_POLICY,
    }
    store = StudyStore(tmp_path / "study.sqlite3", provenance)
    instrument = {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "instrument_id": "test-instrument",
        "instrument_version": "test-1",
        "configured": True,
        "tasks": {
            "1a": {"items": [{"id": "test_faithfulness"}]},
            "1b": {"items": [{"id": "test_inspection"}]},
        },
    }
    yield DirectStudyClient(recommender, store, instrument), recommender, store


def meta(
    system: str,
    task_id: str,
    input_value: dict,
    *,
    participant_id: str = "participant-1",
    session_id: str = "session-1",
    revision: int = 1,
    trial_id: str | None = None,
    attempt: int | None = None,
    target: int | None = None,
    request_id: str | None = None,
) -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "participant_id": participant_id,
        "session_id": session_id,
        "system": system,
        "task_id": task_id,
        "trial_id": trial_id,
        "attempt": attempt,
        "representation_revision": revision,
        "request_id": request_id or f"request-{uuid4()}",
        "input_signature": signature(input_value),
        "target_movie_id": target,
    }


def recommendation_payload(system: str, representation, **overrides) -> tuple[dict, dict]:
    payload = {
        "liked_movie_ids": [10],
        "preference_evidence": [{"movie_id": 10, "rating": 5 if system == "TEARS" else None}],
        "excluded_movie_ids": [],
        "catalog_fingerprint": EXPECTED_MATRIX_FINGERPRINT,
        "onboarding_fingerprint": EXPECTED_ONBOARDING_FINGERPRINT,
        "context": "",
        "alpha": 0.5,
        "top_k": 3,
        "min_release_year": YEAR_FILTER_POLICY["value"],
    }
    payload["summary" if system == "TEARS" else "genres"] = representation
    payload.update(overrides)
    snapshot = {
        "system": system,
        "liked_movie_ids": payload["liked_movie_ids"],
        "preference_evidence": payload["preference_evidence"],
        "excluded_movie_ids": payload["excluded_movie_ids"],
        "catalog_fingerprint": payload["catalog_fingerprint"],
        "onboarding_fingerprint": payload["onboarding_fingerprint"],
        "context": payload["context"].strip(),
        "alpha": payload["alpha"],
        "top_k": payload["top_k"],
        "min_release_year": payload["min_release_year"],
        "representation": representation,
    }
    return payload, snapshot


def test_dual_summary_recommendations_use_saved_backend_and_cache_edits(study_client, monkeypatch):
    client, recommender, store = study_client
    source_meta = meta("TEARS", "1a", {})
    source_id = source_meta["request_id"]
    store.log_event("summary_result", source_meta, "completed", {
        "representation": "They enjoy comedy.",
        "backend_summary": "Summary: They enjoy comedy and friendship. They dislike horror.",
    })
    calls = []
    def sync(*args):
        calls.append(args)
        return "Summary: They enjoy drama and friendship. They dislike horror.", [{"old": "comedy", "new": "drama"}]
    monkeypatch.setattr(pilot_api, "synchronize_edit", sync)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: None))
    for display in ("They enjoy comedy.", "They enjoy drama.", "They enjoy drama."):
        payload, snapshot = recommendation_payload("TEARS", display)
        payload["summary_source_request_id"] = source_id
        snapshot["summary_source_request_id"] = source_id
        payload["study"] = meta("TEARS", "1b", snapshot)
        response = client.post("/api/recommend", json=payload)
        assert response.status_code == 200, response.text
        assert "friendship" in recommender.last_tears_summary
        assert "They dislike horror" in recommender.last_tears_summary
        assert ("comedy" if "comedy" in display else "drama") in recommender.last_tears_summary
        event = store.event_by_request(payload["study"]["request_id"], "recommendation_result")
        assert event["payload"]["effective_model_input"]["summary"] == recommender.last_tears_summary
        assert event["payload"]["representation"] == display
    assert len(calls) == 1


def baseline(client: DirectStudyClient, system: str, representation) -> tuple[dict, dict]:
    payload, snapshot = recommendation_payload(system, representation)
    payload["study"] = meta(system, "1b", snapshot)
    endpoint = "/api/recommend" if system == "TEARS" else "/api/gers"
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    render = client.post(
        "/api/study/render",
        json={
            "recommendations": [
                {"movie_id": item["movie_id"], "rank": item["rank"], "score": item["score"]}
                for item in body["items"]
            ],
            "study": payload["study"],
        },
    )
    assert render.status_code == 200, render.text
    return payload, body


def start_trial(
    client: DirectStudyClient, system: str, task_id: str, baseline_request_id: str
) -> dict:
    trial_id = f"trial-{uuid4()}"
    snapshot = {
        "system": system,
        "task_id": task_id,
        "trial_id": trial_id,
        "baseline_request_id": baseline_request_id,
        "target_movie_id": 101,
    }
    study = meta(system, task_id, snapshot, trial_id=trial_id, target=101)
    response = client.post(
        "/api/study/trials",
        json={
            "baseline_request_id": baseline_request_id,
            "target_movie_id": 101,
            "study": study,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["trial"]


@pytest.mark.parametrize("system", ["TEARS", "GERS"])
def test_candidate_policy_change_requires_new_baseline_and_blocks_existing_edits(study_client, monkeypatch, system):
    import study_runtime
    client, _, _ = study_client
    representation = valid_summary() if system == "TEARS" else ["Drama"]
    _, body = baseline(client, system, representation)
    baseline_id = body["study"]["request_id"]
    existing = start_trial(client, system, "2", baseline_id)
    monkeypatch.setattr(study_runtime, "CANDIDATE_POLICY_ID", "next-candidate-policy")
    trial_id = f"trial-{uuid4()}"
    snapshot = {"system": system, "task_id": "2", "trial_id": trial_id,
                "baseline_request_id": baseline_id, "target_movie_id": 101}
    response = client.post("/api/study/trials", json={
        "baseline_request_id": baseline_id, "target_movie_id": 101,
        "study": meta(system, "2", snapshot, trial_id=trial_id, target=101)})
    assert response.status_code == 409
    assert "current candidate policy" in response.json()["detail"]
    payload, snapshot = recommendation_payload(system, representation)
    payload["study"] = meta(system, "2", snapshot, revision=2,
                            trial_id=existing["trial_id"], attempt=1, target=101)
    response = client.post("/api/recommend" if system == "TEARS" else "/api/gers", json=payload)
    assert response.status_code == 409


def test_task_2_and_3_trials_enforce_target_immutable_baseline_and_history(study_client):
    client, _, store = study_client
    _, baseline_body = baseline(client, "TEARS", valid_summary("baseline"))

    for task_id, marker in (("2", "raise"), ("3", "lower")):
        trial = start_trial(client, "TEARS", task_id, baseline_body["study"]["request_id"])
        trial_id = trial["trial_id"]

        first_payload, first_snapshot = recommendation_payload(
            "TEARS", valid_summary(f"{marker} one")
        )
        first_payload["study"] = meta(
            "TEARS",
            task_id,
            first_snapshot,
            revision=2,
            trial_id=trial_id,
            attempt=1,
            target=101,
        )
        first = client.post("/api/recommend", json=first_payload)
        assert first.status_code == 200, first.text
        expected_rank = 1 if task_id == "2" else 3
        assert first.json()["trial"]["history"][0]["target_rank"] == expected_rank

        changed_payload, changed_snapshot = recommendation_payload(
            "TEARS", valid_summary(f"{marker} immutable"), top_k=4
        )
        changed_payload["study"] = meta(
            "TEARS",
            task_id,
            changed_snapshot,
            revision=3,
            trial_id=trial_id,
            attempt=2,
            target=101,
        )
        changed = client.post("/api/recommend", json=changed_payload)
        assert changed.status_code == 409
        assert "immutable" in changed.json()["detail"]
        assert store.get_trial(trial_id)["attempt_count"] == 1

        wrong_target_payload, wrong_target_snapshot = recommendation_payload(
            "TEARS", valid_summary(f"{marker} wrongtarget")
        )
        wrong_target_payload["study"] = meta(
            "TEARS",
            task_id,
            wrong_target_snapshot,
            revision=3,
            trial_id=trial_id,
            attempt=2,
            target=202,
        )
        wrong_target = client.post("/api/recommend", json=wrong_target_payload)
        assert wrong_target.status_code == 409
        assert store.get_trial(trial_id)["target_movie_id"] == 101

        markers = ["notreturned", f"{marker} three", f"{marker} four", f"{marker} five"]
        for attempt, attempt_marker in enumerate(markers, start=2):
            payload, snapshot = recommendation_payload(
                "TEARS", valid_summary(attempt_marker)
            )
            payload["study"] = meta(
                "TEARS",
                task_id,
                snapshot,
                revision=attempt + 1,
                trial_id=trial_id,
                attempt=attempt,
                target=101,
            )
            result = client.post("/api/recommend", json=payload)
            assert result.status_code == 200, result.text
            trial = result.json()["trial"]

        assert trial["attempt_count"] == 5
        assert trial["attempts_remaining"] == 0
        assert trial["max_attempts"] == 5
        assert trial["history"][1]["target_state"] == "not_returned"
        assert trial["history"][1]["target_rank"] is None
        assert all(item["attempt_number"] == index for index, item in enumerate(trial["history"], 1))

        sixth_payload, sixth_snapshot = recommendation_payload(
            "TEARS", valid_summary(f"{marker} sixth")
        )
        sixth_payload["study"] = meta(
            "TEARS",
            task_id,
            sixth_snapshot,
            revision=7,
            trial_id=trial_id,
            attempt=5,
            target=101,
        )
        sixth = client.post("/api/recommend", json=sixth_payload)
        assert sixth.status_code == 409
        assert "hard maximum" in sixth.json()["detail"]

        close_snapshot = {
            "system": "TEARS",
            "task_id": task_id,
            "trial_id": trial_id,
            "target_movie_id": 101,
            "action": "complete",
        }
        closed = client.post(
            f"/api/study/trials/{trial_id}/complete",
            json={
                "study": meta(
                    "TEARS",
                    task_id,
                    close_snapshot,
                    revision=7,
                    trial_id=trial_id,
                    target=101,
                )
            },
        )
        assert closed.status_code == 200, closed.text
        assert closed.json()["trial"]["status"] == "completed"


@pytest.mark.parametrize(
    ("task_id", "edited_summary", "expected_rank"),
    (
        (
            "2",
            "I want the recommendations to raise thoughtful science fiction with emotional stakes, imaginative worlds, strong character relationships, and a hopeful tone while keeping the pacing engaging and the storytelling accessible for a relaxed evening.",
            1,
        ),
        (
            "3",
            " ".join(
                [
                    "Please lower grim horror and graphic violence while favoring humane drama,"
                    "gentle comedy, emotionally grounded characters, inventive settings, and stories"
                    "that remain engaging without becoming punishing or bleak."
                ]
                * 12
            ),
            3,
        ),
        (
            "2",
            "Please raise films that feel newly adventurous and visually expressive. I am rewriting the profile around curiosity, playful momentum, unusual worlds, warm ensemble dynamics, and satisfying emotional resolution; quieter character moments are welcome, but relentless cruelty, empty spectacle, and cynical endings are not what I want from this recommendation list.",
            1,
        ),
    ),
    ids=("shortened_30_to_50_words", "expanded", "substantially_rewritten"),
)
def test_participant_edited_summaries_rerank_verbatim_without_generated_contract(
    study_client, task_id: str, edited_summary: str, expected_rank: int
) -> None:
    client, recommender, store = study_client
    _, baseline_body = baseline(client, "TEARS", valid_summary("baseline"))
    trial = start_trial(client, "TEARS", task_id, baseline_body["study"]["request_id"])
    payload, snapshot = recommendation_payload("TEARS", edited_summary)
    payload["study"] = meta(
        "TEARS",
        task_id,
        snapshot,
        revision=2,
        trial_id=trial["trial_id"],
        attempt=1,
        target=101,
    )

    response = client.post("/api/recommend", json=payload)

    assert response.status_code == 200, response.text
    assert "format_parts" not in response.text
    assert "word_count" not in response.text
    assert response.json()["trial"]["history"][0]["target_rank"] == expected_rank
    assert recommender.last_tears_summary == edited_summary
    event = store.event_by_request(
        payload["study"]["request_id"], "recommendation_result"
    )
    assert event is not None
    assert event["payload"]["request"]["representation"] == edited_summary
    assert event["payload"]["representation"] == edited_summary
    assert event["payload"]["effective_model_input"]["summary"] == edited_summary


def test_gers_genre_counts_reach_model_and_durable_effective_input(study_client):
    client, recommender, store = study_client
    genres = ["Drama", "Drama", "Comedy", "Drama"]
    _, body = baseline(client, "GERS", genres)
    assert body["items"]
    assert recommender.last_gers_genres == genres

    event = store.event_by_request(body["study"]["request_id"], "recommendation_result")
    assert event is not None
    effective = event["payload"]["effective_model_input"]
    assert effective["genres_received"] == genres
    assert effective["genre_counts"] == {"Comedy": 1, "Drama": 3}
    assert effective["normalized_genre_weights"] == {"Comedy": 0.25, "Drama": 0.75}
    assert event["payload"]["candidate_filter"] == YEAR_FILTER_POLICY
    assert event["provenance"]["models"]["gers"]["sha256"] == "g" * 64


def test_questionnaire_plumbing_persists_manifest_defined_responses(study_client):
    client, _, store = study_client
    responses = {"test_faithfulness": "test-response"}
    evaluation_input = {
        "instrument_id": "test-instrument",
        "instrument_version": "test-1",
        "responses": responses,
        "representation": valid_summary("evaluation"),
        "preference_evidence": [{"movie_id": 10, "rating": 5}],
        "context": "",
        "source_request_id": None,
        "system": "TEARS",
        "task_id": "1a",
    }
    response = client.post(
        "/api/study/evaluations",
        json={
            "instrument_id": "test-instrument",
            "instrument_version": "test-1",
            "responses": responses,
            "representation": evaluation_input["representation"],
            "preference_evidence": evaluation_input["preference_evidence"],
            "context": "",
            "source_request_id": None,
            "study": meta("TEARS", "1a", evaluation_input),
        },
    )
    assert response.status_code == 200, response.text
    assert store.counts()["evaluations"] == 1
    assert store.counts()["events"] == 1


def test_task_1b_evaluation_is_bound_to_rendered_representation(study_client):
    client, _, store = study_client
    representation = valid_summary("inspection")
    _, body = baseline(client, "TEARS", representation)
    responses = {"test_inspection": "manifest-defined-response"}
    source_request_id = body["study"]["request_id"]
    evaluation_input = {
        "instrument_id": "test-instrument",
        "instrument_version": "test-1",
        "responses": responses,
        "representation": representation,
        "preference_evidence": [{"movie_id": 10, "rating": 5}],
        "context": "",
        "source_request_id": source_request_id,
        "system": "TEARS",
        "task_id": "1b",
    }
    request = {
        "instrument_id": evaluation_input["instrument_id"],
        "instrument_version": evaluation_input["instrument_version"],
        "responses": responses,
        "representation": representation,
        "preference_evidence": evaluation_input["preference_evidence"],
        "context": "",
        "source_request_id": source_request_id,
        "study": meta("TEARS", "1b", evaluation_input),
    }

    response = client.post("/api/study/evaluations", json=request)
    assert response.status_code == 200, response.text
    event = store.event_by_request(
        request["study"]["request_id"], "questionnaire_response"
    )
    assert event is not None
    assert event["payload"]["representation"] == representation
    assert event["payload"]["source_request_id"] == source_request_id
    assert event["payload"]["responses"] == responses

    stale_input = {**evaluation_input, "representation": valid_summary("stale")}
    stale = client.post(
        "/api/study/evaluations",
        json={
            "instrument_id": stale_input["instrument_id"],
            "instrument_version": stale_input["instrument_version"],
            "responses": responses,
            "representation": stale_input["representation"],
            "preference_evidence": stale_input["preference_evidence"],
            "context": "",
            "source_request_id": source_request_id,
            "study": meta("TEARS", "1b", stale_input),
        },
    )
    assert stale.status_code == 409
    assert "rendered baseline" in stale.json()["detail"]


def test_render_rejects_stale_receipts_and_changed_ordering(study_client):
    client, _, store = study_client
    payload, snapshot = recommendation_payload("TEARS", valid_summary("render"))
    payload["study"] = meta("TEARS", "1b", snapshot)
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200, response.text
    items = response.json()["items"]
    rendered = [
        {"movie_id": item["movie_id"], "rank": item["rank"], "score": item["score"]}
        for item in items
    ]

    stale_study = {**payload["study"], "input_signature": "f" * 64}
    stale = client.post(
        "/api/study/render",
        json={"recommendations": rendered, "study": stale_study},
    )
    assert stale.status_code == 409
    assert "stale or mismatched" in stale.json()["detail"]

    changed = client.post(
        "/api/study/render",
        json={"recommendations": list(reversed(rendered)), "study": payload["study"]},
    )
    assert changed.status_code == 409
    assert "exactly match" in changed.json()["detail"]
    assert store.event_by_request(
        payload["study"]["request_id"], "ui_render"
    ) is None

    accepted = client.post(
        "/api/study/render",
        json={"recommendations": rendered, "study": payload["study"]},
    )
    assert accepted.status_code == 200, accepted.text


def test_gers_task_2_and_3_accept_only_genre_profile_edits(study_client):
    client, _, store = study_client
    _, baseline_body = baseline(client, "GERS", ["Drama", "Comedy"])
    baseline_request_id = baseline_body["study"]["request_id"]

    for task_id in ("2", "3"):
        trial = start_trial(client, "GERS", task_id, baseline_request_id)
        payload, snapshot = recommendation_payload(
            "GERS", ["Drama", "Drama", "Comedy"]
        )
        payload["study"] = meta(
            "GERS",
            task_id,
            snapshot,
            revision=2,
            trial_id=trial["trial_id"],
            attempt=1,
            target=101,
        )
        result = client.post("/api/gers", json=payload)
        assert result.status_code == 200, result.text
        updated = result.json()["trial"]
        assert updated["target_movie_id"] == 101
        assert updated["attempt_count"] == 1
        assert updated["history"][0]["target_rank"] == 2
        event = store.event_by_request(
            payload["study"]["request_id"], "recommendation_result"
        )
        assert event["payload"]["effective_model_input"]["genre_counts"] == {
            "Comedy": 1,
            "Drama": 2,
        }


def test_frozen_policy_task_4_and_durable_provenance(study_client):
    client, _, store = study_client
    payload, snapshot = recommendation_payload(
        "TEARS",
        valid_summary("unwind"),
        context="I want to unwind",
    )
    payload["study"] = meta("TEARS", "4", snapshot)
    result = client.post("/api/recommend", json=payload)
    assert result.status_code == 200, result.text
    event = store.event_by_request(
        payload["study"]["request_id"], "recommendation_result"
    )
    assert event["participant_id"] == "participant-1"
    assert event["session_id"] == "session-1"
    assert event["system"] == "TEARS"
    assert event["task_id"] == "4"
    assert event["representation_revision"] == 1
    assert event["latency_ms"] is not None
    assert event["payload"]["context"] == "I want to unwind"
    assert event["payload"]["candidate_filter"] == YEAR_FILTER_POLICY
    assert event["payload"]["alpha"]["semantics"] == (
        "not_applicable_compatibility_value"
    )
    assert event["payload"]["effective_model_input"]["kind"] == "tokenized_text"
    assert event["provenance"]["protocol"]["id"] == PROTOCOL_ID
    assert event["provenance"]["models"]["tears"]["sha256"] == "t" * 64

    changed_year, changed_year_snapshot = recommendation_payload(
        "TEARS", valid_summary("year"), min_release_year=2014
    )
    changed_year["study"] = meta("TEARS", "1b", changed_year_snapshot)
    rejected_year = client.post("/api/recommend", json=changed_year)
    assert rejected_year.status_code == 409
    assert "all release years" in rejected_year.json()["detail"]

    changed_alpha, changed_alpha_snapshot = recommendation_payload(
        "TEARS", valid_summary("alpha"), alpha=0.25
    )
    changed_alpha["study"] = meta("TEARS", "1b", changed_alpha_snapshot)
    rejected_alpha = client.post("/api/recommend", json=changed_alpha)
    assert rejected_alpha.status_code == 409
    assert "non-applicable" in rejected_alpha.json()["detail"]


def test_trial_target_selection_requires_a_durably_rendered_baseline(study_client):
    client, _, _ = study_client
    payload, snapshot = recommendation_payload("TEARS", valid_summary("unrendered"))
    payload["study"] = meta("TEARS", "1b", snapshot)
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 200, response.text

    trial_id = f"trial-{uuid4()}"
    trial_snapshot = {
        "system": "TEARS",
        "task_id": "2",
        "trial_id": trial_id,
        "baseline_request_id": response.json()["study"]["request_id"],
        "target_movie_id": 101,
    }
    rejected = client.post(
        "/api/study/trials",
        json={
            "baseline_request_id": trial_snapshot["baseline_request_id"],
            "target_movie_id": 101,
            "study": meta(
                "TEARS",
                "2",
                trial_snapshot,
                trial_id=trial_id,
                target=101,
            ),
        },
    )
    assert rejected.status_code == 409
    assert "rendered" in rejected.json()["detail"]


def test_summary_requests_require_the_exact_input_signature(study_client, monkeypatch):
    monkeypatch.setattr(pilot_api, "sparse_positive_fallback", lambda *a, **k: None)
    client, _, store = study_client
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    summary_payload = {
        "movies": [
            {"title": "Evidence Movie (2019)", "rating": 5, "genres": ["Drama"]}
        ],
        "disliked": [],
        "context": "",
    }
    snapshot = {
        "system": "TEARS",
        "movies": summary_payload["movies"],
        "disliked": [],
        "context": "",
    }
    study = meta("TEARS", "1a", snapshot)
    study["input_signature"] = "0" * 64
    response = client.post(
        "/api/summarize", json={**summary_payload, "study": study}
    )
    assert response.status_code == 409
    error = store.event_by_request(study["request_id"], "summary_error")
    assert error is not None
    assert error["status"] == "failed"

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    invalid_calls = []

    def invalid_summary_response(**kwargs):
        invalid_calls.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps(
                {"summary": "A short invalid generated summary."}
            )
        )

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=lambda: SimpleNamespace(
                responses=SimpleNamespace(
                    create=invalid_summary_response
                )
            )
        ),
    )
    valid_study = meta("TEARS", "1a", snapshot)
    invalid_generated = client.post(
        "/api/summarize", json={**summary_payload, "study": valid_study}
    )
    assert invalid_generated.status_code == 422
    assert "online privacy and format validator" in invalid_generated.json()["detail"]
    generated_error = store.event_by_request(
        valid_study["request_id"], "summary_error"
    )
    assert generated_error is not None
    assert "online privacy and format validator" in generated_error["error_json"]
    assert len(invalid_calls) == pilot_api.SUMMARY_MAX_ATTEMPTS
    assert len(generated_error["payload"]["generation_attempts"]) == (
        pilot_api.SUMMARY_MAX_ATTEMPTS
    )
    assert all(
        attempt["validator_errors"]
        for attempt in generated_error["payload"]["generation_attempts"]
    )


def test_summary_route_repairs_a_validator_failure_and_logs_attempts(
    study_client, monkeypatch
):
    client, _, store = study_client
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    outputs = iter(
        [
            "Summary: The viewer rated Evidence Movie 5/5 and prefers it.",
            (valid_summary("repaired") + " They may enjoy drama."),
        ]
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps({"summary": next(outputs)})
        )

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=lambda: SimpleNamespace(
                responses=SimpleNamespace(create=create)
            )
        ),
    )
    summary_payload = {
        "movies": [
            {"title": "Evidence Movie (2019)", "rating": 5, "genres": ["Drama"]}
        ],
        "disliked": [],
        "context": "I want to unwind",
    }
    snapshot = {
        "system": "TEARS",
        "movies": summary_payload["movies"],
        "disliked": [],
        "context": "I want to unwind",
    }
    study = meta("TEARS", "1a", snapshot)

    response = client.post(
        "/api/summarize", json={**summary_payload, "study": study}
    )

    assert response.status_code == 200, response.text
    assert response.json()["generation"] == {
        "attempt_count": 2,
        "retry_count": 1,
        "reused": False,
        "word_count": len((valid_summary("repaired") + " They may enjoy drama.").split()),
    }
    assert len(calls) == 2
    assert calls[0]["text"]["verbosity"] == "high"
    repair_prompt = calls[1]["input"][1]["content"]
    assert "rating_leakage" in repair_prompt
    assert "privacy and format validator" in repair_prompt
    assert "140-180" not in repair_prompt
    assert "four sentences" not in repair_prompt
    assert "original private evidence" in repair_prompt
    assert "VIEWING CONTEXT: I want to unwind" in repair_prompt
    event = store.event_by_request(study["request_id"], "summary_result")
    assert event is not None
    assert event["payload"]["validator"] == {"valid": True, "errors": []}
    assert event["payload"]["backend_summary"] == response.json()["summary"]
    assert event["payload"]["summary_policy"] == pilot_api.SUMMARY_POLICY
    assert event["payload"]["backend_generation_attempts"] == []
    assert [item["word_count"] for item in event["payload"]["generation_attempts"]] == [
        len("Summary: The viewer rated Evidence Movie 5/5 and prefers it.".split()),
        len((valid_summary("repaired") + " They may enjoy drama.").split()),
    ]


def test_summary_route_gives_rating_leakage_semantic_retry_guidance(
    study_client, monkeypatch
):
    client, _, _ = study_client
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    leaked = (valid_summary("rating-safe") + " They may enjoy drama.").replace(
        "The viewer prefers rating-safe",
        "The viewer rated rating-safe 5/5 and prefers",
        1,
    )
    outputs = iter([leaked, (valid_summary("rating-safe") + " They may enjoy drama.")])
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps({"summary": next(outputs)})
        )

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=lambda: SimpleNamespace(
                responses=SimpleNamespace(create=create)
            )
        ),
    )
    summary_payload = {
        "movies": [
            {"title": "Evidence Movie (2019)", "rating": 5, "genres": ["Drama"]}
        ],
        "disliked": [],
        "context": "",
    }
    snapshot = {
        "system": "TEARS",
        "movies": summary_payload["movies"],
        "disliked": [],
        "context": "",
    }

    response = client.post(
        "/api/summarize",
        json={
            **summary_payload,
            "study": meta("TEARS", "1a", snapshot),
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["generation"]["retry_count"] == 1
    assert "for: rating_leakage" in calls[1]["input"][1]["content"]
    assert "expressing the underlying preference semantically" in calls[1]["input"][1][
        "content"
    ]


def test_failed_submitted_edit_consumes_an_attempt_and_logs_the_error(study_client):
    client, _, store = study_client
    _, baseline_body = baseline(client, "TEARS", valid_summary("failurebaseline"))
    trial = start_trial(
        client, "TEARS", "2", baseline_body["study"]["request_id"]
    )
    payload, snapshot = recommendation_payload(
        "TEARS", valid_summary("modelerror")
    )
    payload["study"] = meta(
        "TEARS",
        "2",
        snapshot,
        revision=2,
        trial_id=trial["trial_id"],
        attempt=1,
        target=101,
    )
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 500
    current = store.get_trial(trial["trial_id"])
    assert current["attempt_count"] == 1
    assert current["history"][0]["status"] == "failed"
    error = store.event_by_request(
        payload["study"]["request_id"], "recommendation_error"
    )
    assert error is not None
    assert error["error_json"] is not None


def test_production_instrument_declares_missing_content_without_invented_items():
    instrument = json.loads(pilot_api.INSTRUMENT_PATH.read_text(encoding="utf-8"))
    assert instrument["configured"] is False
    assert instrument["status"] == "missing"
    assert instrument["tasks"]["1a"]["items"] == []
    assert instrument["tasks"]["1b"]["items"] == []
    assert instrument["blocker"]

    deployment = json.loads(pilot_api.DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    assert deployment["protocol_id"] == PROTOCOL_ID
    assert deployment["protocol_version"] == PROTOCOL_VERSION
    assert deployment["candidate_filter"] == YEAR_FILTER_POLICY
    assert deployment["tears_alpha"] == {
        "status": "not_applicable",
        "request_compatibility_value": 0.5,
    }


def test_unsafe_tears_edit_is_rejected_before_model_scoring(study_client):
    client, recommender, store = study_client
    payload, snapshot = recommendation_payload(
        "TEARS", "Participant edit with an unsafe null byte.\x00"
    )
    payload["study"] = meta("TEARS", "1b", snapshot)
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 422
    assert "participant edit inference-safety validator" in response.json()["detail"]
    assert "unsupported_control_character" in response.json()["detail"]
    assert recommender.tears_calls == 0
    error = store.event_by_request(payload["study"]["request_id"], "recommendation_error")
    assert error["status"] == "failed"


def test_gers_task_4_is_explicitly_blocked(study_client):
    client, _, _ = study_client
    payload, snapshot = recommendation_payload(
        "GERS", ["Drama"], context="I want to unwind"
    )
    payload["study"] = meta("GERS", "4", snapshot)
    response = client.post("/api/gers", json=payload)
    assert response.status_code == 409
    assert "GERS Task 4 is blocked" in response.json()["detail"]

    baseline_payload, baseline_snapshot = recommendation_payload(
        "GERS", ["Drama"], context="A discarded non-canonical context"
    )
    baseline_payload["study"] = meta("GERS", "1b", baseline_snapshot)
    discarded = client.post("/api/gers", json=baseline_payload)
    assert discarded.status_code == 409
    assert "no effective context input" in discarded.json()["detail"]


def test_summary_refresh_reuses_validated_text_and_isolates_evidence(study_client, monkeypatch):
    client, _, store = study_client
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    generations = []

    def generate(*args, **kwargs):
        generations.append(args[1])
        return f'Summary: They may enjoy drama and thoughtful storytelling with detail {len(generations)}.', []

    monkeypatch.setattr(pilot_api, '_generate_valid_tears_summary', generate)
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=lambda: None))

    def request(rating=5, context='', participant=None, session=None):
        evidence = {'movies': [{'title': 'Evidence Movie (2019)', 'rating': rating, 'genres': ['Drama']}],
                    'disliked': [], 'context': context}
        study = meta('TEARS', '1a', {'system': 'TEARS', **evidence})
        if participant:
            study['participant_id'] = participant
        if session:
            study['session_id'] = session
        response = client.post('/api/summarize', json={**evidence, 'study': study})
        assert response.status_code == 200, response.text
        return response.json(), study

    first, first_meta = request()
    request(rating=1)  # An unrelated request must not prime or overwrite this profile.
    refresh, refresh_meta = request(session='new-session')
    assert refresh['summary'] == first['summary']
    assert refresh['summary_source_request_id'] == refresh_meta['request_id']
    assert refresh['generation']['reused'] is True
    assert refresh['generation']['retry_count'] == 0
    assert len(generations) == 2
    event = store.event_by_request(refresh_meta['request_id'], 'summary_result')
    assert event['payload']['generation_cache']['source_request_id'] == first_meta['request_id']
    assert event['payload']['backend_summary'] == first['summary']
    request(context='I want to unwind')
    request(participant='another-viewer')
    monkeypatch.setattr(pilot_api, 'PILOT_API_SOURCE_SHA256', 'changed-renderer-or-validator')
    request()
    assert len(generations) == 5


def test_concurrent_generations_converge_and_cache_survives_store_recreation(study_client):
    _, _, store = study_client
    first = store.remember_generated_summary('viewer', 'evidence', 'first', 'request-a')
    assert store.remember_generated_summary('viewer', 'evidence', 'second', 'request-b') == first
    reloaded = type(store)(store.path, store.provenance)
    assert reloaded.generated_summary('viewer', 'evidence') == first
    assert reloaded.generated_summary('another-viewer', 'evidence') is None


def test_online_title_evidence_preserves_year_for_disambiguation():
    payload = pilot_api.SummaryRequest.model_construct(
        movies=[pilot_api.SummaryMovieEvidence(title='Soul (2020)', rating=5, genres=['Animation'])],
        disliked=[], context='')
    lines, evidence = pilot_api._online_generation_evidence(payload)
    assert 'Soul (2020)' in '\n'.join(lines)
    assert evidence['inference_payload']['positive_evidence'][0]['private_movie_title'] == 'Soul'
