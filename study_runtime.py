"""Durable, model-agnostic orchestration and audit storage for the pilot study."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Iterator, Mapping
from uuid import uuid4
from tears_preference_ranking import POLICY_ID as TEARS_RANKING_POLICY


PROTOCOL_ID = "tears-gers-human-study-tasks"
PROTOCOL_VERSION = "1.0.0"
MAX_PROFILE_EDIT_ATTEMPTS = 5
CANONICAL_CONTEXT = "I want to unwind"
YEAR_FILTER_POLICY = {
    "enabled": False,
    "field": "release_year",
    "operator": ">=",
    "value": None,
    "scope": "candidate_ranking",
}
CANDIDATE_POLICY_ID = "all-years-selected-only-v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def signature(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if isinstance(item, float) and item.is_integer():
            return int(item)
        if isinstance(item, list):
            return [normalize(value) for value in item]
        if isinstance(item, tuple):
            return [normalize(value) for value in item]
        if isinstance(item, dict):
            return {str(key): normalize(value) for key, value in item.items()}
        return item

    return hashlib.sha256(canonical_json(normalize(value)).encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recommendation_input_snapshot(system: str, payload: Any) -> dict[str, Any]:
    """Return the exact client/server signature domain for a recommendation."""

    base = {
        "system": system,
        "liked_movie_ids": list(payload.liked_movie_ids),
        "preference_evidence": [
            movie.model_dump(mode="json") for movie in payload.preference_evidence
        ],
        "excluded_movie_ids": list(payload.excluded_movie_ids),
        "catalog_fingerprint": payload.catalog_fingerprint,
        "onboarding_fingerprint": payload.onboarding_fingerprint,
        "context": (payload.context or "").strip(),
        "alpha": payload.alpha,
        "top_k": payload.top_k,
        "min_release_year": payload.min_release_year,
    }
    if system == "TEARS":
        # Participant-authored edits are part of the signed evidence verbatim.
        base["representation"] = payload.summary
        if getattr(payload, "summary_source_request_id", None):
            base["summary_source_request_id"] = payload.summary_source_request_id
    else:
        # Order and duplicates are semantically meaningful genre frequencies.
        base["representation"] = list(payload.genres)
    return base


def summary_input_snapshot(payload: Any) -> dict[str, Any]:
    return {
        "system": "TEARS",
        "movies": [movie.model_dump(mode="json") for movie in payload.movies],
        "disliked": list(payload.disliked),
        "context": (payload.context or "").strip(),
    }


def immutable_trial_snapshot(system: str, payload: Any) -> dict[str, Any]:
    """Fields frozen after target selection; representation is deliberately absent."""

    return {
        "candidate_policy_id": CANDIDATE_POLICY_ID,
        **({"tears_ranking_policy": TEARS_RANKING_POLICY} if system == "TEARS" else {}),
        **({"summary_source_request_id": payload.summary_source_request_id}
           if system == "TEARS" and getattr(payload, "summary_source_request_id", None) else {}),
        "system": system,
        "liked_movie_ids": list(payload.liked_movie_ids),
        "preference_evidence": [
            movie.model_dump(mode="json") for movie in payload.preference_evidence
        ],
        "excluded_movie_ids": list(payload.excluded_movie_ids),
        "catalog_fingerprint": payload.catalog_fingerprint,
        "onboarding_fingerprint": payload.onboarding_fingerprint,
        "context": (payload.context or "").strip(),
        "alpha": payload.alpha,
        "top_k": payload.top_k,
        "min_release_year": payload.min_release_year,
        "year_filter": YEAR_FILTER_POLICY,
    }


def representation_snapshot(system: str, payload: Any) -> Any:
    return payload.summary if system == "TEARS" else list(payload.genres)


def target_observation(
    recommendations: list[dict[str, Any]], target_movie_id: int
) -> dict[str, Any]:
    item = next(
        (
            value
            for value in recommendations
            if int(value["movie_id"]) == int(target_movie_id)
        ),
        None,
    )
    if item is None:
        return {"state": "not_returned", "rank": None, "score": None}
    return {
        "state": "returned",
        "rank": int(item["rank"]),
        "score": float(item["score"]),
    }


def effective_tears_input(recommender: Any, summary: str) -> dict[str, Any]:
    encoded = recommender.tokenizer(
        [summary.strip()],
        padding="max_length",
        truncation=True,
        max_length=recommender.config.model.max_text_tokens,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids.detach().cpu().tolist()[0]
    attention_mask = encoded.attention_mask.detach().cpu().tolist()[0]
    return {
        "kind": "tokenized_text",
        "summary": summary.strip(),
        "max_text_tokens": recommender.config.model.max_text_tokens,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_signature": signature(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        ),
    }


def effective_gers_input(
    recommender: Any, genres: list[str], liked_movie_ids: list[int]
) -> dict[str, Any]:
    canonical_counts: Counter[str] = Counter()
    ignored: Counter[str] = Counter()
    for raw_genre in genres:
        value = str(raw_genre).strip()
        canonical = getattr(recommender, "genre_to_index", {}).get(value)
        if canonical is not None:
            canonical_counts[value] += 1
            continue
        aliases = {
            "family": "Children",
            "history": "Drama",
            "music": "Musical",
            "science fiction": "Sci-Fi",
            "sci fi": "Sci-Fi",
            "sci-fi": "Sci-Fi",
            "tv movie": "Drama",
        }
        mapped = aliases.get(value.casefold(), value)
        if mapped in getattr(recommender, "genre_to_index", {}):
            canonical_counts[mapped] += 1
        else:
            ignored[value] += 1
    total = sum(canonical_counts.values())
    normalized = {
        genre: count / total for genre, count in sorted(canonical_counts.items())
    } if total else {}
    valid_liked = [
        int(movie_id)
        for movie_id in liked_movie_ids
        if int(movie_id) in recommender.movie_to_item
    ]
    return {
        "kind": "hybrid_fixed_rating_and_genre_frequency",
        "genres_received": list(genres),
        "genre_counts": dict(sorted(canonical_counts.items())),
        "normalized_genre_weights": normalized,
        "ignored_genres": dict(sorted(ignored.items())),
        "interaction_entries": [
            {
                "movie_id": movie_id,
                "model_item_id": int(recommender.movie_to_item[movie_id]),
                "rating": 5.0,
            }
            for movie_id in valid_liked
        ],
        "selected_movie_rating_semantics": "fixed_5.0_nonzero",
    }


class StudyConflict(ValueError):
    """A submitted study transition violates the canonical trial contract."""


class StudyNotFound(ValueError):
    """A referenced baseline or trial does not exist."""


class StudyStore:
    """Append-oriented SQLite store with transactional trial enforcement."""

    def __init__(self, path: Path, provenance: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.provenance = dict(provenance)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def _initialize(self) -> None:
        with self._transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    system TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    trial_id TEXT,
                    attempt_number INTEGER,
                    representation_revision INTEGER NOT NULL,
                    request_id TEXT NOT NULL,
                    input_signature TEXT NOT NULL,
                    target_movie_id INTEGER,
                    status TEXT NOT NULL,
                    latency_ms REAL,
                    error_json TEXT,
                    provenance_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS events_request_type
                    ON events(request_id, event_type);
                CREATE INDEX IF NOT EXISTS events_trial
                    ON events(trial_id, attempt_number);

                CREATE TABLE IF NOT EXISTS generated_summary_cache (
                    participant_id TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    source_request_id TEXT NOT NULL,
                    PRIMARY KEY (participant_id, cache_key)
                );

                CREATE TABLE IF NOT EXISTS trials (
                    trial_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    system TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    baseline_request_id TEXT NOT NULL,
                    target_movie_id INTEGER NOT NULL,
                    baseline_rank INTEGER NOT NULL,
                    baseline_score REAL NOT NULL,
                    immutable_signature TEXT NOT NULL,
                    immutable_json TEXT NOT NULL,
                    baseline_representation_json TEXT NOT NULL,
                    latest_representation_signature TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    provenance_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS attempts (
                    trial_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    request_id TEXT NOT NULL UNIQUE,
                    input_signature TEXT NOT NULL,
                    representation_revision INTEGER NOT NULL,
                    representation_signature TEXT NOT NULL,
                    representation_json TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    target_state TEXT,
                    target_rank INTEGER,
                    target_score REAL,
                    latency_ms REAL,
                    error_json TEXT,
                    recommendations_json TEXT,
                    PRIMARY KEY (trial_id, attempt_number),
                    FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
                );

                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    recorded_at TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    system TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    trial_id TEXT,
                    request_id TEXT NOT NULL,
                    instrument_id TEXT NOT NULL,
                    instrument_version TEXT NOT NULL,
                    responses_json TEXT NOT NULL,
                    provenance_json TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _meta(meta: Any) -> dict[str, Any]:
        return meta.model_dump(mode="json") if hasattr(meta, "model_dump") else dict(meta)

    def log_event(
        self,
        event_type: str,
        meta: Any,
        status: str,
        payload: Mapping[str, Any],
        *,
        latency_ms: float | None = None,
        error: Mapping[str, Any] | None = None,
    ) -> str:
        values = self._meta(meta)
        event_id = str(uuid4())
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    event_type,
                    utc_now(),
                    values["participant_id"],
                    values["session_id"],
                    values["system"],
                    values["task_id"],
                    values.get("trial_id"),
                    values.get("attempt"),
                    values["representation_revision"],
                    values["request_id"],
                    values["input_signature"],
                    values.get("target_movie_id"),
                    status,
                    latency_ms,
                    canonical_json(error) if error else None,
                    canonical_json(self.provenance),
                    canonical_json(dict(payload)),
                ),
            )
        return event_id

    def event_by_request(
        self, request_id: str, event_type: str
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM events WHERE request_id = ? AND event_type = ?",
                (request_id, event_type),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        result["provenance"] = json.loads(result.pop("provenance_json"))
        return result

    def generated_summary(self, participant_id: str, cache_key: str):
        """Reuse validated generation for identical evidence within one participant."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT summary, source_request_id FROM generated_summary_cache "
                "WHERE participant_id = ? AND cache_key = ?",
                (participant_id, cache_key),
            ).fetchone()
        return dict(row) if row else None

    def remember_generated_summary(
        self, participant_id: str, cache_key: str, summary: str, source_request_id: str
    ):
        # Concurrent generations converge on the first validated result. Never
        # share personal context between participants or cache participant edits.
        with self._transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO generated_summary_cache VALUES (?, ?, ?, ?)",
                (participant_id, cache_key, summary, source_request_id),
            )
            row = connection.execute(
                "SELECT summary, source_request_id FROM generated_summary_cache "
                "WHERE participant_id = ? AND cache_key = ?",
                (participant_id, cache_key),
            ).fetchone()
        return dict(row)

    def synchronized_summary(self, meta: Any, source_id: str, display: str):
        """Reuse the same encoded profile for identical participant edits."""
        values = self._meta(meta)
        with self._connect() as connection:
            row = connection.execute(
                """SELECT payload_json FROM events
                   WHERE event_type = 'recommendation_result' AND status = 'completed'
                     AND participant_id = ? AND session_id = ? AND system = 'TEARS'
                     AND json_extract(payload_json, '$.effective_model_input.summary_source_request_id') = ?
                     AND json_extract(payload_json, '$.effective_model_input.display_summary') = ?
                   ORDER BY rowid DESC LIMIT 1""",
                (values["participant_id"], values["session_id"], source_id, display),
            ).fetchone()
        return json.loads(row[0])["effective_model_input"] if row else None

    def create_trial(
        self,
        meta: Any,
        baseline_request_id: str,
        target_movie_id: int,
    ) -> dict[str, Any]:
        values = self._meta(meta)
        expected_intent = "raise" if values["task_id"] == "2" else "lower"
        baseline = self.event_by_request(baseline_request_id, "recommendation_result")
        if baseline is None:
            raise StudyNotFound("The referenced baseline recommendation was not found")
        rendered_baseline = self.event_by_request(baseline_request_id, "ui_render")
        if rendered_baseline is None:
            raise StudyConflict(
                "The baseline must be durably recorded as rendered before target selection"
            )
        if (
            baseline["participant_id"] != values["participant_id"]
            or baseline["session_id"] != values["session_id"]
            or baseline["system"] != values["system"]
            or baseline["task_id"] != "1b"
        ):
            raise StudyConflict(
                "The baseline must be this participant/session/system's Task 1b result"
            )
        if baseline["representation_revision"] != values["representation_revision"]:
            raise StudyConflict(
                "Trial creation must preserve the baseline representation revision"
            )
        recommendations = baseline["payload"]["recommendations"]
        observation = target_observation(recommendations, target_movie_id)
        if observation["state"] != "returned":
            raise StudyConflict("The target must be selected from the baseline ranking")
        immutable = baseline["payload"]["immutable_trial_input"]
        if immutable.get("candidate_policy_id") != CANDIDATE_POLICY_ID:
            raise StudyConflict("Get a new baseline recommendation under the current candidate policy before starting a trial")
        if values["system"] == "TEARS" and immutable.get("tears_ranking_policy") != TEARS_RANKING_POLICY:
            raise StudyConflict("Get a new baseline recommendation under the current TEARS ranking policy before starting a trial")
        representation = baseline["payload"]["representation"]
        with self._transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO trials (
                        trial_id, created_at, participant_id, session_id, system,
                        task_id, intent, baseline_request_id, target_movie_id,
                        baseline_rank, baseline_score, immutable_signature,
                        immutable_json, baseline_representation_json,
                        latest_representation_signature, attempt_count, status,
                        provenance_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        0, 'active', ?
                    )
                    """,
                    (
                        values["trial_id"],
                        utc_now(),
                        values["participant_id"],
                        values["session_id"],
                        values["system"],
                        values["task_id"],
                        expected_intent,
                        baseline_request_id,
                        int(target_movie_id),
                        observation["rank"],
                        observation["score"],
                        signature(immutable),
                        canonical_json(immutable),
                        canonical_json(representation),
                        signature(representation),
                        canonical_json(self.provenance),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise StudyConflict("That trial ID already exists") from error
        return self.get_trial(values["trial_id"])

    def reserve_attempt(self, meta: Any, immutable: Any, representation: Any) -> None:
        values = self._meta(meta)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM trials WHERE trial_id = ?",
                (values["trial_id"],),
            ).fetchone()
            if row is None:
                raise StudyNotFound("The study trial does not exist")
            if any(
                (
                    row["participant_id"] != values["participant_id"],
                    row["session_id"] != values["session_id"],
                    row["system"] != values["system"],
                    row["task_id"] != values["task_id"],
                    row["target_movie_id"] != values.get("target_movie_id"),
                )
            ):
                raise StudyConflict("Study metadata does not match the immutable trial")
            if row["status"] == "limit_reached":
                raise StudyConflict("The hard maximum of five submitted edits was reached")
            if row["status"] != "active":
                raise StudyConflict("The study trial is not active")
            expected_attempt = int(row["attempt_count"]) + 1
            if expected_attempt > MAX_PROFILE_EDIT_ATTEMPTS:
                raise StudyConflict("The hard maximum of five submitted edits was reached")
            if values.get("attempt") != expected_attempt:
                raise StudyConflict(f"The next submitted edit must be attempt {expected_attempt}")
            if row["immutable_signature"] != signature(immutable):
                raise StudyConflict("An immutable baseline field changed during the trial")
            representation_signature = signature(representation)
            if row["latest_representation_signature"] == representation_signature:
                raise StudyConflict("A submitted attempt must change the profile representation")
            connection.execute(
                """
                INSERT INTO attempts (
                    trial_id, attempt_number, request_id, input_signature,
                    representation_revision, representation_signature,
                    representation_json, submitted_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'submitted')
                """,
                (
                    values["trial_id"],
                    expected_attempt,
                    values["request_id"],
                    values["input_signature"],
                    values["representation_revision"],
                    representation_signature,
                    canonical_json(representation),
                    utc_now(),
                ),
            )
            connection.execute(
                """
                UPDATE trials
                SET attempt_count = ?, latest_representation_signature = ?,
                    status = CASE WHEN ? = ? THEN 'limit_reached' ELSE status END
                WHERE trial_id = ?
                """,
                (
                    expected_attempt,
                    representation_signature,
                    expected_attempt,
                    MAX_PROFILE_EDIT_ATTEMPTS,
                    values["trial_id"],
                ),
            )

    def complete_attempt(
        self,
        trial_id: str,
        attempt: int,
        recommendations: list[dict[str, Any]],
        latency_ms: float,
    ) -> dict[str, Any]:
        with self._transaction() as connection:
            target = connection.execute(
                "SELECT target_movie_id FROM trials WHERE trial_id = ?",
                (trial_id,),
            ).fetchone()
            if target is None:
                raise StudyNotFound("The study trial does not exist")
            observation = target_observation(
                recommendations, int(target["target_movie_id"])
            )
            connection.execute(
                """
                UPDATE attempts SET completed_at = ?, status = 'completed',
                    target_state = ?, target_rank = ?, target_score = ?,
                    latency_ms = ?, recommendations_json = ?
                WHERE trial_id = ? AND attempt_number = ?
                """,
                (
                    utc_now(),
                    observation["state"],
                    observation["rank"],
                    observation["score"],
                    latency_ms,
                    canonical_json(recommendations),
                    trial_id,
                    attempt,
                ),
            )
        return self.get_trial(trial_id)

    def fail_attempt(
        self, trial_id: str, attempt: int, error: Mapping[str, Any], latency_ms: float
    ) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE attempts SET completed_at = ?, status = 'failed',
                    latency_ms = ?, error_json = ?
                WHERE trial_id = ? AND attempt_number = ?
                """,
                (utc_now(), latency_ms, canonical_json(error), trial_id, attempt),
            )

    def get_trial(self, trial_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            trial = connection.execute(
                "SELECT * FROM trials WHERE trial_id = ?", (trial_id,)
            ).fetchone()
            if trial is None:
                raise StudyNotFound("The study trial does not exist")
            attempts = connection.execute(
                """
                SELECT attempt_number, request_id, representation_revision,
                    submitted_at, completed_at, status, target_state,
                    target_rank, target_score, latency_ms, error_json
                FROM attempts WHERE trial_id = ? ORDER BY attempt_number
                """,
                (trial_id,),
            ).fetchall()
        history = []
        for row in attempts:
            item = dict(row)
            if item["error_json"]:
                item["error"] = json.loads(item.pop("error_json"))
            else:
                item.pop("error_json")
                item["error"] = None
            history.append(item)
        return {
            "trial_id": trial["trial_id"],
            "system": trial["system"],
            "task_id": trial["task_id"],
            "intent": trial["intent"],
            "target_movie_id": trial["target_movie_id"],
            "baseline_request_id": trial["baseline_request_id"],
            "baseline_rank": trial["baseline_rank"],
            "baseline_score": trial["baseline_score"],
            "attempt_count": trial["attempt_count"],
            "attempts_remaining": MAX_PROFILE_EDIT_ATTEMPTS
            - int(trial["attempt_count"]),
            "max_attempts": MAX_PROFILE_EDIT_ATTEMPTS,
            "status": trial["status"],
            "history": history,
        }

    def close_trial(self, meta: Any) -> dict[str, Any]:
        values = self._meta(meta)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM trials WHERE trial_id = ?", (values["trial_id"],)
            ).fetchone()
            if row is None:
                raise StudyNotFound("The study trial does not exist")
            if any(
                (
                    row["participant_id"] != values["participant_id"],
                    row["session_id"] != values["session_id"],
                    row["system"] != values["system"],
                    row["task_id"] != values["task_id"],
                    row["target_movie_id"] != values.get("target_movie_id"),
                )
            ):
                raise StudyConflict("Study metadata does not match the trial")
            if row["status"] not in {"active", "limit_reached"}:
                raise StudyConflict("The study trial is not active")
            pending = connection.execute(
                "SELECT 1 FROM attempts WHERE trial_id = ? AND status = 'submitted' LIMIT 1",
                (values["trial_id"],),
            ).fetchone()
            if pending is not None:
                raise StudyConflict(
                    "The study trial cannot close while an edit attempt is in flight"
                )
            connection.execute(
                "UPDATE trials SET status = 'completed' WHERE trial_id = ?",
                (values["trial_id"],),
            )
        return self.get_trial(values["trial_id"])

    def persist_evaluation(
        self,
        meta: Any,
        instrument_id: str,
        instrument_version: str,
        responses: Mapping[str, Any],
    ) -> str:
        values = self._meta(meta)
        evaluation_id = str(uuid4())
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    utc_now(),
                    values["participant_id"],
                    values["session_id"],
                    values["system"],
                    values["task_id"],
                    values.get("trial_id"),
                    values["request_id"],
                    instrument_id,
                    instrument_version,
                    canonical_json(dict(responses)),
                    canonical_json(self.provenance),
                ),
            )
        return evaluation_id

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("events", "trials", "attempts", "evaluations")
            }
