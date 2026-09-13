import json
import sqlite3

from scripts.run_frozen_v10_remaining_shards import (
    attempt_rows,
    existing_final_audit,
    promote_objectively_valid_rows,
    selected_rows,
    sha256,
    status_count,
)


def _row(user_id: int, attempt: int, failures: list[str]) -> dict:
    return {
        "user_id": user_id,
        "shard": 0,
        "attempt": attempt,
        "custom_id": f"user-{user_id}-r{attempt}",
        "raw_summary": f"Viewer {user_id} has a distinct preference profile with enough descriptive words for validation.",
        "final_summary": f"Viewer {user_id} has a distinct preference profile with enough descriptive words for validation.",
        "raw_word_count": 20,
        "final_word_count": 20,
        "cleanup_operations": [],
        "failures": failures,
        "input_tokens": 100,
        "cached_input_tokens": 0,
        "output_tokens": 20,
        "reasoning_tokens": 0,
        "cost_usd": 0.01,
    }


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_targeted_retry_promotes_valid_rows_without_resubmitting_them(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "production.sqlite3"
    checkpoint.parent.mkdir(parents=True)
    with sqlite3.connect(checkpoint) as database:
        database.executescript(
            """
            CREATE TABLE accepted(
              user_id INTEGER PRIMARY KEY, shard INTEGER, attempt INTEGER, custom_id TEXT,
              raw_summary TEXT, final_summary TEXT, normalized_hash TEXT, simhash TEXT,
              input_tokens INTEGER, cached_input_tokens INTEGER, output_tokens INTEGER,
              reasoning_tokens INTEGER, cost_usd REAL, cleanup_json TEXT, accepted_at INTEGER
            );
            CREATE TABLE attempts(
              shard INTEGER, attempt INTEGER, user_id INTEGER, status TEXT,
              reasons_json TEXT, custom_id TEXT, PRIMARY KEY(shard, attempt, user_id)
            );
            CREATE TABLE near_buckets(
              bucket TEXT, user_id INTEGER, PRIMARY KEY(bucket, user_id)
            );
            """
        )
        database.executemany(
            "INSERT INTO attempts VALUES(?,?,?,?,?,?)",
            [
                (0, 0, 1, "accepted", "[]", "user-1-r0"),
                (0, 0, 2, "failed", '["categorical_negative_on_mixed_genre"]', "user-2-r0"),
                (0, 0, 3, "failed", '["literal_placeholder"]', "user-3-r0"),
            ],
        )

    attempt_zero = [
        _row(1, 0, []),
        _row(2, 0, ["categorical_negative_on_mixed_genre"]),
        _row(3, 0, ["literal_placeholder"]),
    ]
    _write_rows(
        tmp_path / "validated" / "attempts" / "shard-000-attempt-00-accepted.jsonl",
        attempt_zero[:1],
    )
    _write_rows(
        tmp_path / "validated" / "attempts" / "shard-000-attempt-00-failed.jsonl",
        attempt_zero[1:],
    )

    assert promote_objectively_valid_rows(tmp_path, 0, attempt_rows(tmp_path, 0, 0)) == 2
    with sqlite3.connect(checkpoint) as database:
        assert database.execute("SELECT user_id FROM accepted ORDER BY user_id").fetchall() == [(1,), (2,)]

        database.execute(
            "INSERT INTO attempts VALUES(?,?,?,?,?,?)",
            (0, 1, 3, "accepted", "[]", "user-3-r1"),
        )
        database.commit()
    retry = _row(3, 1, [])
    _write_rows(
        tmp_path / "validated" / "attempts" / "shard-000-attempt-01-accepted.jsonl",
        [retry],
    )
    _write_rows(
        tmp_path / "validated" / "attempts" / "shard-000-attempt-01-failed.jsonl",
        [],
    )

    assert promote_objectively_valid_rows(tmp_path, 0, [retry]) == 1
    selected = selected_rows(tmp_path, 0, 3)
    assert [(row["user_id"], row["attempt"]) for row in selected] == [(1, 0), (2, 0), (3, 1)]


def test_status_count_accepts_legacy_shard0_field_names():
    legacy = {
        "accepted_users": 15000,
        "api_failed": 0,
        "objective_validation": {"allowed_cleanup_users": 44},
    }
    modern = {
        "accepted_users": 5948,
        "api_failures": 0,
        "objective_failures": 0,
        "cleanup_users": 38,
    }
    assert status_count(legacy, "api_failures", "api_failed") == 0
    assert status_count(legacy, "objective_failures") == 0
    assert status_count(
        legacy,
        "cleanup_users",
        default=int((legacy.get("objective_validation") or {}).get("allowed_cleanup_users") or 0),
    ) == 44
    assert status_count(modern, "api_failures", "api_failed") == 0
    assert status_count(modern, "cleanup_users") == 38


def test_existing_final_audit_reuses_complete_corpus(tmp_path):
    corpus = tmp_path / "validated" / "final_summaries.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text('{"summary":"ok","user_id":1}\n{"summary":"ok","user_id":2}\n', encoding="utf-8")
    (tmp_path / "validated" / "raw_summaries.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (tmp_path / "validated" / "all_validated_summaries.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    audit = {
        "corpus_sha256": sha256(corpus),
        "expected_summaries": 2,
        "failures_remaining": 0,
        "final_summaries_path": str(corpus),
        "valid_summaries": 2,
    }
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "final_audit_report.json").write_text(json.dumps(audit), encoding="utf-8")
    assert existing_final_audit(tmp_path, {"users": 2})["valid_summaries"] == 2
    assert existing_final_audit(tmp_path, {"users": 3}) is None
