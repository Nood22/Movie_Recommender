from __future__ import annotations

import inspect
import json
from pathlib import Path
import sqlite3

from tears_training import evidence_gated_full_cohort as production
from tears_training.evidence_gated_summary_harness import build_response_request


def test_frozen_v10_bytes_and_protocol_are_enforced() -> None:
    result = production._frozen_v10_checks()
    assert result["pass"], result
    assert production.MODEL == "gpt-5-mini-2025-08-07"


def test_production_harness_has_no_v12_tmdb_or_semantic_repair() -> None:
    source = inspect.getsource(production)
    assert "evidence_gated_summary_harness_v12" not in source
    assert "deterministic_negative" not in inspect.getsource(build_response_request)
    assert "semantic_repair(" not in source
    assert production.MAX_RETRIES == 2


def test_semantic_gate_respects_v10_adjudication_policy() -> None:
    evidence = {
        "negative_evidence_status": "NONE",
        "genre_classifications": {"supported_negative": [], "mixed_conflicting": ["Action"]},
    }
    failures, _ = production._semantic_failures(
        "Summary: The user likes Action because of genuinely positive examples. No clear negative preference is evident.",
        evidence,
    )
    assert failures == []
    failures, _ = production._semantic_failures(
        "Summary: The user likes dramas. The user dislikes Action.", evidence
    )
    assert "fabricated_dislike_for_none" in failures


def test_retry_excludes_every_successful_user(tmp_path: Path) -> None:
    root = tmp_path
    plan = {
        "fingerprint": "x", "shards": [{"shard": 0, "users": 2,
        "request_path": str(root / "requests" / "shard-000-attempt-00.jsonl"),
        "manifest_path": str(root / "manifests" / "shard-000-attempt-00.jsonl")}],
    }
    production._initialize_state(root, plan)
    request_path, manifest_path, _ = production._request_paths(root, 0, 0)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(
        json.dumps({"custom_id": "a", "body": {"input": []}}) + "\n"
        + json.dumps({"custom_id": "b", "body": {"input": []}}) + "\n"
    )
    manifest_path.write_text(
        json.dumps({"user_id": 1, "base_custom_id": "a"}) + "\n"
        + json.dumps({"user_id": 2, "base_custom_id": "b"}) + "\n"
    )
    with sqlite3.connect(production._db_path(root)) as db:
        db.execute(
            "INSERT INTO accepted VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (1, 0, 0, "a", "raw", "Summary: final", "hash", "0", 1, 0, 1, 0, 0.0, "[]", 0),
        )
        db.commit()
    retry_path, retry_manifest, count = production._prepare_attempt(root, plan, 0, 1)
    assert count == 1
    assert [row["custom_id"] for row in production._jsonl(retry_path)] == ["b-r1"]
    assert [row["user_id"] for row in production._jsonl(retry_manifest)] == [2]


def test_shard_plan_stays_below_batch_limits() -> None:
    sizes = production._shard_sizes()
    assert sum(sizes) == 200_948
    assert max(sizes) == 15_000
    assert len(sizes) == 14
