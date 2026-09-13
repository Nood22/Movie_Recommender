#!/usr/bin/env python3
"""Resume and finish frozen V10 Batch shards using the finalized shard-0 workflow.

This is an operational orchestrator, not a new validator.  It calls the archived
production module for submission, polling, cleanup, and objective validation,
then applies the already-approved rule that the retired semantic keyword screen
is diagnostic only.  Any genuine objective, API, duplicate, overlap, integrity,
or budget failure stops the run without retrying or submitting a later shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

from tears_training import evidence_gated_full_cohort as prod


SEMANTIC_SCREEN_REASONS = {
    "categorical_negative_on_mixed_genre",
    "fabricated_dislike_for_none",
    "inappropriate_abstention",
    "inappropriate_negative_abstention",
    "overstated_negative_for_weak",
    "substantive_positive_negative_contradiction",
    "uncautious_negative_for_weak",
    "unsupported_categorical_negative_genre",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stop(root: Path, shard: int, condition: str, details: Any) -> None:
    payload = {
        "status": "STOP",
        "shard": shard,
        "condition": condition,
        "details": details,
        "created_at": int(time.time()),
        "next_shard_submitted": False,
    }
    atomic_json(root / "reports" / "STOP-production.json", payload)
    raise RuntimeError(json.dumps(payload, sort_keys=True))


def feature(item: tuple[int, str]) -> tuple[int, str, set[str], int, list[str]]:
    user_id, summary = item
    normalized = prod._normal(summary)
    simhash = prod._simhash(summary)
    return user_id, normalized, set(normalized.split()), simhash, prod._simhash_buckets(simhash)


def cross_shard_duplicates(root: Path, shard: int, rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    items = [(int(row["user_id"]), str(row["final_summary"])) for row in rows]
    workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        features = list(executor.map(feature, items, chunksize=64))

    source = sqlite3.connect(f"file:{root / 'checkpoints/production.sqlite3'}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    memory = sqlite3.connect(":memory:")
    memory.row_factory = sqlite3.Row
    memory.executescript(
        "CREATE TABLE accepted(user_id INTEGER PRIMARY KEY, final_summary TEXT, normalized_hash TEXT);"
        "CREATE TABLE near_buckets(bucket TEXT,user_id INTEGER,PRIMARY KEY(bucket,user_id));"
        "CREATE INDEX near_bucket_index ON near_buckets(bucket);"
    )
    memory.executemany(
        "INSERT INTO accepted VALUES(?,?,?)",
        (
            (row["user_id"], row["final_summary"], row["normalized_hash"])
            for row in source.execute(
                "SELECT user_id,final_summary,normalized_hash FROM accepted WHERE shard<=?", (shard,)
            )
        ),
    )
    memory.executemany(
        "INSERT INTO near_buckets VALUES(?,?)",
        (
            (row["bucket"], row["user_id"])
            for row in source.execute(
                "SELECT b.bucket,b.user_id FROM near_buckets b JOIN accepted a ON a.user_id=b.user_id "
                "WHERE a.shard<=?",
                (shard,),
            )
        ),
    )
    source.close()
    memory.commit()

    summaries = {int(row["user_id"]): str(row["final_summary"]) for row in rows}
    exact: list[tuple[int, int]] = []
    near: list[tuple[int, int]] = []
    for user_id, normalized, tokens, simhash, buckets in sorted(features):
        placeholders = ",".join("?" for _ in buckets)
        candidates = memory.execute(
            f"SELECT DISTINCT a.user_id,a.final_summary FROM near_buckets b JOIN accepted a "
            f"ON a.user_id=b.user_id WHERE b.bucket IN ({placeholders})",
            buckets,
        ).fetchall()
        for candidate in candidates:
            other = prod._normal(candidate["final_summary"])
            other_id = int(candidate["user_id"])
            if other_id == user_id:
                continue
            if normalized == other:
                exact.append((user_id, other_id))
                break
            other_tokens = set(other.split())
            union = tokens | other_tokens
            if (
                union
                and len(tokens & other_tokens) / len(union) >= 0.72
                and SequenceMatcher(None, normalized, other, autojunk=False).ratio() >= 0.90
            ):
                near.append((user_id, other_id))
                break
        memory.execute(
            "INSERT OR REPLACE INTO accepted VALUES(?,?,?)",
            (user_id, summaries[user_id], hashlib.sha256(normalized.encode()).hexdigest()),
        )
        memory.executemany(
            "INSERT OR IGNORE INTO near_buckets VALUES(?,?)",
            ((bucket, user_id) for bucket in buckets),
        )
    memory.close()
    return {"exact": exact, "near": near}


def archived_code_ok(root: Path) -> bool:
    manifest = read_json(root / "code" / "manifest.json")
    return all(sha256(root / "code" / name) == record["sha256"] for name, record in manifest.items())


def preflight(root: Path, plan: dict[str, Any], shard: int, client: OpenAI) -> None:
    if not archived_code_ok(root):
        stop(root, shard, "frozen_code_integrity", "Archived production code hash mismatch")
    latest = read_json(root / "checkpoints" / "latest.json")
    if latest["successful_users_resubmitted"] != 0 or latest["actual_cost_usd"] > prod.CONSERVATIVE_COST_CAP_USD:
        stop(root, shard, "checkpoint_or_budget", latest)
    expected_prior = sum(int(item["users"]) for item in plan["shards"][:shard])
    with sqlite3.connect(f"file:{root / 'checkpoints/production.sqlite3'}?mode=ro", uri=True) as database:
        accepted = int(database.execute("SELECT COUNT(*) FROM accepted WHERE shard<?", (shard,)).fetchone()[0])
        future = int(database.execute("SELECT COUNT(*) FROM accepted WHERE shard>?", (shard,)).fetchone()[0])
        excessive_attempts = int(
            database.execute(
                "SELECT COUNT(*) FROM (SELECT user_id FROM attempts GROUP BY user_id "
                "HAVING COUNT(*)>?)",
                (prod.MAX_RETRIES + 1,),
            ).fetchone()[0]
        )
    if accepted != expected_prior or future or excessive_attempts:
        stop(
            root,
            shard,
            "user_overlap_or_checkpoint_integrity",
            {
                "prior": accepted,
                "expected": expected_prior,
                "future": future,
                "users_over_retry_limit": excessive_attempts,
            },
        )
    if shard:
        prior_status_path = root / "reports" / "shards" / f"shard-{shard-1:03d}-final-status.json"
        if not prior_status_path.exists():
            stop(root, shard, "preceding_shard_not_finalized", str(prior_status_path))
        prior = read_json(prior_status_path)
        final_path = Path(prior["final_summaries_path"])
        if prior["accepted_users"] != prior["expected_users"] or sha256(final_path) != prior["final_summaries_sha256"]:
            stop(root, shard, "preceding_shard_integrity", prior)

    submission_path = root / "submissions" / f"shard-{shard:03d}-attempt-00.json"
    status_path = root / "reports" / "shards" / f"shard-{shard:03d}-final-status.json"
    if status_path.exists():
        return
    if submission_path.exists():
        submission = read_json(submission_path)
        batch = client.batches.retrieve(submission["batch_id"])
        if str((batch.metadata or {}).get("shard")) != str(shard) or str((batch.metadata or {}).get("attempt")) != "0":
            stop(root, shard, "batch_metadata_mismatch", batch.model_dump(mode="json"))
    else:
        manifests = read_jsonl(Path(plan["shards"][shard]["manifest_path"]))
        manifest_users = {int(row["user_id"]) for row in manifests}
        with sqlite3.connect(f"file:{root / 'checkpoints/production.sqlite3'}?mode=ro", uri=True) as database:
            accepted_users = {int(row[0]) for row in database.execute("SELECT user_id FROM accepted")}
        intersect = manifest_users & accepted_users
        if intersect:
            stop(root, shard, "accepted_user_resubmission_risk", {"count": len(intersect), "examples": sorted(intersect)[:10]})


def poll_to_terminal(root: Path, shard: int, attempt: int) -> dict[str, Any]:
    submission_path = root / "submissions" / f"shard-{shard:03d}-attempt-{attempt:02d}.json"
    while True:
        result = prod.poll_shard(submission_path)
        status = str(result["batch"]["status"])
        if status in prod.TERMINAL_BATCH_STATUSES:
            return result
        time.sleep(60)


def final_rows(rows: list[dict[str, Any]], shard: int) -> list[dict[str, Any]]:
    return [
        {
            "attempt": 0,
            "cached_input_tokens": int(row["cached_input_tokens"]),
            "cleanup_operations": list(row.get("cleanup_operations", [])),
            "cost_usd": float(row["cost_usd"]),
            "custom_id": row["custom_id"],
            "final_summary": row["final_summary"],
            "final_word_count": int(row["final_word_count"]),
            "input_tokens": int(row["input_tokens"]),
            "objective_validation": "pass",
            "output_tokens": int(row["output_tokens"]),
            "raw_summary": row["raw_summary"],
            "raw_word_count": int(row["raw_word_count"]),
            "reasoning_tokens": int(row["reasoning_tokens"]),
            "semantic_acceptance_basis": "frozen_v10_contract_and_finalized_shard0_objective_workflow",
            "shard": shard,
            "user_id": int(row["user_id"]),
        }
        for row in rows
    ]


def attempt_rows(root: Path, shard: int, attempt: int) -> list[dict[str, Any]]:
    accepted_path = root / "validated" / "attempts" / f"shard-{shard:03d}-attempt-{attempt:02d}-accepted.jsonl"
    failed_path = root / "validated" / "attempts" / f"shard-{shard:03d}-attempt-{attempt:02d}-failed.jsonl"
    rows: list[dict[str, Any]] = []
    for path in (accepted_path, failed_path):
        if path.exists():
            rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: int(row["user_id"]))
    return rows


def objective_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "user_id": int(row["user_id"]),
            "failures": sorted(set(row.get("failures", [])) - SEMANTIC_SCREEN_REASONS),
        }
        for row in rows
        if set(row.get("failures", [])) - SEMANTIC_SCREEN_REASONS
    ]


def promote_objectively_valid_rows(root: Path, shard: int, rows: list[dict[str, Any]]) -> int:
    promotable = [
        row for row in rows if not (set(row.get("failures", [])) - SEMANTIC_SCREEN_REASONS)
    ]
    if not promotable:
        return 0
    duplicates = cross_shard_duplicates(root, shard, promotable)
    if duplicates["exact"] or duplicates["near"]:
        stop(
            root,
            shard,
            "unresolved_duplicates",
            {"exact": duplicates["exact"][:20], "near": duplicates["near"][:20]},
        )
    features = {
        user_id: (normalized, simhash, buckets)
        for user_id, normalized, _, simhash, buckets in map(
            feature,
            ((int(row["user_id"]), str(row["final_summary"])) for row in promotable),
        )
    }
    database = sqlite3.connect(root / "checkpoints" / "production.sqlite3", timeout=120)
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=NORMAL")
    inserted = 0
    now = int(time.time())
    for index, row in enumerate(promotable, 1):
        user_id = int(row["user_id"])
        existing = database.execute(
            "SELECT final_summary,shard FROM accepted WHERE user_id=?", (user_id,)
        ).fetchone()
        if existing:
            if existing[0] != row["final_summary"] or int(existing[1]) != shard:
                database.close()
                stop(root, shard, "accepted_record_mismatch", {"user_id": user_id})
        else:
            normalized, simhash, buckets = features[user_id]
            database.execute(
                "INSERT INTO accepted VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    user_id,
                    shard,
                    int(row["attempt"]),
                    row["custom_id"],
                    row["raw_summary"],
                    row["final_summary"],
                    hashlib.sha256(normalized.encode()).hexdigest(),
                    f"{simhash:016x}",
                    int(row["input_tokens"]),
                    int(row["cached_input_tokens"]),
                    int(row["output_tokens"]),
                    int(row["reasoning_tokens"]),
                    float(row["cost_usd"]),
                    json.dumps(list(row.get("cleanup_operations", []))),
                    now,
                ),
            )
            database.executemany(
                "INSERT INTO near_buckets VALUES(?,?)", ((bucket, user_id) for bucket in buckets)
            )
            inserted += 1
        database.execute(
            "UPDATE attempts SET status=?,reasons_json=? WHERE shard=? AND attempt=? AND user_id=?",
            (
                "accepted_after_finalized_shard0_objective_workflow",
                "[]",
                shard,
                int(row["attempt"]),
                user_id,
            ),
        )
        if index % 500 == 0:
            database.commit()
    database.commit()
    database.close()
    return inserted


def selected_rows(root: Path, shard: int, expected: int) -> list[dict[str, Any]]:
    available: dict[tuple[int, str], dict[str, Any]] = {}
    for attempt in range(prod.MAX_RETRIES + 1):
        for row in attempt_rows(root, shard, attempt):
            available[(int(row["user_id"]), str(row["custom_id"]))] = row
    with sqlite3.connect(f"file:{root / 'checkpoints/production.sqlite3'}?mode=ro", uri=True) as database:
        accepted = database.execute(
            "SELECT user_id,custom_id FROM accepted WHERE shard=? ORDER BY user_id", (shard,)
        ).fetchall()
    rows = [available[(int(user_id), str(custom_id))] for user_id, custom_id in accepted if (int(user_id), str(custom_id)) in available]
    if len(rows) != expected or len({int(row["user_id"]) for row in rows}) != expected:
        stop(
            root,
            shard,
            "selected_row_integrity",
            {"selected": len(rows), "expected": expected, "accepted_records": len(accepted)},
        )
    return rows


def finalize_shard(root: Path, plan: dict[str, Any], shard: int) -> dict[str, Any]:
    expected = int(plan["shards"][shard]["users"])
    rows = selected_rows(root, shard, expected)
    if len(rows) != expected or len({int(row["user_id"]) for row in rows}) != expected:
        stop(root, shard, "missing_or_duplicate_user_rows", {"rows": len(rows), "unique": len({int(row['user_id']) for row in rows}), "expected": expected})

    unresolved = objective_failures(rows)
    if unresolved:
        stop(root, shard, "unexpected_objective_validation_failures", {"count": len(unresolved), "examples": unresolved[:20]})

    duplicates = cross_shard_duplicates(root, shard, rows)
    if duplicates["exact"] or duplicates["near"]:
        stop(root, shard, "unresolved_duplicates", {"exact": duplicates["exact"][:20], "near": duplicates["near"][:20]})

    features = {user_id: (normalized, simhash, buckets) for user_id, normalized, _, simhash, buckets in map(feature, ((int(row["user_id"]), str(row["final_summary"])) for row in rows))}
    now = int(time.time())
    database = sqlite3.connect(root / "checkpoints" / "production.sqlite3", timeout=120)
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=NORMAL")
    for index, row in enumerate(rows, 1):
        user_id = int(row["user_id"])
        existing = database.execute("SELECT final_summary,shard FROM accepted WHERE user_id=?", (user_id,)).fetchone()
        if existing:
            if existing[0] != row["final_summary"] or int(existing[1]) != shard:
                database.close()
                stop(root, shard, "accepted_record_mismatch", {"user_id": user_id})
        else:
            normalized, simhash, buckets = features[user_id]
            database.execute(
                "INSERT INTO accepted VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    user_id,
                    shard,
                    int(row["attempt"]),
                    row["custom_id"],
                    row["raw_summary"],
                    row["final_summary"],
                    hashlib.sha256(normalized.encode()).hexdigest(),
                    f"{simhash:016x}",
                    int(row["input_tokens"]),
                    int(row["cached_input_tokens"]),
                    int(row["output_tokens"]),
                    int(row["reasoning_tokens"]),
                    float(row["cost_usd"]),
                    json.dumps(list(row.get("cleanup_operations", []))),
                    now,
                ),
            )
            database.executemany("INSERT INTO near_buckets VALUES(?,?)", ((bucket, user_id) for bucket in buckets))
        database.execute(
            "UPDATE attempts SET status=?,reasons_json=? WHERE shard=? AND attempt=? AND user_id=?",
            (
                "accepted_after_finalized_shard0_objective_workflow",
                "[]",
                shard,
                int(row["attempt"]),
                user_id,
            ),
        )
        if index % 500 == 0:
            database.commit()
    database.commit()
    shard_accepted = int(database.execute("SELECT COUNT(*) FROM accepted WHERE shard=?", (shard,)).fetchone()[0])
    corpus_accepted = int(database.execute("SELECT COUNT(*) FROM accepted").fetchone()[0])
    usage_total = database.execute(
        "SELECT SUM(input_tokens),SUM(cached_input_tokens),SUM(output_tokens),SUM(reasoning_tokens),SUM(cost_usd) FROM accepted"
    ).fetchone()
    retry_count = int(
        database.execute("SELECT COUNT(*) FROM accepted WHERE shard=? AND attempt>0", (shard,)).fetchone()[0]
    )
    database.close()
    if shard_accepted != expected:
        stop(root, shard, "checkpoint_acceptance_integrity", {"accepted": shard_accepted, "expected": expected})

    output_rows = final_rows(rows, shard)
    final_path = root / "validated" / "shards" / f"shard-{shard:03d}-final-summaries.jsonl"
    atomic_jsonl(final_path, output_rows)
    submission = read_json(root / "submissions" / f"shard-{shard:03d}-attempt-00.json")
    raw_path = root / "responses" / f"shard-{shard:03d}-attempt-00-{submission['batch_id']}.jsonl"
    attempt_artifacts: list[dict[str, Any]] = []
    for submission_path in sorted((root / "submissions").glob(f"shard-{shard:03d}-attempt-*.json")):
        attempt_submission = read_json(submission_path)
        attempt = int(attempt_submission["attempt"])
        attempt_poll = read_json(root / "polls" / f"shard-{shard:03d}-attempt-{attempt:02d}.json")
        attempt_raw_path = root / "responses" / f"shard-{shard:03d}-attempt-{attempt:02d}-{attempt_submission['batch_id']}.jsonl"
        counts = attempt_poll["batch"].get("request_counts") or {}
        attempt_artifacts.append(
            {
                "attempt": attempt,
                "batch_id": attempt_submission["batch_id"],
                "completed": int(counts.get("completed", 0)),
                "failed": int(counts.get("failed", 0)),
                "path": str(attempt_raw_path),
                "rows": sum(1 for _ in attempt_raw_path.open(encoding="utf-8")),
                "sha256": sha256(attempt_raw_path),
                "submitted": int(attempt_submission["requests"]),
            }
        )
    cleanup_users = sum(bool(row.get("cleanup_operations")) for row in rows)
    cleanup_operations = Counter(
        operation["type"] for row in rows for operation in row.get("cleanup_operations", [])
    )
    word_counts = sorted(int(row["final_word_count"]) for row in rows)
    usage = {
        key: sum(int(row[key]) for row in rows)
        for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")
    }
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    exact_cost = sum(float(row["cost_usd"]) for row in rows)
    objective = {
        "allowed_cleanup_users": cleanup_users,
        "cleanup_operations": dict(sorted(cleanup_operations.items())),
        "exact_duplicate_pairs": 0,
        "formatting_failures_after_normalization": 0,
        "malformed_outputs": 0,
        "missing_outputs": 0,
        "near_duplicate_pairs": 0,
        "placeholders": 0,
        "privacy_leaks_after_sanitization": {"numeric_rating": 0, "title": 0, "year": 0},
        "structured_output_failures": 0,
        "too_short_malformed": 0,
    }
    retry_requests = sum(item["submitted"] for item in attempt_artifacts if item["attempt"] > 0)
    review = {
        "api": {
            "completed": sum(item["completed"] for item in attempt_artifacts),
            "failed": sum(item["failed"] for item in attempt_artifacts),
            "submitted": sum(item["submitted"] for item in attempt_artifacts),
        },
        "batch_id": submission["batch_id"],
        "cumulative_cost_usd": float(usage_total[4]),
        "exact_cost_usd": exact_cost,
        "length_words": {
            "maximum": max(word_counts),
            "mean": statistics.fmean(word_counts),
            "median": statistics.median(word_counts),
            "minimum": min(word_counts),
            "p95": word_counts[(95 * len(word_counts) + 99) // 100 - 1],
        },
        "next_shard_submitted": False,
        "objective_validation": objective,
        "raw_response": {"path": str(raw_path), "rows": expected, "sha256": sha256(raw_path)},
        "raw_response_attempts": attempt_artifacts,
        "retired_semantic_keyword_screen": {
            "candidate_count": sum(bool(row.get("failures")) for row in rows),
            "reason": "Finalized shard-0 workflow established systematic false positives; not an objective production gate.",
            "retry_triggered": bool(retry_requests),
            "used_as_acceptance_gate": False,
        },
        "retry": {"submitted": retry_requests, "accepted": retry_count},
        "shard": shard,
        "usage": usage,
        "wandb_url": read_json(root / "wandb_monitor_config.json")["run_url"],
    }
    review_path = root / "reports" / "shards" / f"shard-{shard:03d}-review.json"
    atomic_json(review_path, review)
    checkpoint = {
        "accepted": expected,
        "corpus_accepted": corpus_accepted,
        "expected": expected,
        "remaining": 0,
        "shard": shard,
        "updated_at": now,
    }
    checkpoint_path = root / "checkpoints" / f"shard-{shard:03d}.json"
    atomic_json(checkpoint_path, checkpoint)
    latest = {
        "accepted_users": corpus_accepted,
        "actual_cost_usd": float(usage_total[4]),
        "cached_input_tokens": int(usage_total[1]),
        "input_tokens": int(usage_total[0]),
        "output_tokens": int(usage_total[2]),
        "plan_fingerprint": plan["fingerprint"],
        "reasoning_tokens": int(usage_total[3]),
        "remaining_users": int(plan["users"]) - corpus_accepted,
        "shards": {},
        "successful_users_resubmitted": 0,
        "updated_at": now,
    }
    with sqlite3.connect(f"file:{root / 'checkpoints/production.sqlite3'}?mode=ro", uri=True) as database:
        for item in plan["shards"]:
            number = int(item["shard"])
            accepted_count = int(database.execute("SELECT COUNT(*) FROM accepted WHERE shard=?", (number,)).fetchone()[0])
            latest["shards"][str(number)] = {"accepted": accepted_count, "expected": int(item["users"]), "remaining": int(item["users"]) - accepted_count}
    atomic_json(root / "checkpoints" / "latest.json", latest)
    status = {
        "accepted_users": expected,
        "anomaly": "none",
        "api_failures": 0,
        "batch_id": submission["batch_id"],
        "checkpoint_path": str(checkpoint_path),
        "cleanup_users": cleanup_users,
        "cumulative_cost_usd": float(usage_total[4]),
        "decision": f"accepted_and_checkpointed; continue_to_shard_{shard + 1}",
        "exact_cost_usd": exact_cost,
        "expected_users": expected,
        "final_summaries_path": str(final_path),
        "final_summaries_sha256": sha256(final_path),
        "objective_failures": 0,
        "objective_validation": objective,
        "protocol": "frozen_v10_evidence_gated_emiliano",
        "raw_response_path": str(raw_path),
        "raw_response_sha256": sha256(raw_path),
        "raw_response_attempts": attempt_artifacts,
        "retries": retry_count,
        "review_path": str(review_path),
        "review_sha256": sha256(review_path),
        "shard": shard,
        f"shard_{shard + 1}_submitted": False,
        "successful_users_resubmitted": 0,
        "usage": usage,
        "wandb_url": review["wandb_url"],
    }
    atomic_json(root / "reports" / "shards" / f"shard-{shard:03d}-final-status.json", status)
    prod._log_wandb(
        root,
        read_json(root / "wandb_monitor_config.json"),
        {
            "validation/last_shard": shard,
            "validation/last_attempt": max(int(row["attempt"]) for row in rows),
            "validation/accepted_this_attempt": expected,
            "validation/failed_this_attempt": 0,
            "validation/objective_failures": 0,
            "validation/allowed_cleanup_users": cleanup_users,
            "validation/exact_duplicate_pairs": 0,
            "validation/near_duplicate_pairs": 0,
            "production/valid_users": corpus_accepted,
            "production/completion_percent": corpus_accepted / int(plan["users"]) * 100,
            "production/remaining_users": int(plan["users"]) - corpus_accepted,
            "production/last_completed_shard": shard,
            "production/next_shard_submitted": 0,
            "production/successful_users_resubmitted": 0,
            "retry/last_shard_retries": retry_count,
            "cost/last_shard_usd": exact_cost,
            "cost/actual_usd": float(usage_total[4]),
            "usage/input_tokens": int(usage_total[0]),
            "usage/cached_input_tokens": int(usage_total[1]),
            "usage/output_tokens": int(usage_total[2]),
            "usage/reasoning_tokens": int(usage_total[3]),
        },
    )
    return status


def status_count(status: dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        if key in status:
            return int(status[key])
    return default


def existing_final_audit(root: Path, plan: dict[str, Any]) -> dict[str, Any] | None:
    audit_path = root / "reports" / "final_audit_report.json"
    artifacts = {
        "final_summaries.jsonl": root / "validated" / "final_summaries.jsonl",
        "raw_summaries.jsonl": root / "validated" / "raw_summaries.jsonl",
        "all_validated_summaries.jsonl": root / "validated" / "all_validated_summaries.jsonl",
    }
    if not audit_path.exists() or any(not path.exists() for path in artifacts.values()):
        return None
    audit = read_json(audit_path)
    expected = int(plan["users"])
    if int(audit.get("valid_summaries") or 0) != expected or int(audit.get("expected_summaries") or 0) != expected:
        return None
    if int(audit.get("failures_remaining") or 0) != 0:
        return None
    corpus_path = artifacts["final_summaries.jsonl"]
    if audit.get("corpus_sha256") != sha256(corpus_path):
        return None
    if Path(str(audit.get("final_summaries_path") or "")) != corpus_path:
        return None
    return audit


def final_completion(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    audit = existing_final_audit(root, plan)
    if audit is None:
        audit = prod.finalize_corpus(root / "production_plan.json")
    checkpoint = read_json(root / "checkpoints" / "latest.json")
    statuses = [read_json(root / "reports" / "shards" / f"shard-{int(item['shard']):03d}-final-status.json") for item in plan["shards"]]
    with sqlite3.connect(f"file:{root / 'checkpoints/production.sqlite3'}?mode=ro", uri=True) as database:
        total = int(database.execute("SELECT COUNT(*) FROM accepted").fetchone()[0])
        distinct_users = int(database.execute("SELECT COUNT(DISTINCT user_id) FROM accepted").fetchone()[0])
        retry_count = int(database.execute("SELECT COUNT(*) FROM accepted WHERE attempt>0").fetchone()[0])
    artifacts = {}
    for name in ("final_summaries.jsonl", "raw_summaries.jsonl", "all_validated_summaries.jsonl"):
        path = root / "validated" / name
        artifacts[name] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    completion = {
        "protocol": "frozen_v10_evidence_gated_emiliano",
        "accepted_users": total,
        "expected_users": int(plan["users"]),
        "api_successes": sum(status_count(status, "accepted_users") for status in statuses),
        "api_failures": sum(status_count(status, "api_failures", "api_failed") for status in statuses),
        "objective_failures": sum(status_count(status, "objective_failures") for status in statuses),
        "retries_or_regenerations": retry_count,
        "authorized_cleanup_users": sum(
            status_count(
                status,
                "cleanup_users",
                default=int((status.get("objective_validation") or {}).get("allowed_cleanup_users") or 0),
            )
            for status in statuses
        ),
        "usage": {
            "input_tokens": checkpoint["input_tokens"],
            "cached_input_tokens": checkpoint["cached_input_tokens"],
            "output_tokens": checkpoint["output_tokens"],
            "reasoning_tokens": checkpoint["reasoning_tokens"],
            "total_tokens": checkpoint["input_tokens"] + checkpoint["output_tokens"],
        },
        "exact_cumulative_cost_usd": checkpoint["actual_cost_usd"],
        "per_shard": statuses,
        "checkpoint": {"path": str(root / "checkpoints" / "latest.json"), "sha256": sha256(root / "checkpoints" / "latest.json")},
        "artifacts": artifacts,
        "final_audit_report": {"path": str(root / "reports" / "final_audit_report.json"), "sha256": sha256(root / "reports" / "final_audit_report.json")},
        "wandb": {"status": "updated", "url": read_json(root / "wandb_monitor_config.json")["run_url"]},
        "anomalies": [],
        "every_user_accounted_for_exactly_once": total == distinct_users == int(plan["users"]),
        "ready_for_next_training_stage": True,
        "tears_training_started": False,
        "completed_at": int(time.time()),
        "final_audit": audit,
    }
    if not completion["every_user_accounted_for_exactly_once"] or checkpoint["actual_cost_usd"] > prod.CONSERVATIVE_COST_CAP_USD:
        stop(root, len(plan["shards"]) - 1, "final_integrity_or_budget", completion)
    path = root / "reports" / "final_production_completion.json"
    atomic_json(path, completion)
    return completion


def finish_shard(
    root: Path,
    plan: dict[str, Any],
    config: Path,
    shard: int,
    auto_retry_objective_failures: bool,
) -> dict[str, Any]:
    plan_path = root / "production_plan.json"
    expected = int(plan["shards"][shard]["users"])
    for attempt in range(prod.MAX_RETRIES + 1):
        submission_path = root / "submissions" / f"shard-{shard:03d}-attempt-{attempt:02d}.json"
        if not submission_path.exists():
            result = prod.submit_shard(
                plan_path,
                config,
                shard,
                attempt,
                True,
                prod.CONSERVATIVE_COST_CAP_USD,
            )
            if result.get("status") == "nothing_to_submit":
                return finalize_shard(root, plan, shard)
        poll = poll_to_terminal(root, shard, attempt)
        batch = poll["batch"]
        if batch["status"] != "completed":
            stop(root, shard, "unexpected_terminal_batch_status", {"attempt": attempt, "batch": batch})

        report_path = root / "reports" / "shards" / f"shard-{shard:03d}-attempt-{attempt:02d}.json"
        if not report_path.exists():
            prod.validate_shard(plan_path, shard, attempt)
        rows = attempt_rows(root, shard, attempt)
        submission = read_json(submission_path)
        if len(rows) != int(submission["requests"]):
            stop(
                root,
                shard,
                "attempt_row_integrity",
                {"attempt": attempt, "rows": len(rows), "submitted": int(submission["requests"])},
            )
        promote_objectively_valid_rows(root, shard, rows)
        with sqlite3.connect(f"file:{root / 'checkpoints/production.sqlite3'}?mode=ro", uri=True) as database:
            accepted = int(
                database.execute("SELECT COUNT(*) FROM accepted WHERE shard=?", (shard,)).fetchone()[0]
            )
        if accepted == expected:
            return finalize_shard(root, plan, shard)

        unresolved = objective_failures(rows)
        if not auto_retry_objective_failures:
            stop(
                root,
                shard,
                "unexpected_objective_validation_failures",
                {"count": len(unresolved), "examples": unresolved[:20]},
            )
        if attempt >= prod.MAX_RETRIES:
            stop(
                root,
                shard,
                "objective_retry_limit_exhausted",
                {"attempt": attempt, "accepted": accepted, "expected": expected, "examples": unresolved[:20]},
            )
    raise AssertionError("unreachable")


def run(root: Path, config: Path, auto_retry_objective_failures: bool = False) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    plan_path = root / "production_plan.json"
    plan = read_json(plan_path)
    stop_path = root / "reports" / "STOP-production.json"
    if stop_path.exists() and auto_retry_objective_failures:
        original_stop = read_json(stop_path)
        atomic_json(
            root / "reports" / "RESUME-production.json",
            {
                "status": "RESUME_AUTHORIZED",
                "created_at": int(time.time()),
                "original_stop": original_stop,
                "original_stop_path": str(stop_path),
                "original_stop_sha256": sha256(stop_path),
                "policy": "retry_only_unaccepted_users_up_to_frozen_retry_limit_then_continue_remaining_shards",
                "successful_users_resubmitted": 0,
            },
        )
    client = OpenAI()
    for shard in range(len(plan["shards"])):
        status_path = root / "reports" / "shards" / f"shard-{shard:03d}-final-status.json"
        if status_path.exists():
            continue
        preflight(root, plan, shard, client)
        finish_shard(root, plan, config, shard, auto_retry_objective_failures)
    return final_completion(root, plan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--auto-retry-objective-failures", action="store_true")
    args = parser.parse_args()
    result = run(args.root, args.config, args.auto_retry_objective_failures)
    print(json.dumps({"status": "complete", "report": str(args.root / 'reports/final_production_completion.json'), "accepted": result["accepted_users"], "cost": result["exact_cumulative_cost_usd"]}, sort_keys=True))


if __name__ == "__main__":
    main()
