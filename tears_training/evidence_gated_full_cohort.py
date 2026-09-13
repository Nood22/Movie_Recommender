"""Frozen V10 full-cohort generation for the ML-32M TEARS corpus.

This module is an execution harness, not a new summary protocol.  It refuses to
run unless the active V10 prompt/harness bytes match the archived, paid V10
calibration.  It adds only production mechanics: deterministic cohort
materialization, sharding, Batch API submission, allowed non-semantic cleanup,
hard contract validation, duplicate screening, retries, checkpoints, and
standard W&B progress logging.

It deliberately contains no TEARS training command and no semantic repair.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import statistics
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pandas as pd

from .artifacts import atomic_write_bytes, atomic_write_json, sha256_file, stable_hash
from .config import load_config
from . import final_summaries as history_core
from . import production_summary_protocol as lexical
from .evidence_gated_summary_harness import (
    EVIDENCE_GATED_PROMPT,
    EVIDENCE_SCHEMA_VERSION,
    PROMPT_SHA256,
    PROTOCOL_VERSION,
    build_response_request,
    build_separated_evidence,
    privacy_and_format_cleanup,
    prompt_delta,
)


MODEL = "gpt-5-mini-2025-08-07"
USERS = 200_948
SHARD_SIZE = 15_000
MAX_BATCH_BYTES = 200_000_000
MAX_BATCH_REQUESTS = 50_000
MAX_OUTPUT_TOKENS = 450
MAX_RETRIES = 2
PRICING_VERIFIED_AT = "2026-08-16"
BATCH_INPUT_USD_PER_MILLION = 0.125
BATCH_CACHED_INPUT_USD_PER_MILLION = 0.0125
BATCH_OUTPUT_USD_PER_MILLION = 1.00
POINT_PROJECTED_COST_USD = 76.699992831
CONSERVATIVE_COST_CAP_USD = 99.7099906803
V10_INPUT_TOKENS_PER_USER = 1465.806
V10_OUTPUT_TOKENS_PER_USER = 198.465
V10_ROOT = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "production_calibration_1000/v010_20260816_evidence_gated_emiliano"
)
MATRIX_DIR = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/datasets/"
    "catalog_selection/support_20/matrix"
)
DATASET_DIR = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/datasets/catalog_selection"
)
DEFAULT_ROOT = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "full_cohort/v010_20260816_evidence_gated_emiliano_frozen"
)
PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
BATCH_GUIDE = "https://developers.openai.com/api/docs/guides/batch"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _immutable_json(path: Path, value: Any) -> None:
    if path.exists():
        if _json(path) != value:
            raise RuntimeError(f"Refusing to overwrite different artifact: {path}")
        return
    atomic_write_json(path, value)


def _immutable_bytes(path: Path, value: bytes) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise RuntimeError(f"Refusing to overwrite different artifact: {path}")
        return
    atomic_write_bytes(path, value)


class AtomicJsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.temporary = path.with_name(f".{path.name}.tmp")
        self.handle: Any = None
        self.count = 0

    def __enter__(self) -> "AtomicJsonlWriter":
        if self.path.exists():
            raise RuntimeError(f"Artifact already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.temporary.open("w", encoding="utf-8")
        return self

    def write(self, value: Mapping[str, Any]) -> None:
        self.handle.write(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        self.count += 1

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
        if exc_type is None:
            os.replace(self.temporary, self.path)
        else:
            self.temporary.unlink(missing_ok=True)


def _frozen_v10_checks() -> dict[str, Any]:
    archived_harness = V10_ROOT / "code" / "evidence_gated_summary_harness.py"
    archived_protocol = _json(V10_ROOT / "protocol.json")
    current_harness = Path(inspect.getsourcefile(build_separated_evidence) or "")
    checks = {
        "v10_artifact_exists": V10_ROOT.is_dir(),
        "model_exact": MODEL == archived_protocol["generation"]["model"],
        "protocol_version_exact": PROTOCOL_VERSION == archived_protocol["version"],
        "prompt_sha256_exact": PROMPT_SHA256 == archived_protocol["system_prompt_sha256"],
        "prompt_bytes_exact": hashlib.sha256(EVIDENCE_GATED_PROMPT.encode()).hexdigest()
        == archived_protocol["system_prompt_sha256"],
        "harness_bytes_match_archived_v10": sha256_file(current_harness)
        == sha256_file(archived_harness),
        "no_v12_dependency": "v12" not in inspect.getsource(build_separated_evidence).lower(),
        "no_tmdb_semantic_metadata": not archived_protocol["tmdb_semantic_metadata"],
        "no_semantic_post_generation_repair": not archived_protocol["post_generation"][
            "semantic_repair"
        ],
    }
    return {"checks": checks, "pass": all(checks.values())}


def _record_from_history(
    user_id: int,
    split: str,
    activity_band: str,
    history: Sequence[tuple[int, int, str, float, str]],
) -> dict[str, Any]:
    rendered = history_core.render_emiliano_history(history)
    return {
        "user_id": int(user_id),
        "split": split,
        "activity_band": activity_band,
        "history_hash": hashlib.sha256(rendered.encode()).hexdigest(),
        "history_items": len(history),
        "movie_ids": [row[1] for row in history],
        "titles": [row[2] for row in history],
        "ratings": [row[3] for row in history],
        "genres": [row[4] for row in history],
    }


def _history_for_user(
    rows: Sequence[tuple[int, int, str, float, str]], split: str, observed_fraction: float
) -> list[tuple[int, int, str, float, str]]:
    chronological = sorted(rows, key=lambda item: (item[0], item[1]))
    if split in {"validation", "test"}:
        boundary = max(
            1,
            min(len(chronological) - 1, int(math.floor(len(chronological) * observed_fraction))),
        )
        chronological = chronological[:boundary]
    result = list(reversed(chronological[-history_core.MAX_HISTORY_ITEMS :]))
    if not result:
        raise RuntimeError("Frozen cohort user has no observed catalog history")
    return result


def _iter_full_records(config_path: Path) -> Iterator[dict[str, Any]]:
    config = load_config(config_path)
    frozen = history_core.verify_frozen_matrix(MATRIX_DIR, DATASET_DIR)
    if frozen["users"] != USERS:
        raise RuntimeError("Frozen matrix user count changed")
    users = pd.read_csv(DATASET_DIR / "user_splits.csv")
    matrix_users = pd.read_csv(MATRIX_DIR / "users.csv", usecols=["userId"])
    if set(users.userId.astype(int)) != set(matrix_users.userId.astype(int)):
        raise RuntimeError("Frozen split and matrix user IDs differ")
    metadata = {
        int(row.userId): (str(row.split), str(row.activity_band))
        for row in users.itertuples(index=False)
    }
    catalog = pd.read_csv(MATRIX_DIR / "catalog.csv", usecols=["movieId", "title", "genres"])
    movies = {
        int(row.movieId): (str(row.title), str(row.genres))
        for row in catalog.itertuples(index=False)
    }
    seen: set[int] = set()
    current_user: int | None = None
    current: list[tuple[int, int, str, float, str]] = []

    def finish(user_id: int, values: list[tuple[int, int, str, float, str]]) -> dict[str, Any]:
        if user_id not in metadata:
            raise RuntimeError(f"Unexpected ratings user {user_id}")
        split, band = metadata[user_id]
        history = _history_for_user(values, split, config.data.observed_fraction)
        return _record_from_history(user_id, split, band, history)

    ratings_path = config.raw_data / "ratings.csv"
    with ratings_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for source in reader:
            user_id = int(source["userId"])
            if current_user is None:
                current_user = user_id
            elif user_id != current_user:
                if user_id < current_user:
                    raise RuntimeError("ratings.csv is not grouped by ascending userId")
                if current_user in metadata:
                    yield finish(current_user, current)
                    seen.add(current_user)
                current_user = user_id
                current = []
            movie_id = int(source["movieId"])
            if user_id in metadata and movie_id in movies:
                title, genres = movies[movie_id]
                current.append(
                    (int(source["timestamp"]), movie_id, title, float(source["rating"]), genres)
                )
        if current_user is not None and current_user in metadata:
            yield finish(current_user, current)
            seen.add(current_user)
    missing = set(metadata) - seen
    if missing:
        raise RuntimeError(f"Frozen cohort histories missing {len(missing)} users")


def _shard_sizes() -> list[int]:
    full, remainder = divmod(USERS, SHARD_SIZE)
    return [SHARD_SIZE] * full + ([remainder] if remainder else [])


def _request_paths(root: Path, shard: int, attempt: int = 0) -> tuple[Path, Path, Path]:
    stem = f"shard-{shard:03d}-attempt-{attempt:02d}"
    return (
        root / "requests" / f"{stem}.jsonl",
        root / "manifests" / f"{stem}.jsonl",
        root / "submissions" / f"{stem}.json",
    )


def plan_full_cohort(root: Path, config_path: Path) -> dict[str, Any]:
    root = root.resolve()
    plan_path = root / "production_plan.json"
    if plan_path.exists():
        return _verified_plan(plan_path)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Refusing nonempty unplanned production root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    frozen = _frozen_v10_checks()
    if not frozen["pass"]:
        raise RuntimeError(f"Frozen V10 verification failed: {frozen}")
    shard_sizes = _shard_sizes()
    shard_writers: dict[int, tuple[AtomicJsonlWriter, AtomicJsonlWriter, AtomicJsonlWriter]] = {}
    shard_meta: list[dict[str, Any]] = []
    user_ids: list[int] = []
    statuses: dict[str, int] = {"NONE": 0, "WEAK": 0, "STRONG": 0}
    try:
        for shard in range(len(shard_sizes)):
            request_path, manifest_path, _ = _request_paths(root, shard)
            writers = (
                AtomicJsonlWriter(request_path),
                AtomicJsonlWriter(manifest_path),
                AtomicJsonlWriter(root / "evidence" / f"shard-{shard:03d}.jsonl"),
            )
            for writer in writers:
                writer.__enter__()
            shard_writers[shard] = writers
        for index, record in enumerate(_iter_full_records(config_path)):
            shard = index // SHARD_SIZE
            if shard >= len(shard_sizes):
                raise RuntimeError("Cohort exceeds frozen size")
            evidence = build_separated_evidence(record)
            request = build_response_request(record, evidence, model=MODEL, max_output_tokens=MAX_OUTPUT_TOKENS)
            request_writer, manifest_writer, evidence_writer = shard_writers[shard]
            request_writer.write(request)
            manifest_writer.write(
                {
                    **record,
                    "base_custom_id": request["custom_id"],
                    "negative_evidence_status": evidence["negative_evidence_status"],
                    "frozen_evidence_hash": evidence["frozen_evidence_hash"],
                }
            )
            evidence_writer.write(evidence)
            user_ids.append(int(record["user_id"]))
            statuses[evidence["negative_evidence_status"]] += 1
            if (index + 1) % 10_000 == 0:
                print(json.dumps({"planned_users": index + 1, "total": USERS}), flush=True)
        if len(user_ids) != USERS or len(set(user_ids)) != USERS:
            raise RuntimeError("Plan does not contain exactly 200,948 unique users")
        for shard, writers in shard_writers.items():
            for writer in writers:
                writer.__exit__(None, None, None)
            expected = shard_sizes[shard]
            if any(writer.count != expected for writer in writers):
                raise RuntimeError(f"Shard {shard} row count mismatch")
        shard_writers.clear()
    except BaseException as error:
        for writers in shard_writers.values():
            for writer in writers:
                writer.__exit__(type(error), error, error.__traceback__)
        raise

    for shard, users_in_shard in enumerate(shard_sizes):
        request_path, manifest_path, _ = _request_paths(root, shard)
        evidence_path = root / "evidence" / f"shard-{shard:03d}.jsonl"
        size = request_path.stat().st_size
        if size >= MAX_BATCH_BYTES or users_in_shard > MAX_BATCH_REQUESTS:
            raise RuntimeError(f"Shard {shard} violates Batch limits")
        shard_meta.append(
            {
                "shard": shard,
                "users": users_in_shard,
                "request_path": str(request_path),
                "request_bytes": size,
                "request_sha256": sha256_file(request_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "evidence_path": str(evidence_path),
                "evidence_sha256": sha256_file(evidence_path),
                "projected_input_tokens": users_in_shard * V10_INPUT_TOKENS_PER_USER,
                "projected_output_tokens": users_in_shard * V10_OUTPUT_TOKENS_PER_USER,
                "projected_cost_usd": users_in_shard / USERS * POINT_PROJECTED_COST_USD,
            }
        )
    cohort = {
        "users": USERS,
        "user_ids": user_ids,
        "user_ids_sha256": stable_hash(user_ids),
        "matrix_users_sha256": sha256_file(MATRIX_DIR / "users.csv"),
        "split_sha256": sha256_file(DATASET_DIR / "user_splits.csv"),
    }
    atomic_write_json(root / "cohort" / "cohort.json", cohort)
    archived_protocol = _json(V10_ROOT / "protocol.json")
    protocol = {
        **archived_protocol,
        "production_status": "frozen_final_protocol",
        "production_harness_only": True,
        "pricing": {
            "verified_at": PRICING_VERIFIED_AT,
            "source": PRICING_SOURCE,
            "batch_guide": BATCH_GUIDE,
            "usd_per_million": {
                "input": BATCH_INPUT_USD_PER_MILLION,
                "cached_input": BATCH_CACHED_INPUT_USD_PER_MILLION,
                "output": BATCH_OUTPUT_USD_PER_MILLION,
            },
        },
    }
    atomic_write_json(root / "protocol.json", protocol)
    atomic_write_json(root / "prompt" / "prompt_delta.json", prompt_delta())
    _immutable_bytes(root / "prompt" / "evidence_gated_emiliano_prompt.txt", EVIDENCE_GATED_PROMPT.encode())
    code_files = [
        Path(__file__),
        Path(inspect.getsourcefile(build_separated_evidence) or ""),
    ]
    (root / "code").mkdir(parents=True, exist_ok=True)
    for source in code_files:
        shutil.copy2(source, root / "code" / source.name)
    code_manifest = {
        source.name: {"source": str(source.resolve()), "sha256": sha256_file(source)}
        for source in code_files
    }
    atomic_write_json(root / "code" / "manifest.json", code_manifest)
    plan = {
        "scope": "frozen_v10_full_cohort_200948",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "users": USERS,
        "model": MODEL,
        "protocol_version": PROTOCOL_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "frozen_v10": frozen,
        "cohort_path": str(root / "cohort" / "cohort.json"),
        "cohort_user_ids_sha256": cohort["user_ids_sha256"],
        "negative_status_distribution": statuses,
        "shard_size": SHARD_SIZE,
        "shards": shard_meta,
        "waves": [list(range(start, min(start + 2, len(shard_meta)))) for start in range(0, len(shard_meta), 2)],
        "maximum_concurrent_batches": 2,
        "checkpoint_after_every_shard": True,
        "max_retries": MAX_RETRIES,
        "successful_users_resubmitted": False,
        "point_projected_cost_usd": POINT_PROJECTED_COST_USD,
        "conservative_cost_cap_usd": CONSERVATIVE_COST_CAP_USD,
        "projected_input_tokens": round(USERS * V10_INPUT_TOKENS_PER_USER),
        "projected_output_tokens": round(USERS * V10_OUTPUT_TOKENS_PER_USER),
        "tears_training_started": False,
    }
    plan["fingerprint"] = stable_hash(plan)
    atomic_write_json(plan_path, plan)
    _initialize_state(root, plan)
    return plan


def _verified_plan(path: Path, verify_shards: Sequence[int] = ()) -> dict[str, Any]:
    plan = _json(path)
    fingerprint = plan.pop("fingerprint")
    actual = stable_hash(plan)
    plan["fingerprint"] = fingerprint
    if actual != fingerprint:
        raise RuntimeError("Production plan fingerprint mismatch")
    if plan["scope"] != "frozen_v10_full_cohort_200948" or plan["users"] != USERS:
        raise RuntimeError("Not the frozen V10 full-cohort plan")
    frozen = _frozen_v10_checks()
    if not frozen["pass"]:
        raise RuntimeError("Active V10 harness no longer matches archived V10")
    requested = set(verify_shards)
    for shard in plan["shards"]:
        if int(shard["shard"]) not in requested:
            continue
        path = Path(shard["request_path"])
        if sha256_file(path) != shard["request_sha256"]:
            raise RuntimeError(f"Request shard changed: {path}")
    return plan


def _db_path(root: Path) -> Path:
    return root / "checkpoints" / "production.sqlite3"


def _connect(root: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(_db_path(root))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _initialize_state(root: Path, plan: dict[str, Any]) -> None:
    path = _db_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS accepted (
              user_id INTEGER PRIMARY KEY, shard INTEGER NOT NULL, attempt INTEGER NOT NULL,
              custom_id TEXT NOT NULL UNIQUE, raw_summary TEXT NOT NULL, final_summary TEXT NOT NULL,
              normalized_hash TEXT NOT NULL UNIQUE, simhash TEXT NOT NULL,
              input_tokens INTEGER NOT NULL, cached_input_tokens INTEGER NOT NULL,
              output_tokens INTEGER NOT NULL, reasoning_tokens INTEGER NOT NULL,
              cost_usd REAL NOT NULL, cleanup_json TEXT NOT NULL, accepted_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS near_buckets (
              bucket TEXT NOT NULL, user_id INTEGER NOT NULL,
              PRIMARY KEY(bucket, user_id),
              FOREIGN KEY(user_id) REFERENCES accepted(user_id)
            );
            CREATE INDEX IF NOT EXISTS near_bucket_index ON near_buckets(bucket);
            CREATE TABLE IF NOT EXISTS attempts (
              shard INTEGER NOT NULL, attempt INTEGER NOT NULL, user_id INTEGER NOT NULL,
              status TEXT NOT NULL, reasons_json TEXT NOT NULL, custom_id TEXT NOT NULL,
              PRIMARY KEY(shard, attempt, user_id)
            );
            """
        )
        db.commit()
    _checkpoint_snapshot(root, plan)


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _words(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text, re.UNICODE))


def _simhash(text: str) -> int:
    tokens = _normal(text).split()
    vector = [0] * 64
    for index in range(max(1, len(tokens) - 2)):
        value = " ".join(tokens[index : index + 3])
        hashed = int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if (hashed >> bit) & 1 else -1
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


def _simhash_buckets(value: int) -> list[str]:
    return [f"{index}:{(value >> (index * 16)) & 0xffff:04x}" for index in range(4)]


def _near_duplicate(db: sqlite3.Connection, summary: str) -> int | None:
    normalized = _normal(summary)
    tokens = set(normalized.split())
    value = _simhash(summary)
    placeholders = ",".join("?" for _ in range(4))
    candidates = db.execute(
        f"SELECT DISTINCT a.user_id,a.final_summary FROM near_buckets b JOIN accepted a "
        f"ON a.user_id=b.user_id WHERE b.bucket IN ({placeholders})",
        _simhash_buckets(value),
    ).fetchall()
    for row in candidates:
        other = _normal(row["final_summary"])
        if normalized == other:
            return int(row["user_id"])
        other_tokens = set(other.split())
        union = tokens | other_tokens
        jaccard = len(tokens & other_tokens) / len(union) if union else 0.0
        if jaccard >= 0.72 and SequenceMatcher(None, normalized, other, autojunk=False).ratio() >= 0.90:
            return int(row["user_id"])
    return None


def _semantic_failures(summary: str, evidence: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    status = str(evidence["negative_evidence_status"])
    sentences = lexical.SENTENCE_SPLIT.split(summary)
    negative = [sentence for sentence in sentences if lexical._is_user_negative_claim(sentence)]
    abstentions = [sentence for sentence in sentences if lexical._is_abstention(sentence)]
    supported = set(evidence["genre_classifications"]["supported_negative"])
    mixed = set(evidence["genre_classifications"]["mixed_conflicting"])
    negative_genres = {genre for sentence in negative for genre in lexical._mentioned_genres(sentence)}
    categorical = [
        sentence for sentence in negative
        if re.search(r"\b(?:does not enjoy|dislikes?|avoids?|not a fan of|averse to)\b", sentence, re.I)
    ]
    cautious = [
        sentence for sentence in negative
        if re.search(r"\b(?:may|might|seems?|suggests?|tends?|less|limited|cautious|tentative|selective)\b", sentence, re.I)
    ]
    failures: list[str] = []
    if status == "NONE" and negative:
        failures.append("fabricated_dislike_for_none")
    if status == "WEAK" and categorical:
        failures.append("overstated_negative_for_weak")
    if status == "WEAK" and negative and not cautious:
        failures.append("uncautious_negative_for_weak")
    if status == "STRONG" and abstentions and not negative:
        failures.append("inappropriate_negative_abstention")
    unsupported = sorted(negative_genres - supported) if categorical else []
    if unsupported:
        failures.append("unsupported_categorical_negative_genre")
    if negative_genres & mixed:
        failures.append("categorical_negative_on_mixed_genre")
    positive_genres: set[str] = set()
    for sentence in sentences:
        if lexical._is_user_negative_claim(sentence) or lexical._is_abstention(sentence):
            continue
        if re.search(r"\b(?:likes?|enjoys?|prefers?|favors?|favours?|drawn to|fond of|enthusiastic about)\b", sentence, re.I):
            positive_genres |= lexical._mentioned_genres(sentence)
    contradictions = sorted(positive_genres & negative_genres)
    if contradictions:
        failures.append("substantive_positive_negative_contradiction")
    return sorted(set(failures)), {
        "negative_sentences": negative,
        "abstentions": abstentions,
        "negative_genres": sorted(negative_genres),
        "positive_genres": sorted(positive_genres),
        "unsupported_negative_genres": unsupported,
        "contradictory_genres": contradictions,
        "supported_negative_genres_omitted_not_a_failure": sorted(supported - negative_genres),
    }


def _wandb_metadata(root: Path, config_path: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    config = load_config(config_path)
    run_id = stable_hash({"purpose": "v10-full-cohort-summary-batch", "plan": plan["fingerprint"]})[:16]
    return {
        "entity": config.tracking.entity,
        "project": config.tracking.project,
        "mode": config.tracking.mode,
        "run_id": run_id,
        "run_name": f"v10-full-cohort-summaries-{run_id[:8]}",
        "run_url": f"https://wandb.ai/{config.tracking.entity}/{config.tracking.project}/runs/{run_id}",
        "job_type": "summary-batch-monitor",
        "weave_enabled": False,
        "root": str(root),
        "plan_fingerprint": plan["fingerprint"],
    }


def _freeze_execution_code_before_first_submission(root: Path) -> None:
    """Freeze production mechanics once; never mutate them after paid work starts."""

    source = Path(__file__)
    destination = root / "code" / source.name
    submissions = list((root / "submissions").glob("shard-*-attempt-*.json"))
    if submissions:
        if not destination.exists() or sha256_file(destination) != sha256_file(source):
            raise RuntimeError("Production execution code changed after paid submission")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    archived_harness = root / "code" / "evidence_gated_summary_harness.py"
    atomic_write_json(
        root / "code" / "manifest.json",
        {
            source.name: {"source": str(source.resolve()), "sha256": sha256_file(source)},
            archived_harness.name: {
                "source": str(Path(inspect.getsourcefile(build_separated_evidence) or "").resolve()),
                "sha256": sha256_file(archived_harness),
            },
        },
    )


def _log_wandb(root: Path, metadata: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    import wandb

    directory = root / "wandb"
    directory.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=metadata["entity"], project=metadata["project"], name=metadata["run_name"],
        id=metadata["run_id"], resume="allow", job_type=metadata["job_type"],
        config=dict(metadata), mode=metadata["mode"], dir=str(directory),
    )
    run.log(dict(metrics))
    for key, value in metrics.items():
        run.summary[key] = value
    run.finish()


def _accepted_ids(root: Path, shard: int | None = None) -> set[int]:
    with _connect(root) as db:
        if shard is None:
            rows = db.execute("SELECT user_id FROM accepted")
        else:
            rows = db.execute("SELECT user_id FROM accepted WHERE shard=?", (shard,))
        return {int(row[0]) for row in rows}


def _prepare_attempt(root: Path, plan: Mapping[str, Any], shard: int, attempt: int) -> tuple[Path, Path, int]:
    base_request = Path(plan["shards"][shard]["request_path"])
    base_manifest = Path(plan["shards"][shard]["manifest_path"])
    request_path, manifest_path, _ = _request_paths(root, shard, attempt)
    if attempt == 0:
        return base_request, base_manifest, int(plan["shards"][shard]["users"])
    if request_path.exists() and manifest_path.exists():
        return request_path, manifest_path, sum(1 for _ in request_path.open(encoding="utf-8"))
    accepted = _accepted_ids(root, shard)
    records = list(_jsonl(base_manifest))
    unresolved = {int(row["user_id"]) for row in records} - accepted
    if not unresolved:
        return request_path, manifest_path, 0
    requests_by_id = {row["custom_id"]: row for row in _jsonl(base_request)}
    with AtomicJsonlWriter(request_path) as request_writer, AtomicJsonlWriter(manifest_path) as manifest_writer:
        for record in records:
            user_id = int(record["user_id"])
            if user_id not in unresolved:
                continue
            request = requests_by_id[record["base_custom_id"]]
            request = json.loads(json.dumps(request))
            request["custom_id"] = f"{request['custom_id']}-r{attempt}"
            request_writer.write(request)
            manifest_writer.write({**record, "attempt_custom_id": request["custom_id"]})
    return request_path, manifest_path, len(unresolved)


def submit_shard(
    plan_path: Path, config_path: Path, shard: int, attempt: int, confirmed: bool, max_total_cost_usd: float
) -> dict[str, Any]:
    if not confirmed:
        raise RuntimeError("Paid Batch submission requires --confirm-spend")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    plan = _verified_plan(plan_path, (shard,))
    root = plan_path.parent
    if not 0 <= shard < len(plan["shards"]) or not 0 <= attempt <= MAX_RETRIES:
        raise RuntimeError("Invalid shard or retry attempt")
    if max_total_cost_usd > CONSERVATIVE_COST_CAP_USD + 1e-9:
        raise RuntimeError("Authorization cannot exceed the frozen conservative cost cap")
    _freeze_execution_code_before_first_submission(root)
    request_path, manifest_path, submission_path = _request_paths(root, shard, attempt)
    if submission_path.exists():
        return _json(submission_path)
    request_path, manifest_path, count = _prepare_attempt(root, plan, shard, attempt)
    if count == 0:
        return {"status": "nothing_to_submit", "shard": shard, "attempt": attempt, "requests": 0}
    accepted = _accepted_ids(root)
    attempt_users = {int(row["user_id"]) for row in _jsonl(manifest_path)}
    overlap = accepted & attempt_users
    if overlap:
        raise RuntimeError(f"Refusing to resubmit {len(overlap)} successful users")
    if count > MAX_BATCH_REQUESTS or request_path.stat().st_size >= MAX_BATCH_BYTES:
        raise RuntimeError("Attempt violates Batch request/file limits")
    prior_submissions = list((root / "submissions").glob("shard-*-attempt-*.json"))
    projected_already = sum(float(_json(path).get("projected_cost_usd", 0)) for path in prior_submissions)
    projected = count / USERS * POINT_PROJECTED_COST_USD
    if projected_already + projected > max_total_cost_usd:
        raise RuntimeError("Cumulative projected spend exceeds authorized cap")
    metadata_path = root / "wandb_monitor_config.json"
    if metadata_path.exists():
        metadata = _json(metadata_path)
    else:
        metadata = _wandb_metadata(root, config_path, plan)
        atomic_write_json(metadata_path, metadata)
        _log_wandb(root, metadata, {
            "production/total_users": USERS, "production/valid_users": 0,
            "production/completion_percent": 0.0, "cost/projected_usd": POINT_PROJECTED_COST_USD,
            "cost/conservative_cap_usd": CONSERVATIVE_COST_CAP_USD,
        })
    from openai import OpenAI

    client = OpenAI()
    with request_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"protocol": PROTOCOL_VERSION, "shard": str(shard), "attempt": str(attempt)},
    )
    result = {
        "scope": plan["scope"], "plan_fingerprint": plan["fingerprint"],
        "shard": shard, "attempt": attempt, "requests": count,
        "request_path": str(request_path), "request_sha256": sha256_file(request_path),
        "manifest_path": str(manifest_path), "input_file_id": uploaded.id,
        "batch_id": batch.id, "status": batch.status, "submitted_at": int(time.time()),
        "projected_cost_usd": projected, "maximum_authorized_total_cost_usd": max_total_cost_usd,
        "wandb_run_url": metadata["run_url"], "successful_users_resubmitted": 0,
    }
    atomic_write_json(submission_path, result)
    _log_wandb(root, metadata, {
        "batch/last_submitted_shard": shard, "batch/last_submitted_attempt": attempt,
        "batch/last_submitted_requests": count, "production/submitted_batches": len(prior_submissions) + 1,
    })
    return result


def poll_shard(submission_path: Path) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    submission = _json(submission_path)
    root = submission_path.parents[1]
    from openai import OpenAI

    client = OpenAI()
    batch = client.batches.retrieve(submission["batch_id"])
    status = batch.model_dump(mode="json")
    for kind, file_id in (("responses", batch.output_file_id), ("errors", batch.error_file_id)):
        if not file_id:
            continue
        payload = client.files.content(file_id).content
        destination = root / kind / f"shard-{submission['shard']:03d}-attempt-{submission['attempt']:02d}-{batch.id}.jsonl"
        _immutable_bytes(destination, payload)
    result = {"submission": str(submission_path), "batch": status, "polled_at": int(time.time())}
    atomic_write_json(root / "polls" / f"shard-{submission['shard']:03d}-attempt-{submission['attempt']:02d}.json", result)
    metadata = _json(root / "wandb_monitor_config.json")
    counts = status.get("request_counts") or {}
    _log_wandb(root, metadata, {
        "batch/last_polled_shard": submission["shard"],
        "batch/last_polled_attempt": submission["attempt"],
        "batch/last_total": int(counts.get("total", submission["requests"])),
        "batch/last_completed": int(counts.get("completed", 0)),
        "batch/last_failed": int(counts.get("failed", 0)),
        "batch/last_terminal": int(status.get("status") in TERMINAL_BATCH_STATUSES),
    })
    return result


def _response_file(root: Path, shard: int, attempt: int, batch_id: str) -> Path:
    return root / "responses" / f"shard-{shard:03d}-attempt-{attempt:02d}-{batch_id}.jsonl"


def _extract_summary(body: Mapping[str, Any]) -> str:
    raw = history_core.extract_output_text(dict(body))
    parsed = json.loads(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("summary"), str):
        raise ValueError("Structured output lacks summary string")
    return parsed["summary"].strip()


def validate_shard(plan_path: Path, shard: int, attempt: int) -> dict[str, Any]:
    plan = _verified_plan(plan_path, (shard,))
    root = plan_path.parent
    _, manifest_path, submission_path = _request_paths(root, shard, attempt)
    if not submission_path.exists():
        raise RuntimeError("Shard attempt was not submitted")
    submission = _json(submission_path)
    poll_path = root / "polls" / f"shard-{shard:03d}-attempt-{attempt:02d}.json"
    poll = _json(poll_path)
    if poll["batch"]["status"] != "completed":
        raise RuntimeError(f"Batch is not completed: {poll['batch']['status']}")
    response_path = _response_file(root, shard, attempt, submission["batch_id"])
    if not response_path.exists():
        raise RuntimeError("Completed batch response file is absent")
    manifests = {str(row.get("attempt_custom_id") or row["base_custom_id"]): row for row in _jsonl(manifest_path)}
    evidence = {int(row["user_id"]): row for row in _jsonl(Path(plan["shards"][shard]["evidence_path"]))}
    responses = {str(row.get("custom_id")): row for row in _jsonl(response_path)}
    accepted_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    with _connect(root) as db:
        for custom_id, record in manifests.items():
            user_id = int(record["user_id"])
            if db.execute("SELECT 1 FROM accepted WHERE user_id=?", (user_id,)).fetchone():
                continue
            response_row = responses.get(custom_id)
            failures: list[str] = []
            raw_summary = ""
            body: Mapping[str, Any] = {}
            if response_row is None:
                failures.append("missing_output")
            else:
                response = response_row.get("response") or {}
                body = response.get("body") or {}
                if response.get("status_code") != 200:
                    failures.append(f"http_status_{response.get('status_code')}")
                else:
                    try:
                        raw_summary = _extract_summary(body)
                    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                        failures.append("malformed_output")
            cleanup = privacy_and_format_cleanup(raw_summary, record["titles"]) if raw_summary else None
            final_summary = cleanup.final_summary if cleanup else ""
            semantic_details: dict[str, Any] = {}
            if cleanup:
                if cleanup.title_matches_after:
                    failures.append("title_leak_after_sanitization")
                if cleanup.year_matches_after:
                    failures.append("year_leak_after_sanitization")
                if cleanup.numeric_rating_matches_after:
                    failures.append("numeric_rating_leak_after_sanitization")
                failures.extend(cleanup.formatting_issues)
                if _words(final_summary) < 20:
                    failures.append("malformed_too_short")
                if re.search(r"\{[^{}]+\}|\b(?:TBD|TODO|N/?A)\b", final_summary, re.I):
                    failures.append("literal_placeholder")
                semantic_errors, semantic_details = _semantic_failures(final_summary, evidence[user_id])
                failures.extend(semantic_errors)
            usage = body.get("usage") or {}
            input_tokens = int(usage.get("input_tokens", 0))
            cached_tokens = int((usage.get("input_tokens_details") or {}).get("cached_tokens", 0))
            output_tokens = int(usage.get("output_tokens", 0))
            reasoning_tokens = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0))
            cost = (
                (input_tokens - cached_tokens) * BATCH_INPUT_USD_PER_MILLION
                + cached_tokens * BATCH_CACHED_INPUT_USD_PER_MILLION
                + output_tokens * BATCH_OUTPUT_USD_PER_MILLION
            ) / 1_000_000
            duplicate_user = None
            normalized_hash = hashlib.sha256(_normal(final_summary).encode()).hexdigest() if final_summary else ""
            if not failures:
                duplicate_user = _near_duplicate(db, final_summary)
                if duplicate_user is not None:
                    failures.append("unresolved_duplicate")
            result_row = {
                "user_id": user_id, "shard": shard, "attempt": attempt, "custom_id": custom_id,
                "raw_summary": raw_summary, "final_summary": final_summary,
                "raw_word_count": _words(raw_summary), "final_word_count": _words(final_summary),
                "cleanup_operations": list(cleanup.operations) if cleanup else [],
                "semantic_validation": semantic_details, "duplicate_of_user_id": duplicate_user,
                "failures": sorted(set(failures)), "input_tokens": input_tokens,
                "cached_input_tokens": cached_tokens, "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens, "cost_usd": cost,
            }
            db.execute(
                "INSERT OR REPLACE INTO attempts VALUES(?,?,?,?,?,?)",
                (shard, attempt, user_id, "failed" if failures else "accepted", json.dumps(sorted(set(failures))), custom_id),
            )
            if failures:
                failed_rows.append(result_row)
                continue
            simhash = _simhash(final_summary)
            db.execute(
                "INSERT INTO accepted VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (user_id, shard, attempt, custom_id, raw_summary, final_summary, normalized_hash,
                 f"{simhash:016x}", input_tokens, cached_tokens, output_tokens, reasoning_tokens,
                 cost, json.dumps(list(cleanup.operations) if cleanup else []), int(time.time())),
            )
            for bucket in _simhash_buckets(simhash):
                db.execute("INSERT INTO near_buckets VALUES(?,?)", (bucket, user_id))
            accepted_rows.append(result_row)
        db.commit()
    accepted_path = root / "validated" / "attempts" / f"shard-{shard:03d}-attempt-{attempt:02d}-accepted.jsonl"
    failed_path = root / "validated" / "attempts" / f"shard-{shard:03d}-attempt-{attempt:02d}-failed.jsonl"
    with AtomicJsonlWriter(accepted_path) as writer:
        for row in accepted_rows:
            writer.write(row)
    with AtomicJsonlWriter(failed_path) as writer:
        for row in failed_rows:
            writer.write(row)
    snapshot = _checkpoint_snapshot(root, plan, shard=shard)
    report = {
        "shard": shard, "attempt": attempt, "submitted": len(manifests),
        "accepted_this_attempt": len(accepted_rows), "failed_this_attempt": len(failed_rows),
        "failure_counts": _counts(reason for row in failed_rows for reason in row["failures"]),
        "shard_valid_total": snapshot["shards"][str(shard)]["accepted"],
        "shard_remaining": snapshot["shards"][str(shard)]["remaining"],
        "corpus_valid_total": snapshot["accepted_users"],
    }
    atomic_write_json(root / "reports" / "shards" / f"shard-{shard:03d}-attempt-{attempt:02d}.json", report)
    metadata = _json(root / "wandb_monitor_config.json")
    _log_wandb(root, metadata, {
        "validation/last_shard": shard, "validation/last_attempt": attempt,
        "validation/accepted_this_attempt": len(accepted_rows),
        "validation/failed_this_attempt": len(failed_rows),
        "production/valid_users": snapshot["accepted_users"],
        "production/completion_percent": snapshot["accepted_users"] / USERS * 100,
        "cost/actual_usd": snapshot["actual_cost_usd"],
    })
    return report


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _checkpoint_snapshot(root: Path, plan: Mapping[str, Any], shard: int | None = None) -> dict[str, Any]:
    with _connect(root) as db:
        total = int(db.execute("SELECT COUNT(*) FROM accepted").fetchone()[0])
        usage = db.execute(
            "SELECT COALESCE(SUM(input_tokens),0),COALESCE(SUM(cached_input_tokens),0),"
            "COALESCE(SUM(output_tokens),0),COALESCE(SUM(reasoning_tokens),0),COALESCE(SUM(cost_usd),0) FROM accepted"
        ).fetchone()
        shards: dict[str, Any] = {}
        for item in plan["shards"]:
            number = int(item["shard"])
            accepted = int(db.execute("SELECT COUNT(*) FROM accepted WHERE shard=?", (number,)).fetchone()[0])
            shards[str(number)] = {"expected": int(item["users"]), "accepted": accepted, "remaining": int(item["users"]) - accepted}
    value = {
        "plan_fingerprint": plan["fingerprint"], "accepted_users": total, "remaining_users": USERS - total,
        "input_tokens": int(usage[0]), "cached_input_tokens": int(usage[1]),
        "output_tokens": int(usage[2]), "reasoning_tokens": int(usage[3]),
        "actual_cost_usd": float(usage[4]), "shards": shards, "updated_at": int(time.time()),
        "successful_users_resubmitted": 0,
    }
    atomic_write_json(root / "checkpoints" / "latest.json", value)
    if shard is not None:
        atomic_write_json(root / "checkpoints" / f"shard-{shard:03d}.json", {"shard": shard, **shards[str(shard)], "corpus_accepted": total, "updated_at": value["updated_at"]})
    return value


def production_status(plan_path: Path) -> dict[str, Any]:
    plan = _verified_plan(plan_path)
    root = plan_path.parent
    snapshot = _checkpoint_snapshot(root, plan)
    submissions = [_json(path) for path in sorted((root / "submissions").glob("shard-*-attempt-*.json"))]
    polls = [_json(path) for path in sorted((root / "polls").glob("shard-*-attempt-*.json"))]
    return {
        **snapshot, "submissions": len(submissions),
        "batch_status_counts": _counts(str(row["batch"].get("status", "unknown")) for row in polls),
        "wandb_run_url": _json(root / "wandb_monitor_config.json")["run_url"] if (root / "wandb_monitor_config.json").exists() else None,
    }


def _length_stats(values: Sequence[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    def percentile(p: float) -> float:
        rank = (len(ordered) - 1) * p
        low = int(rank); high = min(low + 1, len(ordered) - 1); part = rank - low
        return ordered[low] + part * (ordered[high] - ordered[low])
    return {"mean": statistics.mean(values), "median": statistics.median(values), "minimum": min(values), "maximum": max(values), "p95": percentile(0.95)}


def finalize_corpus(plan_path: Path) -> dict[str, Any]:
    plan = _verified_plan(plan_path)
    root = plan_path.parent
    snapshot = _checkpoint_snapshot(root, plan)
    if snapshot["accepted_users"] != USERS:
        raise RuntimeError(f"Cannot finalize: {snapshot['remaining_users']} users remain invalid")
    with _connect(root) as db:
        rows = db.execute("SELECT * FROM accepted ORDER BY user_id").fetchall()
    final_path = root / "validated" / "final_summaries.jsonl"
    raw_path = root / "validated" / "raw_summaries.jsonl"
    audit_path = root / "validated" / "all_validated_summaries.jsonl"
    with AtomicJsonlWriter(final_path) as final_writer, AtomicJsonlWriter(raw_path) as raw_writer, AtomicJsonlWriter(audit_path) as audit_writer:
        for row in rows:
            final_writer.write({"user_id": int(row["user_id"]), "summary": row["final_summary"]})
            raw_writer.write({"user_id": int(row["user_id"]), "summary": row["raw_summary"]})
            audit_writer.write(dict(row))
    lengths = [_words(str(row["final_summary"])) for row in rows]
    attempts = _counts(str(row["attempt"]) for row in rows)
    report = {
        "protocol": "frozen V10 evidence-gated Emiliano", "valid_summaries": len(rows),
        "expected_summaries": USERS, "failures_remaining": 0, "accepted_attempt_counts": attempts,
        "usage": {key: snapshot[key] for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")},
        "exact_cost_usd": snapshot["actual_cost_usd"], "pricing": _json(root / "protocol.json")["pricing"],
        "validation": {
            "hard_failures_remaining": 0, "privacy_leaks_after_sanitization": 0,
            "formatting_failures_after_normalization": 0, "malformed_or_missing": 0,
            "placeholders": 0, "unresolved_duplicates": 0,
            "semantic_post_generation_repairs": 0,
            "strict_positive_classifier_disagreement_is_not_a_failure": True,
            "complete_negative_genre_enumeration_required": False,
        },
        "length_words": _length_stats(lengths),
        "corpus_sha256": sha256_file(final_path), "final_summaries_path": str(final_path),
        "raw_summaries_path": str(raw_path), "audit_records_path": str(audit_path),
        "wandb_url": _json(root / "wandb_monitor_config.json")["run_url"],
        "tears_training_started": False,
    }
    atomic_write_json(root / "reports" / "final_audit_report.json", report)
    metadata = _json(root / "wandb_monitor_config.json")
    _log_wandb(root, metadata, {
        "production/valid_users": USERS, "production/completion_percent": 100.0,
        "cost/actual_usd": snapshot["actual_cost_usd"], "usage/input_tokens": snapshot["input_tokens"],
        "usage/output_tokens": snapshot["output_tokens"], "validation/hard_failures_remaining": 0,
    })
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    plan.add_argument("--config", type=Path, required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--config", type=Path, required=True)
    submit.add_argument("--shard", type=int, required=True)
    submit.add_argument("--attempt", type=int, default=0)
    submit.add_argument("--max-total-cost-usd", type=float, default=CONSERVATIVE_COST_CAP_USD)
    submit.add_argument("--confirm-spend", action="store_true")
    poll = sub.add_parser("poll")
    poll.add_argument("--submission", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--shard", type=int, required=True)
    validate.add_argument("--attempt", type=int, default=0)
    status = sub.add_parser("status")
    status.add_argument("--plan", type=Path, required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--plan", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.action == "plan":
        result = plan_full_cohort(args.root, args.config)
    elif args.action == "submit":
        result = submit_shard(args.plan, args.config, args.shard, args.attempt, args.confirm_spend, args.max_total_cost_usd)
    elif args.action == "poll":
        result = poll_shard(args.submission)
    elif args.action == "validate":
        result = validate_shard(args.plan, args.shard, args.attempt)
    elif args.action == "status":
        result = production_status(args.plan)
    else:
        result = finalize_corpus(args.plan)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
