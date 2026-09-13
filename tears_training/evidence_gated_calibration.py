"""Paired 1,000-user calibration for the evidence-gated Emiliano adaptation.

The command surface is intentionally limited to plan, submit, poll, and
validate for the existing frozen 1,000-user cohort. No full-cohort generation
or TEARS training action exists.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Iterable

import numpy as np

from .artifacts import atomic_write_bytes, atomic_write_json, sha256_file, stable_hash
from .config import load_config
from . import final_summaries as core
from . import production_summary_protocol as legacy
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


USERS = 1_000
MANUAL_AUDIT_USERS = 120
MANUAL_AUDIT_SEED = 20260819
MODEL = "gpt-5-mini-2025-08-07"
INPUT_PRICE_PER_MILLION = 0.25
CACHED_INPUT_PRICE_PER_MILLION = 0.025
OUTPUT_PRICE_PER_MILLION = 2.00
SOURCE_ROOT = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "production_calibration_1000/v002_20260816_v2_generic_repair"
)
PREVIOUS_VALIDATED = SOURCE_ROOT.parent / "v008_20260816_v2_generic_repair_final"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite different artifact: {path}")
        return payload.count(b"\n"), len(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)
    return payload.count(b"\n"), len(payload)


def _write_text(path: Path, value: str) -> None:
    payload = value.encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _words(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))


def _percentile(values: list[int], p: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] + fraction * (ordered[high] - ordered[low])


def _length_stats(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "p95_linear": _percentile(values, 0.95),
        "below_120": sum(value < 120 for value in values),
        "below_150": sum(value < 150 for value in values),
    }


def _genericity_checks(source_plan: dict[str, Any]) -> dict[str, Any]:
    from . import evidence_gated_summary_harness as harness

    source = inspect.getsource(harness)
    tree = ast.parse(source)
    original_ids = set(map(int, source_plan["sample"]["user_ids"]))
    integer_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    }
    imports = {
        alias.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden = (
        "REPAIR_SPECS",
        "INAPPROPRIATE_ABSTENTION_USERS",
        "manual_label",
        "adjudication_note",
        "repair_summary(",
        "compose_final_summary(",
    )
    checks = {
        "no_frozen_user_id_literals": not (integer_literals & original_ids),
        "no_user_specific_exception_maps": not any(
            isinstance(node, ast.Dict)
            and node.keys
            and all(
                isinstance(key, ast.Constant)
                and isinstance(key.value, int)
                and not isinstance(key.value, bool)
                for key in node.keys
                if key is not None
            )
            for node in ast.walk(tree)
        ),
        "no_manual_label_or_semantic_repair_dependency": not any(
            value.lower() in source.lower() for value in forbidden
        ),
        "no_openai_dependency_in_evidence_harness": not any(
            "openai" in value for value in imports
        ),
        "no_tmdb_dependency": not any("tmdb" in value for value in imports),
        "prompt_versioned_as_adaptation": "evidence-gated" in PROTOCOL_VERSION,
    }
    return {"checks": checks, "pass": all(checks.values())}


def _storage_preflight(output_dir: Path, projected_bytes: int) -> dict[str, Any]:
    stat = os.statvfs(output_dir.parent)
    available = stat.f_bavail * stat.f_frsize
    return {
        "target_parent": str(output_dir.parent),
        "filesystem_available_bytes": available,
        "projected_calibration_bytes": projected_bytes,
        "safety_factor": 10,
        "passes_conservative_space_check": available >= projected_bytes * 10,
        "quota_tool_status": (
            "account quota is not exposed by the available quota tools; target is "
            "network scratch and the 1,000-user artifact is small"
        ),
    }


def plan_calibration(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing artifact: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    source_plan = _json(SOURCE_ROOT / "request_plan.json")
    records = source_plan["records"]
    if len(records) != USERS or len({row["user_id"] for row in records}) != USERS:
        raise RuntimeError("Source is not the frozen 1,000-user calibration")
    genericity = _genericity_checks(source_plan)
    if not genericity["pass"]:
        raise RuntimeError(f"Genericity preflight failed: {genericity}")

    evidence_rows: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    request_records: list[dict[str, Any]] = []
    for source in records:
        evidence = build_separated_evidence(source)
        request = build_response_request(source, evidence)
        evidence_rows.append(evidence)
        request_rows.append(request)
        request_records.append(
            {
                "user_id": int(source["user_id"]),
                "custom_id": request["custom_id"],
                "history_hash": source["history_hash"],
                "split": source["split"],
                "activity_band": source["activity_band"],
                "history_items": source["history_items"],
                "titles": source["titles"],
                "negative_evidence_status": evidence["negative_evidence_status"],
                "frozen_evidence_hash": evidence["frozen_evidence_hash"],
            }
        )
    if len({row["custom_id"] for row in request_rows}) != USERS:
        raise RuntimeError("Request custom IDs are not unique")

    statuses = Counter(row["negative_evidence_status"] for row in evidence_rows)
    mixed_users = sum(
        bool(row["genre_classifications"]["mixed_conflicting"])
        for row in evidence_rows
    )
    mixed_instances = sum(
        len(row["genre_classifications"]["mixed_conflicting"])
        for row in evidence_rows
    )
    invariants = {
        "same_exact_user_ids_in_same_order": [row["user_id"] for row in records]
        == source_plan["sample"]["user_ids"],
        "exactly_1000_unique_users": len({row["user_id"] for row in records}) == USERS,
        "all_statuses_known": set(statuses) <= {"NONE", "WEAK", "STRONG"},
        "none_has_zero_negative_examples": all(
            row["negative_example_count"] == 0
            for row in evidence_rows
            if row["negative_evidence_status"] == "NONE"
        ),
        "positive_negative_example_blocks_disjoint": all(
            not (
                {
                    item["source_movie_id"]
                    for item in row["inference_payload"]["positive_evidence"]
                }
                & {
                    item["source_movie_id"]
                    for item in row["inference_payload"]["negative_evidence"]
                }
            )
            for row in evidence_rows
        ),
        "supported_positive_negative_genres_disjoint": all(
            not (
                set(row["genre_classifications"]["supported_positive"])
                & set(row["genre_classifications"]["supported_negative"])
            )
            for row in evidence_rows
        ),
        "mixed_genres_excluded_from_supported_lists": all(
            not (
                set(row["genre_classifications"]["mixed_conflicting"])
                & (
                    set(row["genre_classifications"]["supported_positive"])
                    | set(row["genre_classifications"]["supported_negative"])
                )
            )
            for row in evidence_rows
        ),
        "no_numeric_rating_fields_in_inference_payload": all(
            "rating" not in json.dumps(row["inference_payload"]).lower()
            for row in evidence_rows
        ),
    }
    # "preference_signal" is intentionally qualitative; ensure the overly broad
    # substring check above does not confuse it with a numeric rating field.
    invariants["no_numeric_rating_fields_in_inference_payload"] = all(
        not re.search(
            r'"(?:rating|numeric_rating)"\s*:',
            json.dumps(row["inference_payload"]),
            re.I,
        )
        for row in evidence_rows
    )

    request_path = output_dir / "requests" / "calibration-1000.jsonl"
    count, request_bytes = _write_jsonl(request_path, request_rows)
    _write_jsonl(output_dir / "evidence" / "all_user_evidence.jsonl", evidence_rows)
    projected_artifact_bytes = request_bytes * 12
    storage = _storage_preflight(output_dir, projected_artifact_bytes)
    if not storage["passes_conservative_space_check"]:
        raise RuntimeError(f"Storage preflight failed: {storage}")
    estimated_input = sum(
        core.estimate_tokens(
            row["body"]["input"][0]["content"]
            + row["body"]["input"][1]["content"]
        )
        for row in request_rows
    )
    reserved_output = USERS * 450
    projected_cost = (
        estimated_input * INPUT_PRICE_PER_MILLION
        + reserved_output * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000
    protocol = {
        "version": PROTOCOL_VERSION,
        "classification": "experimental evidence-gated Emiliano adaptation",
        "model": MODEL,
        "system_prompt_sha256": PROMPT_SHA256,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "generation": {
            "endpoint": "/v1/responses",
            "model": MODEL,
            "reasoning_effort": "minimal",
            "text_verbosity": "low",
            "max_output_tokens": 450,
            "store": False,
            "structured_output": "strict JSON schema with one summary string",
        },
        "post_generation": {
            "allowed": [
                "privacy title/year/numeric-rating sanitization",
                "Summary prefix and whitespace normalization",
                "malformed-output handling",
                "duplicate detection",
            ],
            "semantic_repair": False,
            "deterministic_negative_append": False,
        },
        "tmdb_semantic_metadata": False,
    }
    protocol["fingerprint"] = stable_hash(protocol)
    plan = {
        "scope": "evidence_gated_paired_calibration_1000_only",
        "expected_requests": USERS,
        "requests": count,
        "unique_users": USERS,
        "source_calibration_plan": str((SOURCE_ROOT / "request_plan.json").resolve()),
        "source_calibration_fingerprint": source_plan["fingerprint"],
        "same_user_ids_sha256": source_plan["sample"]["user_ids_sha256"],
        "same_user_ids": source_plan["sample"]["user_ids"],
        "protocol": protocol,
        "records": request_records,
        "request_file": {
            "path": str(request_path),
            "sha256": sha256_file(request_path),
            "bytes": request_bytes,
            "requests": count,
        },
        "estimated_input_tokens": estimated_input,
        "reserved_output_tokens": reserved_output,
        "projected_reserved_cost_usd": projected_cost,
    }
    plan["fingerprint"] = stable_hash(plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "request_plan.json", plan)
    atomic_write_json(output_dir / "protocol.json", protocol)
    atomic_write_json(output_dir / "prompt" / "prompt_delta.json", prompt_delta())
    _write_text(output_dir / "prompt" / "evidence_gated_emiliano_prompt.txt", EVIDENCE_GATED_PROMPT)
    _write_text(output_dir / "prompt" / "original_emiliano_prompt.txt", prompt_delta()["source_prompt"])
    preflight = {
        "offline_only": True,
        "paid_requests_made": 0,
        "users": USERS,
        "status_distribution": dict(sorted(statuses.items())),
        "status_percent": {
            key: 100 * value / USERS for key, value in sorted(statuses.items())
        },
        "mixed_conflicting_genre_users": mixed_users,
        "mixed_conflicting_genre_instances": mixed_instances,
        "genericity": genericity,
        "evidence_invariants": invariants,
        "storage": storage,
        "request_sha256": plan["request_file"]["sha256"],
        "evidence_sha256": sha256_file(
            output_dir / "evidence" / "all_user_evidence.jsonl"
        ),
        "projected_reserved_cost_usd": projected_cost,
        "api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "wandb_credentials_available": bool(
            os.environ.get("WANDB_API_KEY") or (Path.home() / ".netrc").is_file()
        ),
        "full_cohort_started": False,
        "tears_training_started": False,
    }
    preflight["pass"] = bool(
        genericity["pass"]
        and all(invariants.values())
        and storage["passes_conservative_space_check"]
        and preflight["api_key_present"]
        and preflight["wandb_credentials_available"]
    )
    atomic_write_json(output_dir / "reports" / "preflight_report.json", preflight)
    source_files = (
        Path(inspect.getsourcefile(build_separated_evidence) or ""),
        Path(__file__),
        Path("tests/test_evidence_gated_summary_harness.py"),
    )
    for path in source_files:
        _write_text(output_dir / "code" / path.name, path.read_text())
    atomic_write_json(
        output_dir / "code" / "manifest.json",
        {path.name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in source_files},
    )
    return preflight


def _verified_plan(path: Path) -> dict[str, Any]:
    plan = _json(path)
    fingerprint = plan.pop("fingerprint")
    actual = stable_hash(plan)
    plan["fingerprint"] = fingerprint
    if fingerprint != actual:
        raise RuntimeError("Plan fingerprint mismatch")
    if (
        plan["scope"] != "evidence_gated_paired_calibration_1000_only"
        or plan["requests"] != USERS
        or plan["unique_users"] != USERS
    ):
        raise RuntimeError("Plan is not the exact paired 1,000-user calibration")
    request_path = Path(plan["request_file"]["path"])
    if sha256_file(request_path) != plan["request_file"]["sha256"]:
        raise RuntimeError("Request hash mismatch")
    preflight = _json(path.parent / "reports" / "preflight_report.json")
    if not preflight["pass"] or preflight["paid_requests_made"] != 0:
        raise RuntimeError("Offline preflight did not pass")
    return plan


def _wandb_metadata(root: Path, config_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    config = load_config(config_path)
    run_id = stable_hash(
        {"purpose": "summary-batch-monitor-v1", "plan": plan["fingerprint"]}
    )[:16]
    return {
        "entity": config.tracking.entity,
        "project": config.tracking.project,
        "mode": config.tracking.mode,
        "run_id": run_id,
        "run_name": f"evidence-gated-summary-calibration-1000-{run_id[:8]}",
        "run_url": f"https://wandb.ai/{config.tracking.entity}/{config.tracking.project}/runs/{run_id}",
        "job_type": "summary-batch-monitor",
        "weave_enabled": False,
        "root": str(root),
        "plan_fingerprint": plan["fingerprint"],
    }


def _log_wandb(root: Path, metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
    import wandb

    wandb_dir = root / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=metadata["entity"],
        project=metadata["project"],
        name=metadata["run_name"],
        id=metadata["run_id"],
        resume="allow",
        job_type=metadata["job_type"],
        config=metadata,
        mode=metadata["mode"],
        dir=str(wandb_dir),
    )
    run.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    run.finish()


def submit_calibration(
    plan_path: Path,
    config_path: Path,
    *,
    confirmed: bool,
    max_cost_usd: float,
) -> dict[str, Any]:
    if not confirmed:
        raise RuntimeError("Paid submission requires explicit confirmation")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    plan = _verified_plan(plan_path)
    root = plan_path.parent
    if (root / "submission.json").exists():
        raise RuntimeError("Submission already exists; refusing duplicate spend")
    projection = float(plan["projected_reserved_cost_usd"])
    if projection <= 0 or projection > max_cost_usd:
        raise RuntimeError(
            f"Projected reservation ${projection:.6f} exceeds cap ${max_cost_usd:.6f}"
        )
    metadata = _wandb_metadata(root, config_path, plan)
    atomic_write_json(root / "wandb_monitor_config.json", metadata)
    _log_wandb(
        root,
        metadata,
        {
            "batch/total_requests": USERS,
            "batch/completed_requests": 0,
            "batch/failed_requests": 0,
            "batch/completion_percent": 0.0,
            "cost/estimated_usd": projection,
            "monitor/preflight_pass": 1,
        },
    )
    from openai import OpenAI

    client = OpenAI()
    with Path(plan["request_file"]["path"]).open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "protocol": PROTOCOL_VERSION,
            "scope": "paired-calibration-1000-only",
        },
    )
    result = {
        "scope": plan["scope"],
        "submitted_requests": USERS,
        "plan_fingerprint": plan["fingerprint"],
        "input_file_id": uploaded.id,
        "batch_id": batch.id,
        "status": batch.status,
        "submitted_at": int(time.time()),
        "projected_reserved_cost_usd": projection,
        "maximum_authorized_cost_usd": max_cost_usd,
        "wandb_run_url": metadata["run_url"],
        "weave_enabled": False,
        "full_cohort_started": False,
        "tears_training_started": False,
    }
    atomic_write_json(root / "submission.json", result)
    _log_wandb(
        root,
        metadata,
        {
            "batch/total_requests": USERS,
            "batch/completed_requests": 0,
            "batch/failed_requests": 0,
            "batch/completion_percent": 0.0,
            "cost/estimated_usd": projection,
            "monitor/submitted": 1,
        },
    )
    return result


def poll_calibration(submission_path: Path) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    submission = _json(submission_path)
    if submission["submitted_requests"] != USERS:
        raise RuntimeError("Submission is not exactly 1,000 requests")
    from openai import OpenAI

    client = OpenAI()
    batch = client.batches.retrieve(submission["batch_id"])
    status = batch.model_dump(mode="json")
    root = submission_path.parent
    for kind, file_id in (("responses", batch.output_file_id), ("errors", batch.error_file_id)):
        if not file_id:
            continue
        payload = client.files.content(file_id).content
        path = root / kind / f"{batch.id}.jsonl"
        if path.exists() and path.read_bytes() != payload:
            raise RuntimeError(f"Downloaded {kind} changed")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(path, payload)
    result = {"batch": status, "polled_at": int(time.time())}
    atomic_write_json(root / "poll.json", result)
    counts = status.get("request_counts") or {}
    metadata = _json(root / "wandb_monitor_config.json")
    total = int(counts.get("total", USERS))
    completed = int(counts.get("completed", 0))
    failed = int(counts.get("failed", 0))
    _log_wandb(
        root,
        metadata,
        {
            "batch/total_requests": total,
            "batch/completed_requests": completed,
            "batch/failed_requests": failed,
            "batch/completion_percent": completed / total * 100 if total else 0.0,
            "batch/status_code": 1 if status.get("status") == "completed" else 0,
        },
    )
    return result


def _response_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "responses").glob("*.jsonl")):
        rows.extend(_jsonl(path))
    return rows


def _negative_sentences(summary: str) -> list[str]:
    return [
        sentence
        for sentence in legacy.SENTENCE_SPLIT.split(summary)
        if legacy._is_user_negative_claim(sentence) or legacy._is_abstention(sentence)
    ]


def _automatic_semantic_flags(summary: str, evidence: dict[str, Any]) -> dict[str, Any]:
    status = evidence["negative_evidence_status"]
    negatives = _negative_sentences(summary)
    abstentions = [sentence for sentence in negatives if legacy._is_abstention(sentence)]
    claims = [sentence for sentence in negatives if not legacy._is_abstention(sentence)]
    mentioned = sorted({genre for sentence in claims for genre in legacy._mentioned_genres(sentence)})
    supported_negative = set(evidence["genre_classifications"]["supported_negative"])
    mixed = set(evidence["genre_classifications"]["mixed_conflicting"])
    categorical = [
        sentence
        for sentence in claims
        if re.search(r"\b(?:does not enjoy|dislikes?|avoids?|not a fan of|averse to)\b", sentence, re.I)
    ]
    cautious = [
        sentence
        for sentence in claims
        if re.search(r"\b(?:may|might|seems?|suggests?|tends?|less|limited|cautious|tentative)\b", sentence, re.I)
    ]
    return {
        "negative_sentences": negatives,
        "negative_claim_sentences": claims,
        "abstention_sentences": abstentions,
        "mentioned_negative_genres": mentioned,
        "none_fabricated_dislike": status == "NONE" and bool(claims),
        "weak_categorical_broad_claim": status == "WEAK" and bool(categorical),
        "weak_claim_without_cautious_language": status == "WEAK" and bool(claims) and not cautious,
        "strong_inappropriate_abstention": status == "STRONG" and bool(abstentions) and not claims,
        "supported_negative_genres_omitted": sorted(
            supported_negative - set(mentioned)
        ) if status == "STRONG" else [],
        "mixed_genres_stated_negative": sorted(mixed & set(mentioned)),
        "unsupported_categorical_genres": sorted(
            set(mentioned) - supported_negative
        ) if categorical else [],
    }


def _duplicate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        exact[_normal(row["final_summary"])].append(row["user_id"])
    exact_groups = [ids for ids in exact.values() if len(ids) > 1]
    near: list[dict[str, Any]] = []
    prepared = [
        (row["user_id"], _normal(row["final_summary"])) for row in rows
    ]
    for index, (left_id, left) in enumerate(prepared):
        left_words = set(left.split())
        for right_id, right in prepared[index + 1 :]:
            if min(len(left), len(right)) / max(len(left), len(right)) < 0.88:
                continue
            right_words = set(right.split())
            if len(left_words & right_words) / len(left_words | right_words) < 0.72:
                continue
            similarity = SequenceMatcher(None, left, right).ratio()
            if similarity >= 0.90:
                near.append({"left_user_id": left_id, "right_user_id": right_id, "similarity": similarity})
    return {"exact_groups": exact_groups, "exact_group_count": len(exact_groups), "near_pairs": near, "near_pair_count": len(near)}


def _manual_sample(rows: list[dict[str, Any]], evidence: dict[int, dict[str, Any]]) -> dict[str, Any]:
    strata: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        user_id = row["user_id"]
        status = evidence[user_id]["negative_evidence_status"]
        mixed = "mixed" if evidence[user_id]["genre_classifications"]["mixed_conflicting"] else "not_mixed"
        strata[f"{status}/{mixed}"].append(user_id)
    minimums = {key: min(15, len(values)) for key, values in strata.items()}
    selected: list[int] = []
    for offset, key in enumerate(sorted(strata)):
        values = np.asarray(sorted(strata[key]), dtype=np.int64)
        np.random.default_rng(MANUAL_AUDIT_SEED + offset).shuffle(values)
        selected.extend(map(int, values[: minimums[key]]))
    remaining = MANUAL_AUDIT_USERS - len(selected)
    pool = np.asarray(sorted(set(row["user_id"] for row in rows) - set(selected)), dtype=np.int64)
    np.random.default_rng(MANUAL_AUDIT_SEED + 100).shuffle(pool)
    selected.extend(map(int, pool[:remaining]))
    selected = sorted(selected)
    if len(selected) != MANUAL_AUDIT_USERS or len(set(selected)) != MANUAL_AUDIT_USERS:
        raise RuntimeError("Manual sample is not exactly 120 unique users")
    return {
        "seed": MANUAL_AUDIT_SEED,
        "users": MANUAL_AUDIT_USERS,
        "user_ids": selected,
        "stratum_counts": dict(
            Counter(
                f"{evidence[user_id]['negative_evidence_status']}/"
                + ("mixed" if evidence[user_id]["genre_classifications"]["mixed_conflicting"] else "not_mixed")
                for user_id in selected
            )
        ),
    }


def validate_calibration(plan_path: Path) -> dict[str, Any]:
    plan = _verified_plan(plan_path)
    root = plan_path.parent
    records = {row["custom_id"]: row for row in plan["records"]}
    evidence = {
        int(row["user_id"]): row
        for row in _jsonl(root / "evidence" / "all_user_evidence.jsonl")
    }
    response_rows = _response_rows(root)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for response_row in response_rows:
        custom_id = response_row.get("custom_id")
        if custom_id not in records or custom_id in seen:
            continue
        seen.add(custom_id)
        record = records[custom_id]
        response = response_row.get("response") or {}
        body = response.get("body") or {}
        errors: list[str] = []
        raw_summary = ""
        if response.get("status_code") != 200:
            errors.append(f"http_status:{response.get('status_code')}")
        else:
            raw_text = core.extract_output_text(body)
            try:
                raw_summary = str(json.loads(raw_text)["summary"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                errors.append("invalid_structured_output")
        cleanup = privacy_and_format_cleanup(raw_summary, record["titles"]) if raw_summary else None
        final_summary = cleanup.final_summary if cleanup else ""
        semantic = _automatic_semantic_flags(final_summary, evidence[record["user_id"]]) if final_summary else {}
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0))
        cached_tokens = int((usage.get("input_tokens_details") or {}).get("cached_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        reasoning_tokens = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0))
        cost = (
            (input_tokens - cached_tokens) * INPUT_PRICE_PER_MILLION
            + cached_tokens * CACHED_INPUT_PRICE_PER_MILLION
            + output_tokens * OUTPUT_PRICE_PER_MILLION
        ) / 1_000_000
        rows.append(
            {
                "user_id": record["user_id"],
                "custom_id": custom_id,
                "negative_evidence_status": evidence[record["user_id"]]["negative_evidence_status"],
                "mixed_conflicting_genres": evidence[record["user_id"]]["genre_classifications"]["mixed_conflicting"],
                "raw_summary": raw_summary,
                "final_summary": final_summary,
                "raw_word_count": _words(raw_summary),
                "final_word_count": _words(final_summary),
                "cleanup_operations": list(cleanup.operations) if cleanup else [],
                "privacy": {
                    "title": list(cleanup.title_matches_after) if cleanup else [],
                    "year": list(cleanup.year_matches_after) if cleanup else [],
                    "numeric_rating": list(cleanup.numeric_rating_matches_after) if cleanup else [],
                },
                "formatting_issues": list(cleanup.formatting_issues) if cleanup else [],
                "semantic_flags": semantic,
                "errors": errors,
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "cost_usd": cost,
            }
        )
    missing_ids = sorted(
        set(record["user_id"] for record in records.values())
        - set(row["user_id"] for row in rows)
    )
    duplicates = _duplicate_report([row for row in rows if row["final_summary"]])
    raw_lengths = [row["raw_word_count"] for row in rows if row["raw_summary"]]
    final_lengths = [row["final_word_count"] for row in rows if row["final_summary"]]
    cleanup_users = sum(bool(row["cleanup_operations"]) for row in rows)
    total_input = sum(row["input_tokens"] for row in rows)
    total_cached = sum(row["cached_input_tokens"] for row in rows)
    total_output = sum(row["output_tokens"] for row in rows)
    exact_cost = sum(row["cost_usd"] for row in rows)
    sample = _manual_sample(rows, evidence)
    sample_set = set(sample["user_ids"])
    queue = [
        {
            "user_id": row["user_id"],
            "negative_evidence_status": row["negative_evidence_status"],
            "mixed_conflicting_genres": row["mixed_conflicting_genres"],
            "evidence": evidence[row["user_id"]],
            "raw_summary": row["raw_summary"],
            "final_summary": row["final_summary"],
            "automated_semantic_flags": row["semantic_flags"],
        }
        for row in rows
        if row["user_id"] in sample_set
    ]
    report = {
        "scope": plan["scope"],
        "received_unique": len(rows),
        "missing": len(missing_ids),
        "missing_user_ids": missing_ids,
        "api_success": sum(not row["errors"] for row in rows),
        "invalid_structured_output": sum(bool(row["errors"]) for row in rows),
        "privacy_leaks_after_cleanup": {
            key: sum(bool(row["privacy"][key]) for row in rows)
            for key in ("title", "year", "numeric_rating")
        },
        "formatting_failures": sum(bool(row["formatting_issues"]) for row in rows),
        "placeholders": sum(
            bool(re.search(r"\{[^{}]+\}|\b(?:TBD|TODO)\b", row["final_summary"], re.I))
            for row in rows
        ),
        "semantic_automated_screen": {
            key: sum(bool(row["semantic_flags"].get(key)) for row in rows)
            for key in (
                "none_fabricated_dislike",
                "weak_categorical_broad_claim",
                "weak_claim_without_cautious_language",
                "strong_inappropriate_abstention",
                "supported_negative_genres_omitted",
                "mixed_genres_stated_negative",
                "unsupported_categorical_genres",
            )
        },
        "cleanup": {
            "users": cleanup_users,
            "percent": cleanup_users / USERS * 100,
            "semantic_post_generation_repair_users": 0,
        },
        "duplicates": duplicates,
        "length": {"raw": _length_stats(raw_lengths), "final": _length_stats(final_lengths)},
        "usage": {
            "input_tokens": total_input,
            "cached_input_tokens": total_cached,
            "output_tokens": total_output,
            "reasoning_tokens": sum(row["reasoning_tokens"] for row in rows),
        },
        "cost": {
            "exact_calibration_usd": exact_cost,
            "mean_per_user_usd": exact_cost / USERS,
            "projected_full_200948_usd": exact_cost / USERS * 200_948,
        },
        "manual_audit_sample": sample,
        "manual_audit_pending": True,
        "full_cohort_started": False,
        "tears_training_started": False,
    }
    _write_jsonl(root / "validated" / "all_summaries.jsonl", rows)
    _write_jsonl(root / "validated" / "raw_summaries.jsonl", [
        {"user_id": row["user_id"], "summary": row["raw_summary"]} for row in rows
    ])
    _write_jsonl(root / "validated" / "final_summaries.jsonl", [
        {"user_id": row["user_id"], "summary": row["final_summary"]} for row in rows
    ])
    _write_jsonl(root / "reports" / "manual_audit_queue.jsonl", queue)
    atomic_write_json(root / "reports" / "automated_validation_report.json", report)
    metadata = _json(root / "wandb_monitor_config.json")
    _log_wandb(
        root,
        metadata,
        {
            "validation/valid_summaries": report["api_success"],
            "validation/invalid_summaries": report["missing"] + report["invalid_structured_output"],
            "usage/input_tokens": total_input,
            "usage/output_tokens": total_output,
            "cost/actual_usd": exact_cost,
            "quality/summary_length_mean": report["length"]["final"]["mean"],
            "quality/summary_length_p95": report["length"]["final"]["p95_linear"],
            "quality/cleanup_users": cleanup_users,
            "quality/formatting_failures": report["formatting_failures"],
        },
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--output-dir", type=Path, required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--config", type=Path, required=True)
    submit.add_argument("--max-cost-usd", type=float, default=5.0)
    submit.add_argument("--confirm-spend", action="store_true")
    poll = sub.add_parser("poll")
    poll.add_argument("--submission", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.action == "plan":
        result = plan_calibration(args.output_dir)
    elif args.action == "submit":
        result = submit_calibration(
            args.plan,
            args.config,
            confirmed=args.confirm_spend,
            max_cost_usd=args.max_cost_usd,
        )
    elif args.action == "poll":
        result = poll_calibration(args.submission)
    else:
        result = validate_calibration(args.plan)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
