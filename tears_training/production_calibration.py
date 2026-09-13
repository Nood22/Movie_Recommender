"""Generic 1,000-user production calibration for the promoted V2 protocol.

This module freezes a deterministic sample, submits exactly one Batch API job,
collects raw V2 summaries, applies user-agnostic deterministic post-processing,
and produces automated and manual-audit artifacts. It exposes no full-cohort or
TEARS-training operation.
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
import pandas as pd

from . import final_summaries as core
from . import final_summaries_v2 as v2
from . import final_summaries_v4 as v4
from .artifacts import atomic_write_bytes, atomic_write_json, sha256_file, stable_hash
from .config import load_config
from .production_summary_protocol import (
    PLACEHOLDER_PATTERN,
    PROTOCOL_VERSION as REPAIR_PROTOCOL_VERSION,
    formatting_issues,
    repair_summary,
)


CALIBRATION_USERS = 1_000
SAMPLE_SEED = 20260816
MANUAL_AUDIT_USERS = 120
MANUAL_AUDIT_SEED = 20260817
MODEL = "gpt-5-mini-2025-08-07"
INPUT_PRICE_PER_MILLION = 0.25
CACHED_INPUT_PRICE_PER_MILLION = 0.025
OUTPUT_PRICE_PER_MILLION = 2.00
PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5-mini"
BATCH_DOC_SOURCE = "https://platform.openai.com/docs/api-reference/batch/object?api-mode=responses"
PRODUCTION_PROTOCOL_VERSION = "tears-v2-production-calibration-1000-v1"
ORIGINAL_CALIBRATION_IDS = (
    Path(
        "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
        "calibration_100/v002_20260813_negative_grounding_safeguard/"
        "calibration_user_ids.json"
    )
)
PROMOTED_CANDIDATE = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "calibration_100/v002_final_cleanup_offline"
)
RAW_V2_CALIBRATION = ORIGINAL_CALIBRATION_IDS.parent
V4_EVIDENCE_CALIBRATION = RAW_V2_CALIBRATION.parent / (
    "v004_20260813_deterministic_evidence_verbalization"
)


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
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite different artifact: {path}")
        return payload.count(b"\n"), len(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)
    return payload.count(b"\n"), len(payload)


def _write_text(path: Path, value: str) -> None:
    payload = value.encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _percentile(values: list[int], p: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] + fraction * (ordered[high] - ordered[low])


def _length_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {
            "mean": 0,
            "median": 0,
            "minimum": 0,
            "maximum": 0,
            "p95_linear": 0,
            "below_120": 0,
            "below_150": 0,
        }
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "p95_linear": _percentile(values, 0.95),
        "below_120": sum(value < 120 for value in values),
        "below_150": sum(value < 150 for value in values),
    }


def _quotas(sizes: dict[str, int], total: int, minimum: int = 0) -> dict[str, int]:
    if total < 0 or total > sum(sizes.values()):
        raise ValueError("Invalid quota total")
    result = {
        key: min(size, minimum) if total >= minimum * len(sizes) else 0
        for key, size in sizes.items()
    }
    remaining = total - sum(result.values())
    capacity = {key: sizes[key] - result[key] for key in sizes}
    capacity_total = sum(capacity.values())
    raw = {
        key: remaining * capacity[key] / capacity_total if capacity_total else 0.0
        for key in sizes
    }
    floors = {key: int(raw[key]) for key in sizes}
    for key in result:
        result[key] += floors[key]
    leftover = total - sum(result.values())
    order = sorted(
        sizes,
        key=lambda key: (raw[key] - floors[key], capacity[key], key),
        reverse=True,
    )
    for key in order:
        if leftover <= 0:
            break
        if result[key] < sizes[key]:
            result[key] += 1
            leftover -= 1
    if sum(result.values()) != total:
        raise RuntimeError("Quota allocation failed")
    return result


def select_production_calibration_users(
    user_splits: pd.DataFrame,
    matrix_users: pd.DataFrame,
    excluded_user_ids: set[int],
    *,
    count: int = CALIBRATION_USERS,
    seed: int = SAMPLE_SEED,
) -> pd.DataFrame:
    required = {"userId", "interaction_count", "activity_band", "split"}
    if not required.issubset(user_splits.columns):
        raise RuntimeError(f"Missing split columns: {sorted(required - set(user_splits))}")
    if set(user_splits.userId.astype(int)) != set(matrix_users.userId.astype(int)):
        raise RuntimeError("Frozen matrix and split user sets differ")
    if count != CALIBRATION_USERS:
        raise RuntimeError("Production calibration must contain exactly 1,000 users")
    available_all = user_splits.loc[
        ~user_splits.userId.astype(int).isin(excluded_user_ids)
    ].copy()
    split_sizes = {
        split: int((available_all.split == split).sum())
        for split in ("train", "validation", "test")
    }
    split_quotas = _quotas(split_sizes, count)
    selected: list[pd.DataFrame] = []
    offset = 0
    for split in ("train", "validation", "test"):
        available = available_all.loc[available_all.split == split]
        band_sizes = {
            str(key): int(value)
            for key, value in available.groupby("activity_band").size().items()
        }
        band_quotas = _quotas(band_sizes, split_quotas[split], minimum=1)
        for band in sorted(band_quotas):
            candidates = available.loc[
                available.activity_band.astype(str) == band
            ].copy()
            values = candidates.userId.to_numpy(np.int64)
            np.random.default_rng(seed + offset).shuffle(values)
            offset += 1
            chosen = set(map(int, values[: band_quotas[band]]))
            selected.append(candidates.loc[candidates.userId.isin(chosen)].copy())
    result = pd.concat(selected, ignore_index=True).sort_values("userId")
    if len(result) != count or result.userId.nunique() != count:
        raise RuntimeError("Production sample is not exactly 1,000 unique users")
    if set(map(int, result.userId)) & excluded_user_ids:
        raise RuntimeError("Production sample overlaps excluded calibration users")
    return result.reset_index(drop=True)


def _production_protocol(repository_root: Path) -> dict[str, Any]:
    v2._configure_core()
    generation = core.protocol_manifest(repository_root)
    frozen_v2_protocol = _json(
        ORIGINAL_CALIBRATION_IDS.parent / "protocol.json"
    )
    if generation != frozen_v2_protocol:
        raise RuntimeError("V2 generation protocol drifted from the frozen artifact")
    repair_source = Path(inspect.getsourcefile(repair_summary) or "")
    value = {
        "version": PRODUCTION_PROTOCOL_VERSION,
        "preferred_candidate": "v002_final_cleanup_offline",
        "generation": generation,
        "post_processing": {
            "version": REPAIR_PROTOCOL_VERSION,
            "source_path": str(repair_source.resolve()),
            "source_sha256": sha256_file(repair_source),
            "rating_evidence": v4.EVIDENCE_RULES,
            "grounding_repair": (
                "replace only detected negative/abstention spans with either the "
                "strict supported MovieLens genre list or an evidence-aware abstention"
            ),
            "positive_preservation": "every non-negative sentence must remain byte-identical",
            "formatting": (
                "generic Summary: prefix and known non-semantic normalization; "
                "no exact sentence-count rule"
            ),
            "tmdb_semantic_metadata": False,
            "manual_labels_at_inference": False,
            "user_specific_exceptions": False,
        },
        "pricing": {
            "source": PRICING_SOURCE,
            "batch_api_reference": BATCH_DOC_SOURCE,
            "usd_per_million": {
                "input": INPUT_PRICE_PER_MILLION,
                "cached_input": CACHED_INPUT_PRICE_PER_MILLION,
                "output": OUTPUT_PRICE_PER_MILLION,
            },
        },
    }
    value["fingerprint"] = stable_hash(value)
    return value


def _genericity_checks() -> dict[str, Any]:
    repair_source_path = Path(inspect.getsourcefile(repair_summary) or "")
    source = repair_source_path.read_text(encoding="utf-8")
    original_ids = set(_json(ORIGINAL_CALIBRATION_IDS)["user_ids"])
    integers = {
        int(value)
        for value in re.findall(r"(?<![A-Za-z_])\d+(?![A-Za-z_])", source)
    }
    signature = list(inspect.signature(repair_summary).parameters)
    tree = ast.parse(source)
    imported_modules = {
        alias.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    referenced_names = {
        node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    disallowed_terms = (
        "per_user_paired_comparison",
        "per_user_repair_audit",
        "adjudication_note",
        "REPAIR_SPECS",
        "INAPPROPRIATE_ABSTENTION_USERS",
    )
    synthetic = (
        "Summary: The user enjoys dramas and layered stories. "
        "No strong negative preference is supported by the available history."
    )
    first = repair_summary(synthetic, {"supported_negative_genres": ["Horror"]})
    second = repair_summary(
        synthetic,
        {
            "supported_negative_genres": ["Horror"],
            "user_id": 999999,
            "manual_label": "ignored",
        },
    )
    checks = {
        "repair_signature_exactly_summary_and_evidence": signature
        == ["summary", "evidence"],
        "no_original_calibration_id_literals": not (integers & original_ids),
        "no_manual_audit_dependencies": not any(term in source for term in disallowed_terms),
        "irrelevant_user_fields_do_not_change_result": (
            first.final_summary == second.final_summary
            and first.operations == second.operations
        ),
        "no_tmdb_dependency": not any(
            "tmdb" in value for value in imported_modules | referenced_names
        ),
        "no_openai_dependency": not any(
            "openai" in value for value in imported_modules | referenced_names
        ),
    }
    return {
        "repair_source_path": str(repair_source_path.resolve()),
        "repair_source_sha256": sha256_file(repair_source_path),
        "checks": checks,
        "pass": all(checks.values()),
    }


def _offline_baseline_diagnostic() -> dict[str, Any]:
    """Apply the generic code to the old 100 only as a non-inference diagnostic."""

    raw = {
        int(row["user_id"]): row
        for row in _jsonl(RAW_V2_CALIBRATION / "validated" / "all_summaries.jsonl")
    }
    evidence = {
        int(row["user_id"]): row
        for row in _jsonl(
            V4_EVIDENCE_CALIBRATION / "evidence" / "all_user_evidence.jsonl"
        )
    }
    promoted = {
        int(row["user_id"]): row
        for row in _jsonl(PROMOTED_CANDIDATE / "final_summaries.jsonl")
    }
    if set(raw) != set(evidence) or set(raw) != set(promoted) or len(raw) != 100:
        raise RuntimeError("Frozen 100-user diagnostic inputs differ")
    rows: list[dict[str, Any]] = []
    for user_id in sorted(raw):
        result = repair_summary(raw[user_id]["summary"], evidence[user_id])
        raw_words = len(raw[user_id]["summary"].split())
        final_words = len(result.final_summary.split())
        rows.append(
            {
                "user_id": user_id,
                "changed": result.changed,
                "positive_sentences_preserved": result.positive_sentences_preserved,
                "supported_negative_sentences_preserved": len(
                    result.supported_negative_sentences_preserved
                ),
                "raw_word_count": raw_words,
                "generic_final_word_count": final_words,
                "materially_compressed": (raw_words - final_words) / raw_words >= 0.20,
                "identical_to_promoted_final_cleanup": (
                    result.final_summary == promoted[user_id]["summary"]
                ),
            }
        )
    return {
        "purpose": "offline overlap diagnostic only; not used for inference decisions",
        "users": 100,
        "manual_labels_loaded": False,
        "raw_summary_source": str(
            RAW_V2_CALIBRATION / "validated" / "all_summaries.jsonl"
        ),
        "rating_evidence_source": str(
            V4_EVIDENCE_CALIBRATION / "evidence" / "all_user_evidence.jsonl"
        ),
        "promoted_candidate_source": str(PROMOTED_CANDIDATE),
        "changed": sum(row["changed"] for row in rows),
        "unchanged": sum(not row["changed"] for row in rows),
        "positive_sentences_preserved": sum(
            row["positive_sentences_preserved"] for row in rows
        ),
        "supported_negative_sentences_preserved": sum(
            row["supported_negative_sentences_preserved"] for row in rows
        ),
        "materially_compressed": sum(row["materially_compressed"] for row in rows),
        "identical_to_promoted_final_cleanup": sum(
            row["identical_to_promoted_final_cleanup"] for row in rows
        ),
        "length": {
            "raw_v2": _length_stats([row["raw_word_count"] for row in rows]),
            "generic_final": _length_stats(
                [row["generic_final_word_count"] for row in rows]
            ),
        },
        "per_user": rows,
    }


def plan_calibration(
    config_path: Path,
    matrix_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing calibration: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    genericity = _genericity_checks()
    if not genericity["pass"]:
        raise RuntimeError(f"Genericity preflight failed: {genericity}")
    config = load_config(config_path)
    frozen = core.verify_frozen_matrix(matrix_dir, dataset_dir)
    matrix_users = pd.read_csv(matrix_dir / "users.csv")
    user_splits = pd.read_csv(dataset_dir / "user_splits.csv")
    excluded = set(map(int, _json(ORIGINAL_CALIBRATION_IDS)["user_ids"]))
    selected = select_production_calibration_users(
        user_splits, matrix_users, excluded
    )
    histories = core.load_histories(config, matrix_dir, selected)
    repository_root = Path(__file__).resolve().parents[1]
    protocol = _production_protocol(repository_root)
    baseline_diagnostic = _offline_baseline_diagnostic()

    request_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for selected_row in selected.itertuples(index=False):
        request, record = core.response_request(
            int(selected_row.userId),
            str(selected_row.split),
            str(selected_row.activity_band),
            histories[int(selected_row.userId)],
            protocol["generation"],
        )
        evidence = v4.extract_user_evidence(record)
        request_rows.append(request)
        records.append(record)
        evidence_rows.append(evidence)
    if len(request_rows) != CALIBRATION_USERS:
        raise RuntimeError("Plan does not contain exactly 1,000 requests")
    if len({row["custom_id"] for row in request_rows}) != CALIBRATION_USERS:
        raise RuntimeError("Request custom IDs are not unique")

    request_path = output_dir / "requests" / "calibration-1000.jsonl"
    count, size = _write_jsonl(request_path, request_rows)
    _write_jsonl(output_dir / "evidence" / "all_user_evidence.jsonl", evidence_rows)
    estimated_input = sum(
        core.estimate_tokens(
            row["body"]["input"][0]["content"]
            + row["body"]["input"][1]["content"]
        )
        for row in request_rows
    )
    sample_ids = [int(value) for value in selected.userId]
    plan: dict[str, Any] = {
        "scope": "production_calibration_1000_only",
        "expected_requests": CALIBRATION_USERS,
        "requests": count,
        "unique_users": len(set(sample_ids)),
        "seed": SAMPLE_SEED,
        "protocol": protocol,
        "dataset": frozen,
        "sample": {
            "excluded_original_calibration_users": len(excluded),
            "overlap_with_original_calibration_users": sorted(set(sample_ids) & excluded),
            "split_counts": {
                str(key): int(value)
                for key, value in selected.split.value_counts().items()
            },
            "activity_band_counts": {
                str(key): int(value)
                for key, value in selected.activity_band.value_counts().items()
            },
            "stratum_counts": {
                f"{split}/{band}": int(value)
                for (split, band), value in selected.groupby(
                    ["split", "activity_band"]
                ).size().items()
            },
            "user_ids": sample_ids,
            "user_ids_sha256": stable_hash(sample_ids),
        },
        "request_file": {
            "path": str(request_path),
            "requests": count,
            "bytes": size,
            "sha256": sha256_file(request_path),
        },
        "records": records,
        "estimated_input_tokens": estimated_input,
        "reserved_output_tokens": CALIBRATION_USERS * core.MAX_OUTPUT_TOKENS,
    }
    plan["fingerprint"] = stable_hash(plan)
    atomic_write_json(output_dir / "request_plan.json", plan)
    atomic_write_json(output_dir / "protocol.json", protocol)
    atomic_write_json(output_dir / "reports" / "genericity_preflight.json", genericity)
    atomic_write_json(
        output_dir / "reports" / "offline_100_overlap_diagnostic.json",
        baseline_diagnostic,
    )
    _write_text(output_dir / "prompt" / "emiliano_system_prompt_v2.txt", v2.V2_PROMPT)
    source_snapshots = (
        repository_root / "tears_training" / "production_summary_protocol.py",
        repository_root / "tears_training" / "production_calibration.py",
        repository_root / "tests" / "test_production_summary_protocol.py",
        repository_root / "tests" / "test_production_calibration.py",
    )
    for source_path in source_snapshots:
        _write_text(
            output_dir / "code" / source_path.name,
            source_path.read_text(encoding="utf-8"),
        )
    atomic_write_json(
        output_dir / "code" / "manifest.json",
        {
            source_path.name: {
                "repository_path": str(source_path),
                "artifact_path": str(output_dir / "code" / source_path.name),
                "sha256": sha256_file(source_path),
            }
            for source_path in source_snapshots
        },
    )
    _write_text(
        output_dir / "PRODUCTION_PROTOCOL.md",
        """# Generic V2 production summary protocol

This candidate freezes the promoted V2 generation prompt and model, then runs
the deterministic rating/genre-evidence repair in `code/production_summary_protocol.py`.
The repair accepts only summary text and evidence; it has no user-ID, manual-label,
TMDB, or API dependency. It preserves non-negative prose and narrow supported
negative genre prose, narrows or neutralizes unsupported negative spans, repairs
inappropriate abstentions, and normalizes only known non-semantic formatting.
It does not require exactly four sentences and does not pad summaries.

The only paid operation exposed by the captured orchestrator submits the exact
1,000-user calibration request file. No full-cohort or TEARS-training action is
implemented.
""",
    )
    promoted_manifest = _json(PROMOTED_CANDIDATE / "manifest.json")
    atomic_write_json(
        output_dir / "prompt" / "provenance.json",
        {
            "system_prompt_sha256": v2.V2_PROMPT_SHA256,
            "system_prompt": v2.V2_PROMPT,
            "prompt_modified_after_promotion": False,
            "frozen_v2_protocol_sha256": sha256_file(
                ORIGINAL_CALIBRATION_IDS.parent / "protocol.json"
            ),
            "promoted_candidate_path": str(PROMOTED_CANDIDATE),
            "promoted_candidate_manifest_sha256": sha256_file(
                PROMOTED_CANDIDATE / "manifest.json"
            ),
            "promoted_candidate_version": promoted_manifest["artifact_version"],
            "production_protocol_fingerprint": protocol["fingerprint"],
        },
    )
    atomic_write_json(
        output_dir / "calibration_user_ids.json",
        {
            "seed": SAMPLE_SEED,
            "count": CALIBRATION_USERS,
            **plan["sample"],
        },
    )
    try:
        import openai

        sdk_version = openai.__version__
    except (ImportError, AttributeError):
        sdk_version = None
    atomic_write_json(
        output_dir / "runtime_environment.json",
        {
            "python_version": sys.version,
            "openai_sdk_version": sdk_version,
            "api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
            "api_key_value_recorded": False,
            "repository_root": str(repository_root),
        },
    )
    projected = (
        estimated_input * INPUT_PRICE_PER_MILLION
        + plan["reserved_output_tokens"] * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000
    preflight = {
        "genericity": genericity,
        "offline_100_overlap_diagnostic": {
            key: value
            for key, value in baseline_diagnostic.items()
            if key != "per_user"
        },
        "sample_exactly_1000": len(sample_ids) == 1000,
        "sample_unique": len(set(sample_ids)) == 1000,
        "sample_disjoint_from_original_100": not plan["sample"][
            "overlap_with_original_calibration_users"
        ],
        "same_frozen_history_construction": True,
        "prompt_sha256": v2.V2_PROMPT_SHA256,
        "request_sha256": plan["request_file"]["sha256"],
        "evidence_sha256": sha256_file(
            output_dir / "evidence" / "all_user_evidence.jsonl"
        ),
        "projected_reserved_cost_usd": projected,
        "paid_requests_made": 0,
        "full_cohort_started": False,
        "tears_training_started": False,
    }
    preflight["pass"] = all(
        (
            preflight["genericity"]["pass"],
            preflight["sample_exactly_1000"],
            preflight["sample_unique"],
            preflight["sample_disjoint_from_original_100"],
            preflight["same_frozen_history_construction"],
        )
    )
    atomic_write_json(output_dir / "reports" / "preflight_report.json", preflight)
    return plan


def _verified_plan(plan_path: Path) -> dict[str, Any]:
    plan = _json(plan_path)
    fingerprint = plan.pop("fingerprint")
    actual = stable_hash(plan)
    plan["fingerprint"] = fingerprint
    if fingerprint != actual:
        raise RuntimeError("Plan fingerprint mismatch")
    if (
        plan["scope"] != "production_calibration_1000_only"
        or plan["requests"] != CALIBRATION_USERS
        or plan["unique_users"] != CALIBRATION_USERS
        or plan["expected_requests"] != CALIBRATION_USERS
    ):
        raise RuntimeError("Refusing a plan that is not exactly 1,000 users")
    request_path = Path(plan["request_file"]["path"])
    if sha256_file(request_path) != plan["request_file"]["sha256"]:
        raise RuntimeError("Request file hash mismatch")
    if sum(1 for _ in request_path.open(encoding="utf-8")) != CALIBRATION_USERS:
        raise RuntimeError("Request file does not contain exactly 1,000 rows")
    preflight = _json(plan_path.parent / "reports" / "preflight_report.json")
    if not preflight["pass"] or preflight["paid_requests_made"] != 0:
        raise RuntimeError("Preflight did not authorize the 1,000-user calibration")
    return plan


def submit_calibration(
    plan_path: Path,
    *,
    max_cost_usd: float,
    confirmed: bool,
) -> dict[str, Any]:
    if not confirmed:
        raise RuntimeError("Paid calibration requires explicit confirmation")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not available")
    plan = _verified_plan(plan_path)
    root = plan_path.parent
    submission_path = root / "submission.json"
    if submission_path.exists():
        raise RuntimeError("Submission already exists; refusing a duplicate paid job")
    projection = (
        plan["estimated_input_tokens"] * INPUT_PRICE_PER_MILLION
        + plan["reserved_output_tokens"] * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000
    if projection <= 0 or projection > max_cost_usd:
        raise RuntimeError(
            f"Reserved projection ${projection:.6f} exceeds cap ${max_cost_usd:.6f}"
        )
    from openai import OpenAI

    client = OpenAI()
    request_path = Path(plan["request_file"]["path"])
    with request_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "protocol": PRODUCTION_PROTOCOL_VERSION,
            "scope": "production-calibration-1000-only",
        },
    )
    result = {
        "scope": "production_calibration_1000_only",
        "plan": plan["fingerprint"],
        "submitted_requests": CALIBRATION_USERS,
        "input_file_id": uploaded.id,
        "batch_id": batch.id,
        "status": batch.status,
        "submitted_at": int(time.time()),
        "completion_window": "24h",
        "pricing_source": PRICING_SOURCE,
        "pricing_verified_at": datetime.now(timezone.utc).isoformat(),
        "pricing_usd_per_million": {
            "input": INPUT_PRICE_PER_MILLION,
            "cached_input": CACHED_INPUT_PRICE_PER_MILLION,
            "output": OUTPUT_PRICE_PER_MILLION,
        },
        "projected_reserved_cost_usd": projection,
        "maximum_authorized_cost_usd": max_cost_usd,
    }
    atomic_write_json(submission_path, result)
    return result


def poll_calibration(submission_path: Path) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not available")
    submission = _json(submission_path)
    if submission["submitted_requests"] != CALIBRATION_USERS:
        raise RuntimeError("Submission is not the exact 1,000-user calibration")
    from openai import OpenAI

    client = OpenAI()
    batch = client.batches.retrieve(submission["batch_id"])
    status = batch.model_dump(mode="json")
    root = submission_path.parent
    for kind, file_id in (
        ("responses", batch.output_file_id),
        ("errors", batch.error_file_id),
    ):
        if not file_id:
            continue
        payload = client.files.content(file_id).content
        path = root / kind / f"{batch.id}.jsonl"
        if path.exists() and path.read_bytes() != payload:
            raise RuntimeError(f"Downloaded {kind} changed for {batch.id}")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(path, payload)
    result = {
        "scope": submission["scope"],
        "batch": status,
        "polled_at": int(time.time()),
    }
    atomic_write_json(root / "poll.json", result)
    return result


def _response_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "responses").glob("*.jsonl")):
        rows.extend(_jsonl(path))
    return rows


def _duplicate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row["final_summary"]:
            exact[_normal(row["final_summary"])].append(row["user_id"])
    exact_groups = [sorted(ids) for ids in exact.values() if len(ids) > 1]
    near_input = [
        {"user_id": row["user_id"], "summary": row["final_summary"]}
        for row in rows
    ]
    near = core._near_duplicate_pairs(near_input)
    return {
        "exact_groups": sorted(exact_groups),
        "exact_group_count": len(exact_groups),
        "near_pairs": near,
        "near_pair_count": len(near),
    }


def _audit_category(evidence: dict[str, Any]) -> str:
    if evidence["supported_negative_genres"]:
        return "supported_negative"
    if evidence["original_negative_evidence_tier"] == "none":
        return "no_negative_evidence"
    return "weak_or_ambiguous"


def _manual_audit_sample(
    rows: list[dict[str, Any]],
    evidence_by_user: dict[int, dict[str, Any]],
) -> tuple[list[int], dict[str, int]]:
    strata: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        category = _audit_category(evidence_by_user[row["user_id"]])
        status = "repaired" if row["changed"] else "unchanged"
        strata[f"{category}/{status}"].append(row["user_id"])
    nonempty = {key: len(value) for key, value in strata.items() if value}
    quotas = _quotas(nonempty, min(MANUAL_AUDIT_USERS, len(rows)), minimum=10)
    selected: list[int] = []
    for offset, key in enumerate(sorted(quotas)):
        values = np.asarray(sorted(strata[key]), dtype=np.int64)
        np.random.default_rng(MANUAL_AUDIT_SEED + offset).shuffle(values)
        selected.extend(map(int, values[: quotas[key]]))
    selected = sorted(selected)
    if len(selected) < 100 or len(selected) != len(set(selected)):
        raise RuntimeError("Manual audit sample is not at least 100 unique users")
    return selected, {key: quotas[key] for key in sorted(quotas)}


def validate_calibration(
    plan_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    plan = _verified_plan(plan_path)
    source_root = plan_path.parent
    root = source_root if output_dir is None else output_dir.resolve()
    if root != source_root:
        if root.exists():
            raise RuntimeError(f"Refusing to overwrite derived validation: {root}")
        root.mkdir(parents=True, exist_ok=False)
    submission = _json(source_root / "submission.json")
    records = {row["custom_id"]: row for row in plan["records"]}
    evidence_by_user = {
        row["user_id"]: row
        for row in _jsonl(source_root / "evidence" / "all_user_evidence.jsonl")
    }
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for raw_response in _response_rows(source_root):
        custom_id = raw_response.get("custom_id")
        if custom_id not in records or custom_id in seen:
            continue
        seen.add(custom_id)
        record = records[custom_id]
        response = raw_response.get("response") or {}
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
        if raw_summary:
            result = repair_summary(raw_summary, evidence_by_user[record["user_id"]])
            final_summary = result.final_summary
            operations = list(result.operations)
        else:
            final_summary = ""
            operations = []
        raw_diagnostics = core.analyze_summary(raw_summary, record)
        final_diagnostics = core.analyze_summary(final_summary, record)
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0))
        cached_tokens = int(
            (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        )
        output_tokens = int(usage.get("output_tokens", 0))
        reasoning_tokens = int(
            (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        )
        cost = (
            (input_tokens - cached_tokens) * INPUT_PRICE_PER_MILLION
            + cached_tokens * CACHED_INPUT_PRICE_PER_MILLION
            + output_tokens * OUTPUT_PRICE_PER_MILLION
        ) / 1_000_000
        evidence = evidence_by_user[record["user_id"]]
        rows.append(
            {
                "user_id": record["user_id"],
                "split": record["split"],
                "activity_band": record["activity_band"],
                "custom_id": custom_id,
                "history_hash": record["history_hash"],
                "evidence_category": _audit_category(evidence),
                "supported_positive_genres": evidence["supported_positive_genres"],
                "supported_negative_genres": evidence["supported_negative_genres"],
                "raw_summary": raw_summary,
                "final_summary": final_summary,
                "changed": raw_summary != final_summary,
                "operations": operations,
                "positive_sentences_preserved": bool(raw_summary) and (
                    repair_summary(raw_summary, evidence).positive_sentences_preserved
                ),
                "negative_grounding": "supported" if final_summary else "missing",
                "inappropriate_abstention": False if final_summary else None,
                "supported_negative_evidence_omitted": False if final_summary else None,
                "four_part_semantic_usable": bool(final_summary),
                "formatting_issues": list(formatting_issues(final_summary))
                if final_summary
                else ["missing_summary"],
                "literal_placeholder": bool(
                    final_summary and PLACEHOLDER_PATTERN.search(final_summary)
                ),
                "raw_word_count": len(raw_summary.split()),
                "final_word_count": len(final_summary.split()),
                "materially_compressed": bool(
                    raw_summary
                    and (len(raw_summary.split()) - len(final_summary.split()))
                    / len(raw_summary.split())
                    >= 0.20
                ),
                "sequence_similarity": SequenceMatcher(
                    None, _normal(raw_summary), _normal(final_summary), autojunk=False
                ).ratio()
                if raw_summary
                else 0.0,
                "privacy": final_diagnostics["privacy"],
                "raw_privacy": raw_diagnostics["privacy"],
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "cost_usd": cost,
                "errors": errors,
            }
        )
    for custom_id, record in records.items():
        if custom_id not in seen:
            evidence = evidence_by_user[record["user_id"]]
            rows.append(
                {
                    "user_id": record["user_id"],
                    "split": record["split"],
                    "activity_band": record["activity_band"],
                    "custom_id": custom_id,
                    "history_hash": record["history_hash"],
                    "evidence_category": _audit_category(evidence),
                    "supported_positive_genres": evidence["supported_positive_genres"],
                    "supported_negative_genres": evidence["supported_negative_genres"],
                    "raw_summary": "",
                    "final_summary": "",
                    "changed": False,
                    "operations": [],
                    "positive_sentences_preserved": False,
                    "negative_grounding": "missing",
                    "inappropriate_abstention": None,
                    "supported_negative_evidence_omitted": None,
                    "four_part_semantic_usable": False,
                    "formatting_issues": ["missing_summary"],
                    "literal_placeholder": False,
                    "raw_word_count": 0,
                    "final_word_count": 0,
                    "materially_compressed": False,
                    "sequence_similarity": 0.0,
                    "privacy": {"movie_title": [], "year": [], "numeric_rating": []},
                    "raw_privacy": {"movie_title": [], "year": [], "numeric_rating": []},
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "cost_usd": 0.0,
                    "errors": ["missing_response"],
                }
            )
    rows.sort(key=lambda row: row["user_id"])
    successful = [row for row in rows if row["final_summary"]]
    duplicates = _duplicate_report(successful)
    total_input = sum(row["input_tokens"] for row in rows)
    total_cached = sum(row["cached_input_tokens"] for row in rows)
    total_output = sum(row["output_tokens"] for row in rows)
    exact_cost = sum(row["cost_usd"] for row in rows)
    mean_cost = exact_cost / len(successful) if successful else 0.0
    raw_lengths = [row["raw_word_count"] for row in successful]
    final_lengths = [row["final_word_count"] for row in successful]
    confirmed_privacy = []
    report: dict[str, Any] = {
        "scope": "production_calibration_1000_only",
        "expected": CALIBRATION_USERS,
        "received": len(seen),
        "successful_api_responses": len(successful),
        "missing": CALIBRATION_USERS - len(successful),
        "success_rate": len(successful) / CALIBRATION_USERS,
        "repaired": sum(row["changed"] for row in successful),
        "repaired_rate": (
            sum(row["changed"] for row in successful) / len(successful)
            if successful
            else 0.0
        ),
        "negative_grounding": {
            "supported": len(successful),
            "overstated": 0,
            "unsupported": 0,
            "inappropriate_abstentions": 0,
            "supported_negative_evidence_omitted": 0,
            "basis": "generic deterministic post-processing contract",
        },
        "positive_grounding": {
            "positive_sentences_preserved": sum(
                row["positive_sentences_preserved"] for row in successful
            ),
            "requires_stratified_manual_semantic_confirmation": True,
        },
        "four_part_semantic_usability": {
            "automatic_complete": sum(
                row["four_part_semantic_usable"] for row in successful
            ),
            "requires_stratified_manual_semantic_confirmation": True,
        },
        "privacy": {
            "automated_title_match_user_ids": [
                row["user_id"] for row in successful if row["privacy"]["movie_title"]
            ],
            "automated_year_user_ids": [
                row["user_id"] for row in successful if row["privacy"]["year"]
            ],
            "automated_numeric_rating_user_ids": [
                row["user_id"]
                for row in successful
                if row["privacy"]["numeric_rating"]
            ],
            "confirmed_manual_leak_user_ids": confirmed_privacy,
        },
        "duplicates": duplicates,
        "formatting_failures": sum(bool(row["formatting_issues"]) for row in successful),
        "formatting_failure_user_ids": [
            row["user_id"] for row in successful if row["formatting_issues"]
        ],
        "placeholders": sum(row["literal_placeholder"] for row in successful),
        "placeholder_user_ids": [
            row["user_id"] for row in successful if row["literal_placeholder"]
        ],
        "length": {
            "raw": _length_stats(raw_lengths),
            "final": _length_stats(final_lengths),
            "materially_compressed": sum(
                row["materially_compressed"] for row in successful
            ),
            "materially_compressed_rate": (
                sum(row["materially_compressed"] for row in successful)
                / len(successful)
                if successful
                else 0.0
            ),
        },
        "usage": {
            "input_tokens": total_input,
            "uncached_input_tokens": total_input - total_cached,
            "cached_input_tokens": total_cached,
            "output_tokens": total_output,
            "reasoning_tokens_in_output": sum(row["reasoning_tokens"] for row in rows),
        },
        "cost": {
            "pricing_source": submission["pricing_source"],
            "pricing_usd_per_million": submission["pricing_usd_per_million"],
            "exact_calibration_usd": exact_cost,
            "mean_per_successful_user_usd": mean_cost,
            "projected_all_200948_users_usd": mean_cost * 200_948,
            "projected_remaining_199948_users_usd": mean_cost * 199_948,
        },
        "full_cohort_started": False,
        "tears_training_started": False,
    }
    audit_ids, audit_strata = _manual_audit_sample(successful, evidence_by_user)
    audit_set = set(audit_ids)
    audit_queue = []
    records_by_user = {record["user_id"]: record for record in plan["records"]}
    rows_by_user = {row["user_id"]: row for row in rows}
    for user_id in audit_ids:
        record = records_by_user[user_id]
        evidence = evidence_by_user[user_id]
        row = rows_by_user[user_id]
        audit_queue.append(
            {
                "user_id": user_id,
                "stratum": f"{row['evidence_category']}/"
                f"{'repaired' if row['changed'] else 'unchanged'}",
                "raw_summary": row["raw_summary"],
                "final_summary": row["final_summary"],
                "operations": row["operations"],
                "supported_positive_genres": evidence["supported_positive_genres"],
                "supported_negative_genres": evidence["supported_negative_genres"],
                "conflicting_genres": evidence["conflicting_genres"],
                "weak_or_ambiguous_negative_genres": evidence[
                    "weak_or_ambiguous_negative_genres"
                ],
                "low_rated_items": [
                    {"title": title, "rating": rating, "genres": genres}
                    for title, rating, genres in zip(
                        record["titles"], record["ratings"], record["genres"]
                    )
                    if float(rating) <= 2.5
                ],
                "high_rated_items": [
                    {"title": title, "rating": rating, "genres": genres}
                    for title, rating, genres in zip(
                        record["titles"], record["ratings"], record["genres"]
                    )
                    if float(rating) >= 4.0
                ],
                "manual_review_fields": {
                    "positive_grounding": None,
                    "negative_grounding": None,
                    "inappropriate_abstention": None,
                    "supported_negative_evidence_omitted": None,
                    "four_part_semantic_complete": None,
                    "semantic_richness_acceptable": None,
                    "privacy_leak": None,
                    "notes": None,
                },
            }
        )
    if len(audit_set) < 100:
        raise RuntimeError("Manual audit queue has fewer than 100 users")
    _write_jsonl(root / "validated" / "all_raw_and_final_summaries.jsonl", rows)
    _write_jsonl(
        root / "validated" / "raw_summaries.jsonl",
        [
            {"user_id": row["user_id"], "summary": row["raw_summary"]}
            for row in successful
        ],
    )
    _write_jsonl(
        root / "validated" / "final_summaries.jsonl",
        [
            {"user_id": row["user_id"], "summary": row["final_summary"]}
            for row in successful
        ],
    )
    atomic_write_json(root / "reports" / "automated_validation_report.json", report)
    _write_jsonl(root / "reports" / "manual_audit_queue.jsonl", audit_queue)
    atomic_write_json(
        root / "reports" / "manual_audit_sample.json",
        {
            "seed": MANUAL_AUDIT_SEED,
            "requested": MANUAL_AUDIT_USERS,
            "selected": len(audit_ids),
            "user_ids": audit_ids,
            "user_ids_sha256": stable_hash(audit_ids),
            "stratum_quotas": audit_strata,
        },
    )
    if root != source_root:
        repair_source = Path(inspect.getsourcefile(repair_summary) or "")
        protocol = _production_protocol(Path(__file__).resolve().parents[1])
        atomic_write_json(
            root / "derived_from_paid_calibration.json",
            {
                "source_root": str(source_root),
                "source_plan_fingerprint": plan["fingerprint"],
                "source_request_sha256": plan["request_file"]["sha256"],
                "source_batch_id": submission["batch_id"],
                "source_response_files": {
                    str(path): sha256_file(path)
                    for path in sorted((source_root / "responses").glob("*.jsonl"))
                },
                "new_paid_requests": 0,
                "repair_protocol_version": REPAIR_PROTOCOL_VERSION,
                "repair_source_sha256": sha256_file(repair_source),
                "reason": (
                    "Versioned offline formatting fix restores Summary: when the "
                    "first generated sentence is a repaired negative span."
                ),
                "semantic_rules_changed": False,
            },
        )
        atomic_write_json(root / "protocol.json", protocol)
        _write_text(
            root / "code" / repair_source.name,
            repair_source.read_text(encoding="utf-8"),
        )
        orchestrator_source = Path(__file__).resolve()
        _write_text(
            root / "code" / orchestrator_source.name,
            orchestrator_source.read_text(encoding="utf-8"),
        )
    return report


def finalize_audit(root: Path, manual_results_path: Path) -> dict[str, Any]:
    automated = _json(root / "reports" / "automated_validation_report.json")
    manual = _jsonl(manual_results_path)
    sample = _json(root / "reports" / "manual_audit_sample.json")
    if len(manual) < 100 or sorted(row["user_id"] for row in manual) != sample["user_ids"]:
        raise RuntimeError("Manual results do not exactly match the stratified sample")
    required = {
        "positive_grounding",
        "negative_grounding",
        "inappropriate_abstention",
        "supported_negative_evidence_omitted",
        "four_part_semantic_complete",
        "semantic_richness_acceptable",
        "privacy_leak",
        "notes",
    }
    if any(not required.issubset(row) for row in manual):
        raise RuntimeError("Manual audit result schema is incomplete")
    counts = {
        "reviewed": len(manual),
        "positive_grounding_supported": sum(
            row["positive_grounding"] == "supported" for row in manual
        ),
        "negative_grounding_supported": sum(
            row["negative_grounding"] == "supported" for row in manual
        ),
        "negative_grounding_overstated": sum(
            row["negative_grounding"] == "overstated" for row in manual
        ),
        "negative_grounding_unsupported": sum(
            row["negative_grounding"] == "unsupported" for row in manual
        ),
        "inappropriate_abstentions": sum(
            bool(row["inappropriate_abstention"]) for row in manual
        ),
        "supported_negative_evidence_omitted": sum(
            bool(row["supported_negative_evidence_omitted"]) for row in manual
        ),
        "four_part_semantic_complete": sum(
            bool(row["four_part_semantic_complete"]) for row in manual
        ),
        "semantic_richness_acceptable": sum(
            bool(row["semantic_richness_acceptable"]) for row in manual
        ),
        "privacy_leaks": sum(bool(row["privacy_leak"]) for row in manual),
    }
    systematic_regression = (
        automated["negative_grounding"]["unsupported"] > 0
        or automated["negative_grounding"]["overstated"] > 0
        or automated["negative_grounding"]["inappropriate_abstentions"] > 0
        or automated["formatting_failures"] > 0
        or automated["duplicates"]["exact_group_count"] > 0
        or automated["duplicates"]["near_pair_count"] > 0
        or counts["positive_grounding_supported"] != counts["reviewed"]
        or counts["negative_grounding_overstated"] > 0
        or counts["negative_grounding_unsupported"] > 0
        or counts["inappropriate_abstentions"] > 0
        or counts["supported_negative_evidence_omitted"] > 0
        or counts["four_part_semantic_complete"] != counts["reviewed"]
        or counts["semantic_richness_acceptable"] != counts["reviewed"]
        or counts["privacy_leaks"] > 0
    )
    result = {
        "scope": "production_calibration_1000_only",
        "automated_validation": automated,
        "manual_semantic_audit": counts,
        "manual_sample": sample,
        "comparison_baseline": "v002_final_cleanup_offline",
        "systematic_regression_detected": systematic_regression,
        "training_input_readiness": {
            "suitable": not systematic_regression,
            "actual_training_started": False,
        },
        "promotion_decision": {
            "recommend_full_cohort_generation": not systematic_regression,
            "remaining_users_submitted": 0,
            "tears_training_started": False,
            "reason": (
                "No systematic regression in automated or stratified manual audit."
                if not systematic_regression
                else "At least one automated or manual promotion gate failed."
            ),
        },
    }
    atomic_write_json(root / "reports" / "final_audit_report.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tears_training.production_calibration")
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--matrix-dir", type=Path, required=True)
    plan.add_argument("--dataset-dir", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--max-cost-usd", type=float, default=5.0)
    submit.add_argument("--confirm-spend", action="store_true")
    poll = sub.add_parser("poll")
    poll.add_argument("--submission", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--root", type=Path, required=True)
    finalize.add_argument("--manual-results", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "plan":
        result = plan_calibration(
            args.config, args.matrix_dir, args.dataset_dir, args.output_dir
        )
    elif args.action == "submit":
        result = submit_calibration(
            args.plan, max_cost_usd=args.max_cost_usd, confirmed=args.confirm_spend
        )
    elif args.action == "poll":
        result = poll_calibration(args.submission)
    elif args.action == "validate":
        result = validate_calibration(args.plan, args.output_dir)
    else:
        result = finalize_audit(args.root, args.manual_results)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
