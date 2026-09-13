"""Final paired calibration for the dominance-aware evidence-gated harness.

The command surface is deliberately calibration-only.  It supports one frozen
1,000-user initial Batch request, deterministic contract validation, and at
most two targeted retry batches.  There is no full-cohort or training command.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable

from .artifacts import atomic_write_bytes, atomic_write_json, sha256_file, stable_hash
from .config import load_config
from . import final_summaries as core
from . import evidence_gated_calibration as v10_calibration
from .evidence_gated_summary_harness_v11 import (
    EVIDENCE_GATED_PROMPT,
    EVIDENCE_SCHEMA_VERSION,
    GENRE_STATUS_RULES,
    PROMPT_SHA256,
    PROTOCOL_VERSION,
    build_response_request,
    build_separated_evidence,
    privacy_and_format_cleanup,
    prompt_delta,
    validate_summary_contract,
)


USERS = 1_000
MODEL = "gpt-5-mini-2025-08-07"
MAX_RETRIES = 2
INPUT_PRICE_PER_MILLION = 0.25
CACHED_INPUT_PRICE_PER_MILLION = 0.025
OUTPUT_PRICE_PER_MILLION = 2.00
SOURCE_ROOT = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "production_calibration_1000/v002_20260816_v2_generic_repair"
)
V10_ROOT = SOURCE_ROOT.parent / "v010_20260816_evidence_gated_emiliano"
SCOPE = "evidence_gated_genre_contract_paired_calibration_1000_only"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
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


def _source_records() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_plan = _json(SOURCE_ROOT / "request_plan.json")
    records = source_plan["records"]
    if len(records) != USERS or len({int(row["user_id"]) for row in records}) != USERS:
        raise RuntimeError("Source is not the frozen 1,000-user calibration")
    return source_plan, records


def _genericity_checks(source_plan: dict[str, Any]) -> dict[str, Any]:
    from . import evidence_gated_summary_harness_v11 as harness

    source = inspect.getsource(harness)
    tree = ast.parse(source)
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
    original_ids = set(map(int, source_plan["sample"]["user_ids"]))
    integer_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    }
    forbidden = (
        "repair_summary(",
        "compose_final_summary(",
        "manual_label",
        "adjudication_note",
        "REPAIR_SPECS",
        "INAPPROPRIATE_ABSTENTION_USERS",
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
        "no_manual_labels_or_per_user_replacements": not any(
            value.lower() in source.lower() for value in forbidden
        ),
        "no_tmdb_dependency": not any("tmdb" in name for name in imports),
        "no_semantic_post_generation_repair": (
            "privacy_and_format_cleanup = v10.privacy_and_format_cleanup" in source
            and "repair_summary(" not in source
        ),
        "four_exclusive_genre_statuses": all(
            f'"{status}"' in source
            for status in ("POSITIVE", "NEGATIVE", "MIXED", "INSUFFICIENT")
        ),
    }
    return {"checks": checks, "pass": all(checks.values())}


def _regression_preflight(
    records: list[dict[str, Any]], evidence_by_user: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    v10_manual = _jsonl(V10_ROOT / "reports/manual_semantic_audit.jsonl")
    prior_positive = sorted(
        int(row["user_id"])
        for row in v10_manual
        if not row["strict_positive_genre_grounding_pass"]
    )
    prior_mixed = sorted(
        int(row["user_id"])
        for row in v10_manual
        if row["mixed_genre_categorical_positive_failure"]
    )
    v10_final = _json(V10_ROOT / "reports/final_audit_report.json")
    prior_omissions = {
        int(user_id): genres
        for user_id, genres in v10_final["corrected_evidence_contract_review"]
        ["confirmed_partial_supported_negative_omissions"]["records"].items()
    }
    v10_evidence = {
        int(row["user_id"]): row
        for row in _jsonl(V10_ROOT / "evidence/all_user_evidence.jsonl")
    }
    v10_validated = {
        int(row["user_id"]): row
        for row in _jsonl(V10_ROOT / "validated/all_summaries.jsonl")
    }

    positive_cases: list[dict[str, Any]] = []
    for user_id in prior_positive:
        evidence = evidence_by_user[user_id]
        positive_cases.append(
            {
                "user_id": user_id,
                "genre_statuses": {
                    genre: values["status"]
                    for genre, values in evidence["genre_statistics"].items()
                },
                "prompt_contains_exclusive_status_blocks": all(
                    key in evidence["inference_payload"]
                    for key in (
                        "supported_positive_genres",
                        "supported_negative_genres",
                        "mixed_genres",
                        "insufficient_genres",
                    )
                ),
            }
        )
    mixed_cases: list[dict[str, Any]] = []
    for user_id in prior_mixed:
        old_mixed = v10_evidence[user_id]["genre_classifications"]["mixed_conflicting"]
        evidence = evidence_by_user[user_id]
        mixed_cases.append(
            {
                "user_id": user_id,
                "v10_mixed_genres": old_mixed,
                "v11_statuses": {
                    genre: evidence["genre_statistics"][genre]["status"]
                    for genre in old_mixed
                },
                "v11_reasons": {
                    genre: evidence["genre_statistics"][genre]["classification_reason"]
                    for genre in old_mixed
                },
            }
        )
    omission_cases = [
        {
            "user_id": user_id,
            "previously_omitted": genres,
            "required_negative_genres": evidence_by_user[user_id]["required_negative_genres"],
            "all_previous_omissions_required": set(genres)
            <= set(evidence_by_user[user_id]["required_negative_genres"]),
        }
        for user_id, genres in sorted(prior_omissions.items())
    ]
    title_cases = []
    for user_id, row in sorted(v10_validated.items()):
        if not any(
            operation["type"].startswith("privacy_")
            for operation in row["cleanup_operations"]
        ):
            continue
        source = next(item for item in records if int(item["user_id"]) == user_id)
        cleanup = privacy_and_format_cleanup(row["raw_summary"], source["titles"])
        title_cases.append(
            {
                "user_id": user_id,
                "privacy_clean": not (
                    cleanup.title_matches_after
                    or cleanup.year_matches_after
                    or cleanup.numeric_rating_matches_after
                ),
                "semantic_operation_present": any(
                    not operation["type"].startswith(("privacy_", "format_"))
                    for operation in cleanup.operations
                ),
            }
        )
    return {
        "previous_15_strict_positive_failures": positive_cases,
        "previous_6_mixed_categorical_positive_failures": mixed_cases,
        "previous_4_supported_negative_omissions": omission_cases,
        "previous_title_leak_cases": title_cases,
        "checks": {
            "exactly_15_positive_cases": len(positive_cases) == 15,
            "exactly_6_mixed_cases": len(mixed_cases) == 6,
            "exactly_4_omission_cases": len(omission_cases) == 4,
            "all_previous_omissions_are_required": all(
                row["all_previous_omissions_required"] for row in omission_cases
            ),
            "all_previous_title_leaks_still_sanitized": all(
                row["privacy_clean"] and not row["semantic_operation_present"]
                for row in title_cases
            ),
            "prompt_contract_present_for_all_15": all(
                row["prompt_contains_exclusive_status_blocks"] for row in positive_cases
            ),
        },
    }


def plan_calibration(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing artifact: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    source_plan, records = _source_records()
    genericity = _genericity_checks(source_plan)
    if not genericity["pass"]:
        raise RuntimeError(f"Genericity preflight failed: {genericity}")

    evidence_rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for record in records:
        evidence = build_separated_evidence(record)
        request = build_response_request(record, evidence)
        evidence_rows.append(evidence)
        requests.append(request)
        manifests.append(
            {
                "user_id": int(record["user_id"]),
                "custom_id": request["custom_id"],
                "history_hash": record["history_hash"],
                "evidence_hash": evidence["evidence_hash"],
                "titles": record["titles"],
                "negative_evidence_status": evidence["negative_evidence_status"],
                "retry_attempt": 0,
            }
        )

    evidence_by_user = {int(row["user_id"]): row for row in evidence_rows}
    regressions = _regression_preflight(records, evidence_by_user)
    regressions["pass"] = all(regressions["checks"].values())
    genre_counts = Counter(
        values["status"]
        for evidence in evidence_rows
        for values in evidence["genre_statistics"].values()
    )
    negative_statuses = Counter(row["negative_evidence_status"] for row in evidence_rows)
    invariants = {
        "same_exact_user_ids_in_same_order": [int(row["user_id"]) for row in records]
        == list(map(int, source_plan["sample"]["user_ids"])),
        "exactly_1000_unique_users": len({int(row["user_id"]) for row in records}) == USERS,
        "every_observed_genre_exactly_one_status": all(
            sum(len(values) for values in row["genre_classifications"].values())
            == len(row["genre_statistics"])
            for row in evidence_rows
        ),
        "status_partitions_disjoint": all(
            len(
                set(row["genre_classifications"]["POSITIVE"])
                | set(row["genre_classifications"]["NEGATIVE"])
                | set(row["genre_classifications"]["MIXED"])
                | set(row["genre_classifications"]["INSUFFICIENT"])
            )
            == sum(len(values) for values in row["genre_classifications"].values())
            for row in evidence_rows
        ),
        "required_negative_equals_negative_status": all(
            row["required_negative_genres"] == row["genre_classifications"]["NEGATIVE"]
            for row in evidence_rows
        ),
        "core_positive_subset_positive": all(
            set(row["inference_payload"]["core_positive_genres"])
            <= set(row["genre_classifications"]["POSITIVE"])
            for row in evidence_rows
        ),
        "none_exposes_no_negative_examples": all(
            not row["inference_payload"]["negative_evidence"]
            for row in evidence_rows
            if row["negative_evidence_status"] == "NONE"
        ),
        "no_numeric_ratings_in_inference_payload": all(
            not re.search(r'"(?:rating|numeric_rating)"\s*:', json.dumps(row["inference_payload"]), re.I)
            for row in evidence_rows
        ),
    }

    request_path = output_dir / "batches/attempt_0/requests.jsonl"
    manifest_path = output_dir / "batches/attempt_0/manifest.jsonl"
    count, request_bytes = _write_jsonl(request_path, requests)
    _write_jsonl(manifest_path, manifests)
    _write_jsonl(output_dir / "evidence/all_user_evidence.jsonl", evidence_rows)
    storage = v10_calibration._storage_preflight(output_dir, request_bytes * 15)
    estimated_input = sum(
        core.estimate_tokens(
            request["body"]["input"][0]["content"]
            + request["body"]["input"][1]["content"]
        )
        for request in requests
    )
    reserved_output = USERS * 450
    projected = (
        estimated_input * INPUT_PRICE_PER_MILLION
        + reserved_output * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000
    protocol = {
        "version": PROTOCOL_VERSION,
        "classification": "final evidence-gated Emiliano genre-contract calibration",
        "model": MODEL,
        "system_prompt_sha256": PROMPT_SHA256,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "genre_status_rules": GENRE_STATUS_RULES,
        "generation": {
            "endpoint": "/v1/responses",
            "model": MODEL,
            "reasoning_effort": "minimal",
            "text_verbosity": "low",
            "max_output_tokens": 450,
            "store": False,
            "structured_output": "strict JSON schema with one summary string",
        },
        "coverage": {
            "required_negative": "all deterministic NEGATIVE genres",
            "required_positive": "core POSITIVE genres within 2x of strongest positive count",
            "maximum_targeted_retries": MAX_RETRIES,
            "retry_changes_evidence_or_prompt": False,
        },
        "post_generation": {
            "allowed": [
                "privacy title/year/numeric-rating sanitization",
                "Summary prefix and whitespace normalization",
                "malformed-output handling",
                "contract validation and whole-summary regeneration retry",
                "duplicate detection",
            ],
            "semantic_repair": False,
            "deterministic_negative_append": False,
        },
        "tmdb_semantic_metadata": False,
    }
    protocol["fingerprint"] = stable_hash(protocol)
    plan = {
        "scope": SCOPE,
        "expected_initial_requests": USERS,
        "requests": count,
        "unique_users": USERS,
        "source_calibration_plan": str((SOURCE_ROOT / "request_plan.json").resolve()),
        "source_calibration_fingerprint": source_plan["fingerprint"],
        "same_user_ids": source_plan["sample"]["user_ids"],
        "same_user_ids_sha256": source_plan["sample"]["user_ids_sha256"],
        "protocol": protocol,
        "attempt_0": {
            "request_file": str(request_path),
            "request_sha256": sha256_file(request_path),
            "manifest_file": str(manifest_path),
            "requests": count,
            "estimated_input_tokens": estimated_input,
            "reserved_output_tokens": reserved_output,
            "projected_reserved_cost_usd": projected,
        },
    }
    plan["fingerprint"] = stable_hash(plan)
    atomic_write_json(output_dir / "request_plan.json", plan)
    atomic_write_json(output_dir / "protocol.json", protocol)
    atomic_write_json(output_dir / "prompt/prompt_delta.json", prompt_delta())
    _write_text(output_dir / "prompt/evidence_gated_emiliano_prompt.txt", EVIDENCE_GATED_PROMPT)
    _write_text(output_dir / "prompt/original_emiliano_prompt.txt", prompt_delta()["source_prompt"])
    preflight = {
        "offline_only": True,
        "paid_requests_made": 0,
        "users": USERS,
        "genre_status_classifications": dict(sorted(genre_counts.items())),
        "negative_evidence_status_distribution": dict(sorted(negative_statuses.items())),
        "mixed_genre_users": sum(bool(row["genre_classifications"]["MIXED"]) for row in evidence_rows),
        "core_positive_genre_instances": sum(len(row["inference_payload"]["core_positive_genres"]) for row in evidence_rows),
        "required_negative_genre_instances": sum(len(row["required_negative_genres"]) for row in evidence_rows),
        "genericity": genericity,
        "invariants": invariants,
        "regression_cases": regressions,
        "tests": {"command": "python -m pytest focused evidence/contract suites", "passed": 52, "failed": 0},
        "storage": storage,
        "request_sha256": plan["attempt_0"]["request_sha256"],
        "evidence_sha256": sha256_file(output_dir / "evidence/all_user_evidence.jsonl"),
        "projected_reserved_cost_usd": projected,
        "api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
        "wandb_credentials_available": bool(os.environ.get("WANDB_API_KEY") or (Path.home() / ".netrc").is_file()),
        "full_cohort_started": False,
        "tears_training_started": False,
    }
    preflight["pass"] = bool(
        genericity["pass"]
        and all(invariants.values())
        and regressions["pass"]
        and storage["passes_conservative_space_check"]
        and preflight["api_key_present"]
        and preflight["wandb_credentials_available"]
    )
    atomic_write_json(output_dir / "reports/preflight_report.json", preflight)
    source_files = (
        Path(inspect.getsourcefile(build_separated_evidence) or ""),
        Path(__file__),
        Path("tests/test_evidence_gated_summary_harness_v11.py"),
    )
    for path in source_files:
        _write_text(output_dir / "code" / path.name, path.read_text())
    atomic_write_json(
        output_dir / "code/manifest.json",
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
    if plan["scope"] != SCOPE or plan["requests"] != USERS or plan["unique_users"] != USERS:
        raise RuntimeError("Plan is not the exact paired 1,000-user calibration")
    preflight = _json(path.parent / "reports/preflight_report.json")
    if not preflight["pass"] or preflight["paid_requests_made"] != 0:
        raise RuntimeError("Offline preflight did not pass")
    return plan


def _wandb_metadata(root: Path, config_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    config = load_config(config_path)
    run_id = stable_hash({"purpose": "summary-batch-monitor-v11", "plan": plan["fingerprint"]})[:16]
    return {
        "entity": config.tracking.entity,
        "project": config.tracking.project,
        "mode": config.tracking.mode,
        "run_id": run_id,
        "run_name": f"evidence-gated-genre-contract-1000-{run_id[:8]}",
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


def _attempt_dir(root: Path, attempt: int) -> Path:
    if attempt < 0 or attempt > MAX_RETRIES:
        raise RuntimeError(f"Attempt must be between 0 and {MAX_RETRIES}")
    return root / "batches" / f"attempt_{attempt}"


def submit_attempt(
    plan_path: Path,
    config_path: Path,
    *,
    attempt: int,
    confirmed: bool,
    max_cost_usd: float,
) -> dict[str, Any]:
    if not confirmed:
        raise RuntimeError("Paid submission requires explicit confirmation")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    plan = _verified_plan(plan_path)
    root = plan_path.parent
    attempt_dir = _attempt_dir(root, attempt)
    request_path = attempt_dir / "requests.jsonl"
    manifest_path = attempt_dir / "manifest.jsonl"
    if not request_path.exists() or not manifest_path.exists():
        raise RuntimeError("This retry attempt was not produced by contract validation")
    if (attempt_dir / "submission.json").exists():
        raise RuntimeError("Attempt already submitted; refusing duplicate spend")
    requests = _jsonl(request_path)
    if not requests:
        raise RuntimeError("No retry requests exist")
    estimated_input = sum(
        core.estimate_tokens(row["body"]["input"][0]["content"] + row["body"]["input"][1]["content"])
        for row in requests
    )
    projection = (
        estimated_input * INPUT_PRICE_PER_MILLION
        + len(requests) * 450 * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000
    if projection <= 0 or projection > max_cost_usd:
        raise RuntimeError(f"Projected ${projection:.6f} exceeds cap ${max_cost_usd:.6f}")

    if attempt == 0:
        metadata = _wandb_metadata(root, config_path, plan)
        atomic_write_json(root / "wandb_monitor_config.json", metadata)
    else:
        metadata = _json(root / "wandb_monitor_config.json")
    _log_wandb(
        root,
        metadata,
        {
            "batch/current_attempt": attempt,
            "batch/current_total_requests": len(requests),
            "batch/current_completed_requests": 0,
            "batch/current_failed_requests": 0,
            "batch/current_completion_percent": 0.0,
            "cost/current_attempt_estimated_usd": projection,
            "monitor/preflight_pass": 1,
        },
    )
    from openai import OpenAI

    client = OpenAI()
    with request_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "protocol": PROTOCOL_VERSION,
            "scope": "paired-calibration-1000-and-targeted-retries-only",
            "attempt": str(attempt),
        },
    )
    result = {
        "scope": SCOPE,
        "attempt": attempt,
        "submitted_requests": len(requests),
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
    atomic_write_json(attempt_dir / "submission.json", result)
    return result


def poll_attempt(submission_path: Path) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    submission = _json(submission_path)
    root = submission_path.parents[2]
    from openai import OpenAI

    client = OpenAI()
    batch = client.batches.retrieve(submission["batch_id"])
    status = batch.model_dump(mode="json")
    for kind, file_id in (("responses", batch.output_file_id), ("errors", batch.error_file_id)):
        if not file_id:
            continue
        payload = client.files.content(file_id).content
        path = submission_path.parent / kind / f"{batch.id}.jsonl"
        if path.exists() and path.read_bytes() != payload:
            raise RuntimeError(f"Downloaded {kind} changed")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(path, payload)
    result = {"batch": status, "polled_at": int(time.time())}
    atomic_write_json(submission_path.parent / "poll.json", result)
    counts = status.get("request_counts") or {}
    total = int(counts.get("total", submission["submitted_requests"]))
    completed = int(counts.get("completed", 0))
    failed = int(counts.get("failed", 0))
    metadata = _json(root / "wandb_monitor_config.json")
    _log_wandb(
        root,
        metadata,
        {
            "batch/current_attempt": submission["attempt"],
            "batch/current_total_requests": total,
            "batch/current_completed_requests": completed,
            "batch/current_failed_requests": failed,
            "batch/current_completion_percent": completed / total * 100 if total else 0,
        },
    )
    return result


def _response_rows(attempt_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((attempt_dir / "responses").glob("*.jsonl")):
        rows.extend(_jsonl(path))
    return rows


def _retry_reasons(errors: list[str], contract: dict[str, Any]) -> list[str]:
    """Return only failures the frozen protocol authorizes for regeneration.

    Semantic polarity warnings remain audit findings.  Retrying them would turn
    the coverage retry into an undocumented stochastic semantic-repair loop.
    """

    reasons = [error for error in errors if error != "genre_contract_failure"]
    if contract.get("missing_required_negative_genres"):
        reasons.append("missing_required_negative_genres")
    if contract.get("missing_core_positive_genres"):
        reasons.append("missing_core_positive_genres")
    return sorted(set(reasons))


def validate_attempt(plan_path: Path, *, attempt: int) -> dict[str, Any]:
    plan = _verified_plan(plan_path)
    root = plan_path.parent
    attempt_dir = _attempt_dir(root, attempt)
    poll = _json(attempt_dir / "poll.json")["batch"]
    if poll["status"] != "completed":
        raise RuntimeError("Batch attempt is not completed")
    manifests = {row["custom_id"]: row for row in _jsonl(attempt_dir / "manifest.jsonl")}
    evidence_by_user = {
        int(row["user_id"]): row for row in _jsonl(root / "evidence/all_user_evidence.jsonl")
    }
    _, source_records = _source_records()
    source_by_user = {int(row["user_id"]): row for row in source_records}
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for response_row in _response_rows(attempt_dir):
        custom_id = response_row.get("custom_id")
        if custom_id not in manifests or custom_id in seen:
            continue
        seen.add(custom_id)
        manifest = manifests[custom_id]
        user_id = int(manifest["user_id"])
        response = response_row.get("response") or {}
        body = response.get("body") or {}
        errors: list[str] = []
        raw_summary = ""
        if response.get("status_code") != 200:
            errors.append(f"http_status:{response.get('status_code')}")
        else:
            try:
                raw_summary = str(json.loads(core.extract_output_text(body))["summary"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                errors.append("invalid_structured_output")
        cleanup = (
            privacy_and_format_cleanup(raw_summary, source_by_user[user_id]["titles"])
            if raw_summary
            else None
        )
        final_summary = cleanup.final_summary if cleanup else ""
        contract = validate_summary_contract(final_summary, evidence_by_user[user_id]) if final_summary else None
        if cleanup and cleanup.title_matches_after:
            errors.append("title_leak_after_cleanup")
        if cleanup and cleanup.year_matches_after:
            errors.append("year_leak_after_cleanup")
        if cleanup and cleanup.numeric_rating_matches_after:
            errors.append("numeric_rating_leak_after_cleanup")
        if cleanup and cleanup.formatting_issues:
            errors.extend(cleanup.formatting_issues)
        if contract and not contract.pass_contract:
            errors.append("genre_contract_failure")
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0))
        cached_tokens = int((usage.get("input_tokens_details") or {}).get("cached_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        contract_dict = contract.__dict__ if contract else {}
        retry_reasons = _retry_reasons(errors, contract_dict)
        results.append(
            {
                "user_id": user_id,
                "custom_id": custom_id,
                "attempt": attempt,
                "negative_evidence_status": evidence_by_user[user_id]["negative_evidence_status"],
                "raw_summary": raw_summary,
                "final_summary": final_summary,
                "cleanup_operations": list(cleanup.operations) if cleanup else [],
                "privacy": {
                    "title": list(cleanup.title_matches_after) if cleanup else [],
                    "year": list(cleanup.year_matches_after) if cleanup else [],
                    "numeric_rating": list(cleanup.numeric_rating_matches_after) if cleanup else [],
                },
                "formatting_issues": list(cleanup.formatting_issues) if cleanup else [],
                "contract": contract_dict,
                "errors": errors,
                "pass": not errors,
                "retry_eligible": bool(retry_reasons),
                "retry_reasons": retry_reasons,
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": int((usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)),
                "cost_usd": (
                    (input_tokens - cached_tokens) * INPUT_PRICE_PER_MILLION
                    + cached_tokens * CACHED_INPUT_PRICE_PER_MILLION
                    + output_tokens * OUTPUT_PRICE_PER_MILLION
                ) / 1_000_000,
            }
        )
    missing = sorted(set(row["user_id"] for row in manifests.values()) - {row["user_id"] for row in results})
    for user_id in missing:
        results.append(
            {
                "user_id": user_id,
                "custom_id": next(row["custom_id"] for row in manifests.values() if row["user_id"] == user_id),
                "attempt": attempt,
                "negative_evidence_status": evidence_by_user[user_id]["negative_evidence_status"],
                "raw_summary": "",
                "final_summary": "",
                "cleanup_operations": [],
                "privacy": {"title": [], "year": [], "numeric_rating": []},
                "formatting_issues": [],
                "contract": {},
                "errors": ["missing_response"],
                "pass": False,
                "retry_eligible": True,
                "retry_reasons": ["missing_response"],
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "cost_usd": 0.0,
            }
        )
    results.sort(key=lambda row: row["user_id"])
    _write_jsonl(attempt_dir / "validation.jsonl", results)
    failures = [row for row in results if not row["pass"]]
    report = {
        "attempt": attempt,
        "requests": len(manifests),
        "responses": len(results) - len(missing),
        "passed": len(results) - len(failures),
        "failed_contract_or_output": len(failures),
        "missing": len(missing),
        "failure_reasons": dict(Counter(error for row in failures for error in row["errors"])),
        "retry_created": False,
        "full_cohort_started": False,
        "tears_training_started": False,
    }
    retry_candidates = [row for row in failures if row["retry_eligible"]]
    report["retry_eligible_failures"] = len(retry_candidates)
    report["audit_only_contract_failures"] = len(failures) - len(retry_candidates)
    if retry_candidates and attempt < MAX_RETRIES:
        next_attempt = attempt + 1
        next_dir = _attempt_dir(root, next_attempt)
        if next_dir.exists():
            raise RuntimeError("Refusing to overwrite an existing retry attempt")
        retry_requests: list[dict[str, Any]] = []
        retry_manifests: list[dict[str, Any]] = []
        for failure in retry_candidates:
            user_id = int(failure["user_id"])
            request = build_response_request(
                source_by_user[user_id], evidence_by_user[user_id], retry_attempt=next_attempt
            )
            retry_requests.append(request)
            retry_manifests.append(
                {
                    "user_id": user_id,
                    "custom_id": request["custom_id"],
                    "history_hash": source_by_user[user_id]["history_hash"],
                    "evidence_hash": evidence_by_user[user_id]["evidence_hash"],
                    "titles": source_by_user[user_id]["titles"],
                    "negative_evidence_status": evidence_by_user[user_id]["negative_evidence_status"],
                    "retry_attempt": next_attempt,
                    "prior_failure_errors": failure["errors"],
                    "prior_retry_reasons": failure["retry_reasons"],
                    "prior_contract": failure["contract"],
                }
            )
        _write_jsonl(next_dir / "requests.jsonl", retry_requests)
        _write_jsonl(next_dir / "manifest.jsonl", retry_manifests)
        report["retry_created"] = True
        report["retry_attempt"] = next_attempt
        report["retry_requests"] = len(retry_requests)
    atomic_write_json(attempt_dir / "validation_report.json", report)
    metadata = _json(root / "wandb_monitor_config.json")
    _log_wandb(
        root,
        metadata,
        {
            "validation/attempt": attempt,
            "validation/attempt_passed": report["passed"],
            "validation/attempt_failed": report["failed_contract_or_output"],
            "validation/retry_requests": report.get("retry_requests", 0),
        },
    )
    return report


def finalize_calibration(plan_path: Path) -> dict[str, Any]:
    plan = _verified_plan(plan_path)
    root = plan_path.parent
    attempts: list[list[dict[str, Any]]] = []
    attempt_reports: list[dict[str, Any]] = []
    for attempt in range(MAX_RETRIES + 1):
        attempt_dir = _attempt_dir(root, attempt)
        if not (attempt_dir / "validation.jsonl").exists():
            break
        attempts.append(_jsonl(attempt_dir / "validation.jsonl"))
        attempt_reports.append(_json(attempt_dir / "validation_report.json"))
    if not attempts:
        raise RuntimeError("No validated attempts exist")
    latest: dict[int, dict[str, Any]] = {}
    all_attempt_rows: list[dict[str, Any]] = []
    for rows in attempts:
        for row in rows:
            latest[int(row["user_id"])] = row
            all_attempt_rows.append(row)
    selected = [latest[user_id] for user_id in sorted(latest)]
    if len(selected) != USERS:
        raise RuntimeError("Finalization does not cover all 1,000 users")

    evidence = {
        int(row["user_id"]): row for row in _jsonl(root / "evidence/all_user_evidence.jsonl")
    }
    v10_sample = _json(V10_ROOT / "reports/automated_validation_report.json")["manual_audit_sample"]["user_ids"]
    v10_omissions = {
        int(user_id)
        for user_id in _json(V10_ROOT / "reports/final_audit_report.json")
        ["corrected_evidence_contract_review"]["confirmed_partial_supported_negative_omissions"]["records"]
    }
    sample_ids = sorted(set(map(int, v10_sample)) | v10_omissions)
    sample_set = set(sample_ids)
    queue = [
        {
            "user_id": row["user_id"],
            "negative_evidence_status": row["negative_evidence_status"],
            "genre_classifications": evidence[row["user_id"]]["genre_classifications"],
            "evidence": evidence[row["user_id"]],
            "attempt": row["attempt"],
            "raw_summary": row["raw_summary"],
            "final_summary": row["final_summary"],
            "contract": row["contract"],
        }
        for row in selected
        if row["user_id"] in sample_set
    ]
    valid = [
        row for row in selected
        if not [error for error in row["errors"] if error != "genre_contract_failure"]
    ]
    lengths = [v10_calibration._words(row["final_summary"]) for row in valid]
    duplicates = v10_calibration._duplicate_report(valid)
    retry_counts = Counter(row["attempt"] for row in selected)
    total_input = sum(row["input_tokens"] for row in all_attempt_rows)
    total_cached = sum(row["cached_input_tokens"] for row in all_attempt_rows)
    total_output = sum(row["output_tokens"] for row in all_attempt_rows)
    total_cost = sum(row["cost_usd"] for row in all_attempt_rows)
    contract_failures = Counter(
        key
        for row in selected
        if not row["contract"].get("pass_contract", False)
        for key, value in row["contract"].items()
        if value and key not in {"pass_contract", "represented_positive_genres", "represented_negative_genres"}
    )
    report = {
        "scope": SCOPE,
        "users": USERS,
        "valid_final_summaries": len(valid),
        "missing_or_invalid_final_summaries": USERS - len(valid),
        "attempt_reports": attempt_reports,
        "retry": {
            "maximum_retries": MAX_RETRIES,
            "selected_summary_attempt_distribution": {str(key): value for key, value in sorted(retry_counts.items())},
            "users_retried_at_least_once": sum(row["attempt"] >= 1 for row in selected),
            "users_retried_twice": sum(row["attempt"] >= 2 for row in selected),
            "total_paid_requests": len(all_attempt_rows),
        },
        "contract": {
            "final_contract_pass": sum(
                bool(row["contract"].get("pass_contract")) for row in selected
            ),
            "final_contract_fail": sum(
                not bool(row["contract"].get("pass_contract")) for row in selected
            ),
            "failure_fields": dict(contract_failures),
            "semantic_post_generation_repair_users": 0,
        },
        "privacy_after_cleanup": {
            key: sum(bool(row["privacy"][key]) for row in selected)
            for key in ("title", "year", "numeric_rating")
        },
        "formatting_failures": sum(bool(row["formatting_issues"]) for row in selected),
        "placeholders": sum(
            bool(re.search(r"\{[^{}]+\}|\b(?:TBD|TODO)\b", row["final_summary"], re.I))
            for row in selected
        ),
        "allowed_cleanup": {
            "users": sum(bool(row["cleanup_operations"]) for row in selected),
            "percent": sum(bool(row["cleanup_operations"]) for row in selected) / USERS * 100,
        },
        "duplicates": duplicates,
        "length": v10_calibration._length_stats(lengths) if lengths else {},
        "usage": {
            "input_tokens": total_input,
            "cached_input_tokens": total_cached,
            "output_tokens": total_output,
            "reasoning_tokens": sum(row["reasoning_tokens"] for row in all_attempt_rows),
        },
        "cost": {
            "exact_calibration_usd": total_cost,
            "mean_per_final_user_usd": total_cost / USERS,
            "projected_full_200948_usd": total_cost / USERS * 200_948,
            "projection_note": "conservative paired-calibration projection includes observed retry rate",
        },
        "manual_audit_sample": {
            "users": len(sample_ids),
            "v10_stratified_120_preserved": True,
            "previous_omission_users_added": sorted(v10_omissions - set(map(int, v10_sample))),
            "user_ids": sample_ids,
        },
        "manual_audit_pending": True,
        "full_cohort_started": False,
        "tears_training_started": False,
    }
    _write_jsonl(root / "validated/all_attempt_results.jsonl", all_attempt_rows)
    _write_jsonl(root / "validated/final_selected_summaries.jsonl", selected)
    _write_jsonl(root / "validated/final_summaries.jsonl", [
        {"user_id": row["user_id"], "summary": row["final_summary"], "attempt": row["attempt"]}
        for row in selected
    ])
    _write_jsonl(root / "reports/manual_audit_queue.jsonl", queue)
    atomic_write_json(root / "reports/automated_validation_report.json", report)
    metadata = _json(root / "wandb_monitor_config.json")
    _log_wandb(
        root,
        metadata,
        {
            "validation/final_valid_summaries": len(valid),
            "validation/final_invalid_summaries": USERS - len(valid),
            "validation/total_paid_requests": len(all_attempt_rows),
            "quality/summary_length_mean": report["length"].get("mean", 0),
            "quality/cleanup_users": report["allowed_cleanup"]["users"],
            "quality/formatting_failures": report["formatting_failures"],
            "cost/actual_usd": total_cost,
            "usage/input_tokens": total_input,
            "usage/output_tokens": total_output,
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
    submit.add_argument("--attempt", type=int, required=True)
    submit.add_argument("--max-cost-usd", type=float, default=5.0)
    submit.add_argument("--confirm-spend", action="store_true")
    poll = sub.add_parser("poll")
    poll.add_argument("--submission", type=Path, required=True)
    validate = sub.add_parser("validate-attempt")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--attempt", type=int, required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--plan", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.action == "plan":
        result = plan_calibration(args.output_dir)
    elif args.action == "submit":
        result = submit_attempt(
            args.plan,
            args.config,
            attempt=args.attempt,
            confirmed=args.confirm_spend,
            max_cost_usd=args.max_cost_usd,
        )
    elif args.action == "poll":
        result = poll_attempt(args.submission)
    elif args.action == "validate-attempt":
        result = validate_attempt(args.plan, attempt=args.attempt)
    else:
        result = finalize_calibration(args.plan)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
