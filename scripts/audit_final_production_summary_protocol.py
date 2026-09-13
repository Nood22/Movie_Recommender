#!/usr/bin/env python3
"""Apply and freeze the final summary architecture on the paid 1,000-user run.

This is a deterministic offline composition/audit only. It makes no API calls,
uses no TMDB data, and never writes into the frozen calibration directories.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import inspect
import json
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

from tears_training import final_production_summary_protocol as final
from tears_training import final_summaries_v4 as evidence_protocol
from tears_training import production_summary_protocol as legacy


CALIBRATION_ROOT = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "production_calibration_1000"
)
SOURCE = CALIBRATION_ROOT / "v002_20260816_v2_generic_repair"
VALIDATED = CALIBRATION_ROOT / "v008_20260816_v2_generic_repair_final"
OUTPUT = Path(
    "artifacts/production_calibration_1000/"
    "final_production_protocol_20260816_frozen"
)
FULL_COHORT_ROOT = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "full_cohort/final_positive_deterministic_negative_20260816"
)
USERS = 1_000
FULL_USERS = 200_948
MODEL = "gpt-5-mini-2025-08-07"
INPUT_PRICE_PER_MILLION = 0.25
CACHED_INPUT_PRICE_PER_MILLION = 0.025
OUTPUT_PRICE_PER_MILLION = 2.00
MAX_OUTPUT_TOKENS = 450


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_json(path: Path, value: Any) -> None:
    write_immutable(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(stable_json(row) + "\n" for row in rows).encode()
    write_immutable(path, payload)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def words(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))


def percentile(values: list[int], p: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    fraction = rank - low
    return ordered[low] + fraction * (ordered[high] - ordered[low])


def length_stats(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "p95_linear": percentile(values, 0.95),
        "below_120": sum(value < 120 for value in values),
        "below_150": sum(value < 150 for value in values),
    }


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def duplicate_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_user = {int(row["user_id"]): row for row in rows}
    by_normal = defaultdict(list)
    for row in rows:
        by_normal[normalized(row["final_summary"])].append(int(row["user_id"]))
    exact_groups = [ids for ids in by_normal.values() if len(ids) > 1]
    near: list[dict[str, Any]] = []
    prepared = [
        (
            int(row["user_id"]),
            normalized(row["final_summary"]),
            set(normalized(row["final_summary"]).split()),
        )
        for row in rows
    ]
    for left_index, (left_id, left, left_words) in enumerate(prepared):
        for right_id, right, right_words in prepared[left_index + 1 :]:
            length_ratio = min(len(left), len(right)) / max(len(left), len(right))
            if length_ratio < 0.88:
                continue
            union = left_words | right_words
            jaccard = len(left_words & right_words) / len(union) if union else 1.0
            if jaccard < 0.72:
                continue
            similarity = SequenceMatcher(None, left, right).ratio()
            if similarity >= 0.90:
                near.append(
                    {
                        "left_user_id": left_id,
                        "right_user_id": right_id,
                        "sequence_similarity": similarity,
                    }
                )
    adjudicated: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for pair in near:
        left = by_user[pair["left_user_id"]]
        right = by_user[pair["right_user_id"]]
        justified = bool(
            left["positive_fallback"]
            and right["positive_fallback"]
            and left["supported_negative_genres"]
            != right["supported_negative_genres"]
            and left["final_summary"] != right["final_summary"]
        )
        record = pair | {
            "justified_deterministic_template_similarity": justified,
            "rationale": (
                "Both profiles appropriately use the same deterministic positive "
                "abstention, while their evidence-specific negative genre lists differ."
                if justified
                else "Not explained by the deterministic fallback template."
            ),
        }
        adjudicated.append(record)
        if not justified:
            unresolved.append(record)
    return {
        "exact_duplicate_groups": exact_groups,
        "exact_duplicate_pairs_or_groups": len(exact_groups),
        "near_duplicate_pairs": adjudicated,
        "near_duplicate_count": len(near),
        "justified_near_duplicate_count": len(adjudicated) - len(unresolved),
        "unresolved_near_duplicate_count": len(unresolved),
        "near_duplicate_threshold": 0.90,
    }


def estimate_tokens(text: str) -> int:
    # Same immutable planning heuristic used by the existing project.
    return max(1, int(len(text) / 4 * 1.20))


def build_cost_plan(
    source_rows: list[dict[str, Any]],
    composition_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    requests = read_jsonl(SOURCE / "requests/calibration-1000.jsonl")
    old_report = json.loads(
        (VALIDATED / "reports/automated_validation_report.json").read_text()
    )
    actual_input = int(old_report["usage"]["input_tokens"])
    actual_output = int(old_report["usage"]["output_tokens"])
    actual_cost = float(old_report["cost"]["exact_calibration_usd"])
    old_input_proxy = 0
    new_input_proxy = 0
    for request in requests:
        messages = request["body"]["input"]
        old_system = messages[0]["content"]
        user_content = messages[1]["content"]
        old_input_proxy += estimate_tokens(old_system + user_content)
        new_input_proxy += estimate_tokens(
            final.FINAL_POSITIVE_SYSTEM_PROMPT + user_content
        )
    input_calibration_factor = actual_input / old_input_proxy
    expected_new_input_1000 = new_input_proxy * input_calibration_factor

    old_output_proxy = sum(
        estimate_tokens(row["raw_summary"] + '{"summary":""}')
        for row in source_rows
    )
    new_output_proxy = sum(
        estimate_tokens(row["extracted_positive_prose"] + '{"summary":""}')
        for row in composition_rows
    )
    output_calibration_factor = actual_output / old_output_proxy
    expected_new_output_1000 = new_output_proxy * output_calibration_factor

    projected_input = expected_new_input_1000 / USERS * FULL_USERS
    projected_output = expected_new_output_1000 / USERS * FULL_USERS
    expected_cost = (
        projected_input * INPUT_PRICE_PER_MILLION / 1_000_000
        + projected_output * OUTPUT_PRICE_PER_MILLION / 1_000_000
    )
    reserved_ceiling = (
        projected_input * INPUT_PRICE_PER_MILLION / 1_000_000
        + FULL_USERS * MAX_OUTPUT_TOKENS * OUTPUT_PRICE_PER_MILLION / 1_000_000
    )
    observed_old_projection = actual_cost / USERS * FULL_USERS
    return {
        "pricing_basis": {
            "source": old_report["cost"]["pricing_source"],
            "batch_usd_per_million_tokens": {
                "uncached_input": INPUT_PRICE_PER_MILLION,
                "cached_input": CACHED_INPUT_PRICE_PER_MILLION,
                "output": OUTPUT_PRICE_PER_MILLION,
            },
            "must_reverify_immediately_before_submission": True,
        },
        "paid_calibration_observed": {
            "users": USERS,
            "input_tokens": actual_input,
            "output_tokens": actual_output,
            "exact_cost_usd": actual_cost,
            "old_v2_projected_full_cohort_cost_usd": observed_old_projection,
        },
        "final_positive_only_projection": {
            "method": (
                "Apply the paid calibration's actual/proxy token ratios to the "
                "same 1,000 histories with the frozen positive-only prompt and "
                "offline-extracted positive response lengths."
            ),
            "estimated_input_tokens_1000": expected_new_input_1000,
            "estimated_output_tokens_1000": expected_new_output_1000,
            "projected_input_tokens_200948": projected_input,
            "projected_output_tokens_200948": projected_output,
            "projected_actual_cost_usd": expected_cost,
            "maximum_output_tokens_per_request": MAX_OUTPUT_TOKENS,
            "conservative_reserved_cost_ceiling_usd": reserved_ceiling,
            "cost_formula": (
                "uncached_input_tokens * $0.25/M + cached_input_tokens * "
                "$0.025/M + output_tokens * $2.00/M"
            ),
            "not_exact_until_api_usage_is_returned": True,
        },
    }


def main() -> None:
    source_rows = read_jsonl(
        VALIDATED / "validated/all_raw_and_final_summaries.jsonl"
    )
    evidence_rows = read_jsonl(SOURCE / "evidence/all_user_evidence.jsonl")
    evidence = {int(row["user_id"]): row for row in evidence_rows}
    plan = json.loads((SOURCE / "request_plan.json").read_text())
    histories = {int(row["user_id"]): row for row in plan["records"]}
    if len(source_rows) != USERS or set(evidence) != set(histories):
        raise RuntimeError("Frozen 1,000-user inputs do not align")

    records: list[dict[str, Any]] = []
    operation_users: dict[str, set[int]] = defaultdict(set)
    operation_occurrences = Counter()
    invalid: list[dict[str, Any]] = []
    for source in source_rows:
        user_id = int(source["user_id"])
        try:
            result = final.compose_final_summary(
                source["raw_summary"], evidence[user_id], histories[user_id]["titles"]
            )
        except Exception as error:
            invalid.append({"user_id": user_id, "error": str(error)})
            continue
        for operation in result.operations:
            operation_occurrences[operation["type"]] += 1
            operation_users[operation["type"]].add(user_id)
        negative = final.deterministic_negative_sentence(
            evidence[user_id]["supported_negative_genres"]
        )
        positive_part = result.final_summary[len("Summary: ") :]
        if positive_part.endswith(negative):
            positive_part = positive_part[: -len(negative)].strip()
        unsupported_negative = int(
            result.final_summary.count(negative) != 1
            or any(
                legacy._is_user_negative_claim(sentence)
                for sentence in final._sentences(positive_part)
            )
        )
        inappropriate_abstention = int(
            bool(evidence[user_id]["supported_negative_genres"])
            and legacy._is_abstention(negative)
        )
        semantic_complete = bool(
            positive_part
            and negative
            and not result.title_matches_after
            and not result.year_leaks_after
            and not result.numeric_rating_leaks_after
            and not result.positive_negative_contradictions_after
            and not result.formatting_issues
        )
        records.append(
            {
                "user_id": user_id,
                "raw_v2_summary": source["raw_summary"],
                "extracted_positive_prose": result.extracted_positive_prose,
                "final_summary": result.final_summary,
                "final_word_count": words(result.final_summary),
                "supported_positive_genres": evidence[user_id][
                    "supported_positive_genres"
                ],
                "supported_negative_genres": evidence[user_id][
                    "supported_negative_genres"
                ],
                "deterministic_negative_sentence": negative,
                "operations": list(result.operations),
                "title_matches_before": list(result.title_matches_before),
                "title_matches_after": list(result.title_matches_after),
                "positive_negative_contradictions_before": list(
                    result.positive_negative_contradictions_before
                ),
                "positive_negative_contradictions_after": list(
                    result.positive_negative_contradictions_after
                ),
                "year_leaks_after": list(result.year_leaks_after),
                "numeric_rating_leaks_after": list(result.numeric_rating_leaks_after),
                "formatting_issues": list(result.formatting_issues),
                "unsupported_or_overstated_negative": bool(unsupported_negative),
                "inappropriate_abstention": bool(inappropriate_abstention),
                "semantic_complete": semantic_complete,
                "positive_fallback": any(
                    operation["type"] == "insert_grounded_positive_fallback"
                    for operation in result.operations
                ),
            }
        )

    if invalid or len(records) != USERS:
        raise RuntimeError(f"Composition failures: {invalid[:10]}")
    duplicates = duplicate_audit(records)
    lengths = [int(row["final_word_count"]) for row in records]
    cost = build_cost_plan(source_rows, records)

    counts = {
        "users": USERS,
        "missing_or_invalid_summaries": len(invalid),
        "unsupported_or_overstated_negative_claims": sum(
            row["unsupported_or_overstated_negative"] for row in records
        ),
        "inappropriate_abstentions": sum(
            row["inappropriate_abstention"] for row in records
        ),
        "substantive_positive_negative_contradictions": sum(
            bool(row["positive_negative_contradictions_after"]) for row in records
        ),
        "title_leaks": sum(bool(row["title_matches_after"]) for row in records),
        "year_leaks": sum(bool(row["year_leaks_after"]) for row in records),
        "numeric_rating_leaks": sum(
            bool(row["numeric_rating_leaks_after"]) for row in records
        ),
        "formatting_failures": sum(bool(row["formatting_issues"]) for row in records),
        "literal_placeholders": sum(
            bool(legacy.PLACEHOLDER_PATTERN.search(row["final_summary"]))
            for row in records
        ),
        "semantic_complete": sum(row["semantic_complete"] for row in records),
        "semantic_incomplete": sum(not row["semantic_complete"] for row in records),
        "positive_fallbacks": sum(row["positive_fallback"] for row in records),
        "title_sanitization_users": len(
            operation_users["privacy_remove_title_example_parenthetical"]
            | operation_users["privacy_remove_direct_title_copy"]
        ),
        "positive_consistency_repair_users": len(
            operation_users["consistency_remove_contradictory_positive_sentence"]
        ),
        "exact_duplicate_groups": duplicates["exact_duplicate_pairs_or_groups"],
        "near_duplicate_pairs": duplicates["near_duplicate_count"],
        "justified_near_duplicate_pairs": duplicates[
            "justified_near_duplicate_count"
        ],
        "unresolved_near_duplicate_pairs": duplicates[
            "unresolved_near_duplicate_count"
        ],
    }
    gates = {
        "unsupported_or_overstated_negative_claims_zero": counts[
            "unsupported_or_overstated_negative_claims"
        ]
        == 0,
        "inappropriate_abstentions_zero": counts["inappropriate_abstentions"] == 0,
        "privacy_leakage_zero": (
            counts["title_leaks"]
            + counts["year_leaks"]
            + counts["numeric_rating_leaks"]
            == 0
        ),
        "substantive_positive_negative_contradictions_zero": counts[
            "substantive_positive_negative_contradictions"
        ]
        == 0,
        "formatting_failures_zero": counts["formatting_failures"] == 0,
        "missing_or_invalid_zero": counts["missing_or_invalid_summaries"] == 0,
        "semantic_completeness_100_percent": counts["semantic_complete"] == USERS,
        "duplicates_clean": (
            counts["exact_duplicate_groups"] == 0
            and counts["unresolved_near_duplicate_pairs"] == 0
        ),
    }
    promoted = all(gates.values())

    module_path = Path(inspect.getsourcefile(final) or "")
    test_path = Path("tests/test_final_production_summary_protocol.py")
    protocol_manifest = {
        "status": "frozen_final_production_protocol" if promoted else "gate_failed",
        "protocol_version": final.PROTOCOL_VERSION,
        "architecture": {
            "llm_role": "positive preference prose only",
            "negative_role": "one deterministic evidence-derived genre sentence or abstention",
            "privacy": "generic history-title/year/numeric-rating sanitization",
            "consistency": "remove genuine broad positive claims colliding with supported negative genres",
            "prefix": "deterministic code-rendered Summary:",
            "tmdb_semantic_metadata": False,
            "manual_labels_at_inference": False,
            "user_specific_exceptions": False,
        },
        "model": MODEL,
        "positive_prompt": final.FINAL_POSITIVE_SYSTEM_PROMPT,
        "positive_prompt_sha256": text_sha256(final.FINAL_POSITIVE_SYSTEM_PROMPT),
        "structured_output": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
        "generation_settings": {
            "reasoning_effort": "minimal",
            "text_verbosity": "low",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
            "history_construction": plan["protocol"]["generation"]["history"],
        },
        "evidence_rules": evidence_protocol.EVIDENCE_RULES,
        "code": {
            "module_path": str(module_path.resolve()),
            "module_sha256": sha256(module_path),
            "tests_path": str(test_path.resolve()),
            "tests_sha256": sha256(test_path),
        },
        "offline_calibration": {
            "source_raw_artifact": str(VALIDATED.resolve()),
            "source_evidence_artifact": str(SOURCE.resolve()),
            "users": USERS,
            "gates": gates,
        },
    }
    protocol_manifest["fingerprint"] = text_sha256(stable_json(protocol_manifest))

    full_plan = {
        "status": "planned_not_submitted",
        "requests_submitted": 0,
        "users": FULL_USERS,
        "model": MODEL,
        "protocol_version": final.PROTOCOL_VERSION,
        "protocol_fingerprint": protocol_manifest["fingerprint"],
        "input": {
            "frozen_cohort_users": plan["dataset"]["users_path"],
            "frozen_observed_history_construction": plan["protocol"]["generation"][
                "history"
            ],
            "maximum_history_items": 50,
        },
        "execution": {
            "request_type": "OpenAI Batch Responses API",
            "shards": 5,
            "shard_sizes": [40_000, 40_000, 40_000, 40_000, 40_948],
            "standard_wandb_monitoring": True,
            "weave": False,
            "checkpoint_after_each_shard": True,
            "full_validation_before_training": True,
            "tears_training_started": False,
        },
        "paths": {
            "root": str(FULL_COHORT_ROOT),
            "request_manifests": str(FULL_COHORT_ROOT / "requests"),
            "raw_api_responses": str(FULL_COHORT_ROOT / "responses"),
            "raw_positive_summaries": str(
                FULL_COHORT_ROOT / "validated/raw_positive_summaries.jsonl"
            ),
            "final_summaries": str(
                FULL_COHORT_ROOT / "validated/final_summaries.jsonl"
            ),
            "composition_records": str(
                FULL_COHORT_ROOT / "validated/composition_records.jsonl"
            ),
            "audit_report": str(FULL_COHORT_ROOT / "reports/final_audit_report.json"),
            "provenance": str(FULL_COHORT_ROOT / "protocol.json"),
            "checkpoints": str(FULL_COHORT_ROOT / "checkpoints"),
        },
        "cost": cost,
        "pre_submission_requirements": [
            "reverify HOME/scratch capacity and quotas",
            "reverify current official model and Batch API pricing",
            "materialize and hash exactly 200,948 unique requests",
            "run genericity and no-user-exception tests",
            "obtain explicit user authorization for the paid submission",
        ],
    }

    audit = {
        "scope": {
            "offline_only": True,
            "openai_requests": 0,
            "new_summaries_generated": 0,
            "tmdb_used": False,
            "source_artifacts_modified": False,
        },
        "counts": counts,
        "length": length_stats(lengths),
        "operations": {
            key: {
                "users": len(operation_users[key]),
                "occurrences": operation_occurrences[key],
            }
            for key in sorted(operation_occurrences)
        },
        "duplicates": duplicates,
        "gates": gates,
        "promotion_decision": (
            "freeze_as_final_production_protocol" if promoted else "do_not_freeze"
        ),
        "protocol_fingerprint": protocol_manifest["fingerprint"],
        "cost_and_plan_file": "full_cohort_generation_plan.json",
    }

    final_rows = [
        {
            "user_id": row["user_id"],
            "summary": row["final_summary"],
            "word_count": row["final_word_count"],
        }
        for row in records
    ]
    write_jsonl(OUTPUT / "composition_records.jsonl", records)
    write_jsonl(OUTPUT / "final_summaries.jsonl", final_rows)
    write_json(OUTPUT / "final_audit_report.json", audit)
    write_json(OUTPUT / "protocol_manifest.json", protocol_manifest)
    write_json(OUTPUT / "full_cohort_generation_plan.json", full_plan)
    readme = (
        "# Frozen final production summary protocol\n\n"
        "Offline composition of the existing paid 1,000-user calibration. No API "
        "requests, new model summaries, TMDB semantics, source-artifact mutation, "
        "full-cohort submission, or TEARS training.\n\n"
        f"Promotion decision: `{audit['promotion_decision']}`.\n"
    )
    write_immutable(OUTPUT / "README.md", readme.encode())
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(cost, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
