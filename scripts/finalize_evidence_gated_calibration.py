#!/usr/bin/env python3
"""Finalize the manually reviewed evidence-gated 1,000-user calibration.

This is reporting code only.  The user IDs below are audit annotations for the
frozen 120-profile manual sample; they are never imported by the production
evidence harness or used at inference time.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


RAW_V2_REPORT = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "production_calibration_1000/v008_20260816_v2_generic_repair_final/"
    "reports/final_audit_report.json"
)
DETERMINISTIC_FINAL_REPORT = Path(
    "artifacts/production_calibration_1000/"
    "final_production_protocol_20260816_frozen/final_audit_report.json"
)

# Manual adjudication after reading every profile and its separated evidence.
# These flag strict prompt-contract failures, not inference-time exceptions.
STRICT_POSITIVE_GENRE_GROUNDING_FAILURES = {
    5282,
    8742,
    13279,
    24796,
    26328,
    31715,
    64863,
    83310,
    98331,
    111189,
    114176,
    124152,
    157895,
    176397,
    198747,
}
MIXED_GENRE_CATEGORICAL_POSITIVE_FAILURES = {
    5282,
    26328,
    64863,
    83310,
    111189,
    198747,
}
SAMPLE_PARTIAL_NEGATIVE_OMISSIONS = {
    13279: ["Fantasy", "Mystery"],
    83310: ["Comedy"],
}

# All 27 lexical omission candidates were read.  Most were alias misses such
# as "romantic", "war-themed", or a genre named in a compound construction.
FULL_CONFIRMED_PARTIAL_NEGATIVE_OMISSIONS = {
    13279: ["Fantasy", "Mystery"],
    83310: ["Comedy"],
    124194: ["Romance"],
    189979: ["Mystery"],
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite a different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_json(path: Path, value: Any) -> None:
    write_once(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
    write_once(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    automated = read_json(root / "reports/automated_validation_report.json")
    preflight = read_json(root / "reports/preflight_report.json")
    protocol = read_json(root / "protocol.json")
    queue = read_jsonl(root / "reports/manual_audit_queue.jsonl")
    validated = read_jsonl(root / "validated/all_summaries.jsonl")
    if len(queue) != 120 or len(validated) != 1000:
        raise RuntimeError("Expected the frozen 120-profile audit and 1,000 validations")
    sample_ids = {int(row["user_id"]) for row in queue}
    annotation_ids = (
        STRICT_POSITIVE_GENRE_GROUNDING_FAILURES
        | MIXED_GENRE_CATEGORICAL_POSITIVE_FAILURES
        | set(SAMPLE_PARTIAL_NEGATIVE_OMISSIONS)
    )
    if not annotation_ids <= sample_ids:
        raise RuntimeError("A manual annotation is outside the frozen sample")

    manual_rows: list[dict[str, Any]] = []
    for row in queue:
        user_id = int(row["user_id"])
        status = row["negative_evidence_status"]
        positive_strict = user_id not in STRICT_POSITIVE_GENRE_GROUNDING_FAILURES
        omitted = SAMPLE_PARTIAL_NEGATIVE_OMISSIONS.get(user_id, [])
        manual_rows.append(
            {
                "user_id": user_id,
                "negative_evidence_status": status,
                "mixed_conflicting_genres": row["mixed_conflicting_genres"],
                "reviewed_against_separated_evidence": True,
                "unsupported_negative_claim": False,
                "overstated_negative_claim": False,
                "inappropriate_abstention": False,
                "supported_negative_evidence_omitted": omitted,
                "positive_preference_traceable_to_positive_examples": True,
                "strict_positive_genre_grounding_pass": positive_strict,
                "mixed_genre_categorical_positive_failure": (
                    user_id in MIXED_GENRE_CATEGORICAL_POSITIVE_FAILURES
                ),
                "substantive_positive_negative_contradiction": False,
                "four_part_semantic_usability": True,
                "semantic_completeness": not omitted,
                "semantic_richness_acceptable": True,
                "editable_tears_profile": True,
                "privacy_leak_after_cleanup": False,
                "adjudication_note": (
                    "Strict positive genre wording exceeded the deterministic "
                    "genre classification."
                    if not positive_strict
                    else "All inspected semantic claims satisfy the evidence contract."
                ),
            }
        )

    sample_statuses = Counter(row["negative_evidence_status"] for row in manual_rows)
    strict_positive_passes = sum(row["strict_positive_genre_grounding_pass"] for row in manual_rows)
    semantic_complete = sum(row["semantic_completeness"] for row in manual_rows)
    manual_report = {
        "reviewed": 120,
        "sample_seed": automated["manual_audit_sample"]["seed"],
        "sample_user_ids_sha256": hashlib.sha256(
            json.dumps(sorted(sample_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        "strata": dict(sample_statuses),
        "unsupported_negative_claims": 0,
        "overstated_negative_claims": 0,
        "inappropriate_abstentions": 0,
        "supported_negative_evidence_omitted_users": len(SAMPLE_PARTIAL_NEGATIVE_OMISSIONS),
        "supported_negative_evidence_omitted_user_ids": sorted(SAMPLE_PARTIAL_NEGATIVE_OMISSIONS),
        "positive_preference_traceable_to_positive_examples": 120,
        "strict_positive_genre_grounding_pass": strict_positive_passes,
        "strict_positive_genre_grounding_fail": 120 - strict_positive_passes,
        "mixed_genre_categorical_positive_failures": len(MIXED_GENRE_CATEGORICAL_POSITIVE_FAILURES),
        "mixed_genre_categorical_positive_failure_user_ids": sorted(
            MIXED_GENRE_CATEGORICAL_POSITIVE_FAILURES
        ),
        "substantive_positive_negative_contradictions": 0,
        "four_part_semantic_usability": 120,
        "semantic_completeness": semantic_complete,
        "semantic_incomplete": 120 - semantic_complete,
        "semantic_richness_acceptable": 120,
        "editable_tears_profile": 120,
        "privacy_leaks_after_cleanup": 0,
        "status_behavior": {
            "NONE_no_fabricated_dislike": sample_statuses["NONE"],
            "WEAK_stayed_narrow_and_qualified": sample_statuses["WEAK"],
            "STRONG_expressed_real_negative_pattern": sample_statuses["STRONG"],
        },
        "interpretation": (
            "Negative grounding passed in the manual sample, but the prompt did not "
            "reliably keep positive genre language within the deterministic genre "
            "classification and two STRONG profiles partially omitted supported genres."
        ),
    }

    raw_v2 = read_json(RAW_V2_REPORT)["automated_validation"]
    repaired_v2_manual = read_json(RAW_V2_REPORT)["manual_semantic_audit"]
    deterministic = read_json(DETERMINISTIC_FINAL_REPORT)
    comparison = {
        "paired_users": 1000,
        "raw_v2": {
            "mean_words": raw_v2["length"]["raw"]["mean"],
            "median_words": raw_v2["length"]["raw"]["median"],
            "p95_words": raw_v2["length"]["raw"]["p95_linear"],
            "semantic_repair_users": 819,
            "any_repair_users": raw_v2["repaired"],
            "manual_contradictions_per_120": 11,
            "confirmed_title_leaks": 4,
            "input_tokens": raw_v2["usage"]["input_tokens"],
            "output_tokens": raw_v2["usage"]["output_tokens"],
            "exact_calibration_cost_usd": raw_v2["cost"]["exact_calibration_usd"],
            "projected_full_cost_usd": raw_v2["cost"]["projected_all_200948_users_usd"],
        },
        "repaired_v2": {
            "mean_words": raw_v2["length"]["final"]["mean"],
            "median_words": raw_v2["length"]["final"]["median"],
            "p95_words": raw_v2["length"]["final"]["p95_linear"],
            "unsupported_negative_claims": 0,
            "overstated_negative_claims": 0,
            "manual_positive_grounding_per_120": repaired_v2_manual["positive_grounding_supported"],
            "manual_semantic_richness_per_120": repaired_v2_manual["semantic_richness_acceptable"],
            "semantic_post_generation_repair_users": 819,
        },
        "positive_llm_plus_deterministic_negative": {
            "mean_words": deterministic["length"]["mean"],
            "median_words": deterministic["length"]["median"],
            "p95_words": deterministic["length"]["p95_linear"],
            "unsupported_or_overstated_negative_claims": 0,
            "substantive_contradictions": 0,
            "llm_negative_content_discarded_users": deterministic["operations"][
                "discard_llm_negative_or_abstention_content"
            ]["users"],
            "positive_consistency_repair_users": deterministic["counts"][
                "positive_consistency_repair_users"
            ],
            "positive_fallback_users": deterministic["counts"]["positive_fallbacks"],
        },
        "evidence_gated_emiliano": {
            "mean_words": automated["length"]["final"]["mean"],
            "median_words": automated["length"]["final"]["median"],
            "p95_words": automated["length"]["final"]["p95_linear"],
            "semantic_post_generation_repair_users": 0,
            "privacy_or_format_cleanup_users": automated["cleanup"]["users"],
            "manual_unsupported_negative_claims_per_120": 0,
            "manual_overstated_negative_claims_per_120": 0,
            "manual_strict_positive_grounding_per_120": strict_positive_passes,
            "manual_semantic_richness_per_120": 120,
            "manual_partial_negative_omissions_per_120": len(SAMPLE_PARTIAL_NEGATIVE_OMISSIONS),
            "input_tokens": automated["usage"]["input_tokens"],
            "output_tokens": automated["usage"]["output_tokens"],
            "exact_calibration_cost_usd": automated["cost"]["exact_calibration_usd"],
            "projected_full_cost_usd": automated["cost"]["projected_full_200948_usd"],
        },
        "interpretation": (
            "The evidence-gated candidate restores and exceeds raw-V2 length while "
            "reducing cleanup from 86.7% to 0.7%, but it does not beat the frozen "
            "deterministic-negative protocol on strict genre consistency."
        ),
    }

    final_report = {
        "scope": {
            "users": 1000,
            "same_paired_users_as_previous_calibration": True,
            "full_cohort_started": False,
            "tears_training_started": False,
            "tmdb_semantic_metadata_used": False,
            "semantic_post_generation_repair_users": 0,
        },
        "protocol": {
            "version": protocol["version"],
            "model": protocol["model"],
            "prompt_sha256": protocol["system_prompt_sha256"],
            "exact_prompt_delta_file": "prompt/prompt_delta.json",
            "genericity_preflight": preflight["genericity"],
            "evidence_invariants": preflight["evidence_invariants"],
        },
        "evidence_distribution": {
            "counts": preflight["status_distribution"],
            "percent": preflight["status_percent"],
            "mixed_conflicting_genre_users": preflight["mixed_conflicting_genre_users"],
            "mixed_conflicting_genre_instances": preflight[
                "mixed_conflicting_genre_instances"
            ],
        },
        "automated_validation": automated,
        "corrected_evidence_contract_review": {
            "note": (
                "The original keyword screen is retained for provenance but is not an "
                "error count; it interpreted evidence caveats as negative claims."
            ),
            "NONE_no_fabricated_dislike": {"passed": 136, "total": 136},
            "WEAK_narrow_and_qualified": {"passed": 663, "total": 663},
            "STRONG_expressed_supported_negative_pattern": {"passed": 201, "total": 201},
            "confirmed_partial_supported_negative_omissions": {
                "users": len(FULL_CONFIRMED_PARTIAL_NEGATIVE_OMISSIONS),
                "of_strong_users": 201,
                "records": FULL_CONFIRMED_PARTIAL_NEGATIVE_OMISSIONS,
            },
        },
        "manual_semantic_audit": manual_report,
        "privacy_and_format_cleanup": {
            "total_users": 7,
            "percent": 0.7,
            "raw_title_copy_users": 4,
            "raw_year_leaks": 0,
            "raw_numeric_rating_leaks": 0,
            "prefix_or_whitespace_users": 3,
            "final_title_year_rating_leaks": 0,
            "final_formatting_failures": 0,
            "semantic_changes": 0,
        },
        "length": {
            **automated["length"],
            "raw_to_final_materially_compressed_users": 0,
            "mean_change_from_raw_v2_words": (
                automated["length"]["final"]["mean"] - raw_v2["length"]["raw"]["mean"]
            ),
            "mean_change_from_repaired_v2_words": (
                automated["length"]["final"]["mean"] - raw_v2["length"]["final"]["mean"]
            ),
        },
        "usage_and_cost": automated["usage"] | automated["cost"],
        "comparison_file": "reports/comparison_report.json",
        "training_input_readiness": {
            "structurally_suitable": True,
            "semantically_recommended": False,
            "reason": (
                "The text is rich and machine-readable, but strict positive genre "
                "grounding and partial supported-negative omission regressions remain."
            ),
        },
        "ux_profile_readiness": {
            "expressive_and_editable_in_manual_sample": 120,
            "actual_controllability_tested": False,
            "recommended_for_final_ux_study": False,
            "reason": (
                "Text richness is excellent, but some genre-polarity claims do not "
                "strictly follow the precomputed evidence classes."
            ),
        },
        "promotion_decision": {
            "recommend_full_cohort_generation": False,
            "systematic_regression_relative_to_frozen_deterministic_negative_protocol": True,
            "failed_gates": [
                "15/120 profiles fail strict positive genre grounding",
                "6/49 mixed-genre sample profiles use categorical positive wording",
                "4/201 STRONG users partially omit a supported negative genre",
            ],
            "remaining_users_submitted": 0,
            "tears_training_started": False,
        },
    }

    cost_plan = {
        "model": protocol["model"],
        "pricing_usd_per_million_tokens": {
            "input": 0.25,
            "cached_input": 0.025,
            "output": 2.0,
        },
        "calibration_users": 1000,
        "calibration_input_tokens": automated["usage"]["input_tokens"],
        "calibration_output_tokens": automated["usage"]["output_tokens"],
        "exact_calibration_cost_usd": automated["cost"]["exact_calibration_usd"],
        "full_cohort_users": 200_948,
        "projected_full_cohort_input_tokens": round(
            automated["usage"]["input_tokens"] / 1000 * 200_948
        ),
        "projected_full_cohort_output_tokens": round(
            automated["usage"]["output_tokens"] / 1000 * 200_948
        ),
        "projected_full_cohort_cost_usd": automated["cost"]["projected_full_200948_usd"],
        "submission_authorized": False,
        "full_cohort_started": False,
    }

    write_jsonl(root / "reports/manual_semantic_audit.jsonl", manual_rows)
    write_json(root / "reports/manual_semantic_audit_report.json", manual_report)
    write_json(root / "reports/comparison_report.json", comparison)
    write_json(root / "reports/full_cohort_cost_plan.json", cost_plan)
    write_json(root / "reports/final_audit_report.json", final_report)
    print(json.dumps(final_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
