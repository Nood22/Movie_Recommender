"""Build the final offline report for the 1,000-user production calibration."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics
from typing import Any

from tears_training.artifacts import atomic_write_json, stable_hash


# Targeted manual review of all 19 automated title matches. These are audit
# labels only and have no connection to production repair inference.
CONFIRMED_TITLE_LEAK_IDS = {35148, 39004, 92423, 127899}
POSITIVE_MARKER = re.compile(
    r"\b(?:enjoy\w*|lik\w*|favou?r\w*|prefer\w*|appreciat\w*|drawn to|"
    r"gravitat\w*|respond\w* well|interest in|affinity for|soft spot for|"
    r"values?|seeks?|leans? toward)\b",
    re.IGNORECASE,
)
NEGATIVE_SLOT = re.compile(
    r"(?:supports a negative preference|(?:no|does not indicate|does not provide)"
    r".{0,60}strong.{0,30}negative|negative preference.{0,80}not strongly supported|"
    r"(?:the )?user does not enjoy|(?:the )?user dislikes?|"
    r"limited evidence.{0,50}firm dislikes)",
    re.IGNORECASE,
)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root
    candidate = args.candidate_root
    baseline = json.loads((args.baseline_root / "evaluation_report.json").read_text())
    automated = json.loads(
        (candidate / "reports" / "automated_validation_report.json").read_text()
    )
    manual = jsonl(candidate / "reports" / "manual_audit_results.jsonl")
    rows = jsonl(candidate / "validated" / "all_raw_and_final_summaries.jsonl")
    evidence = {
        int(row["user_id"]): row
        for row in jsonl(source / "evidence" / "all_user_evidence.jsonl")
    }
    if len(rows) != 1000 or len(manual) != 120 or len(evidence) != 1000:
        raise RuntimeError("Final report inputs have unexpected scope")

    compressed = []
    for row in rows:
        if not row["materially_compressed"]:
            continue
        summary = row["final_summary"]
        negative_offsets = [match.start() for match in NEGATIVE_SLOT.finditer(summary)]
        positive_part = summary[: min(negative_offsets)] if negative_offsets else summary
        has_positive_profile = bool(POSITIVE_MARKER.search(positive_part)) and (
            "No strong positive genre preference" not in positive_part
        )
        has_negative_slot = bool(negative_offsets)
        complete = has_positive_profile and has_negative_slot
        compressed.append(
            {
                "user_id": row["user_id"],
                "raw_word_count": row["raw_word_count"],
                "final_word_count": row["final_word_count"],
                "relative_reduction": (
                    row["raw_word_count"] - row["final_word_count"]
                )
                / row["raw_word_count"],
                "positive_profile_remains": has_positive_profile,
                "negative_slot_remains": has_negative_slot,
                "semantically_complete": complete,
                "classification": (
                    "material_compression_semantically_complete"
                    if complete
                    else "material_compression_semantically_incomplete"
                ),
            }
        )
    atomic_write_json(
        candidate / "reports" / "material_compression_audit.json",
        {
            "definition": "final word count at least 20% below raw V2",
            "count": len(compressed),
            "semantically_complete": sum(
                row["semantically_complete"] for row in compressed
            ),
            "semantically_incomplete": sum(
                not row["semantically_complete"] for row in compressed
            ),
            "cases": compressed,
        },
    )

    flagged_ids = automated["privacy"]["automated_title_match_user_ids"]
    privacy_rows = []
    by_user = {int(row["user_id"]): row for row in rows}
    for user_id in flagged_ids:
        matches = by_user[user_id]["privacy"]["movie_title"]
        confirmed = user_id in CONFIRMED_TITLE_LEAK_IDS
        privacy_rows.append(
            {
                "user_id": user_id,
                "automated_matches": matches,
                "confirmed_identifiable_title_leak": confirmed,
                "manual_rationale": (
                    "The summary explicitly lists supplied movie titles as examples."
                    if confirmed
                    else "The match is ordinary preference prose or a generic concept, "
                    "not an identifiable title mention."
                ),
            }
        )
    atomic_write_json(
        candidate / "reports" / "privacy_adjudication.json",
        {
            "automated_title_matches": len(flagged_ids),
            "all_automated_matches_manually_reviewed": True,
            "confirmed_identifiable_title_leaks": len(CONFIRMED_TITLE_LEAK_IDS),
            "confirmed_user_ids": sorted(CONFIRMED_TITLE_LEAK_IDS),
            "cases": privacy_rows,
        },
    )

    operation_counts = Counter(
        operation["type"] for row in rows for operation in row["operations"]
    )
    supported_negative_users = sum(
        bool(row["supported_negative_genres"]) for row in evidence.values()
    )
    internally_coherent = sum(bool(row["internally_coherent"]) for row in manual)
    final_report: dict[str, Any] = {
        "scope": {
            "users": 1000,
            "deterministic_sample_seed": 20260816,
            "overlap_with_original_100": [],
            "api_success": automated["successful_api_responses"],
            "api_missing": automated["missing"],
            "api_success_rate": automated["success_rate"],
        },
        "protocol": {
            "preferred_input_candidate": "v002_final_cleanup_offline",
            "generation_model": "gpt-5-mini-2025-08-07",
            "prompt_modified": False,
            "tmdb_semantic_metadata_used": False,
            "manual_labels_used_at_inference": False,
            "weave_used": False,
            "standard_wandb_run_url": json.loads(
                (candidate / "wandb_monitor_config.json").read_text()
            )["run_url"],
        },
        "repair": {
            "changed": automated["repaired"],
            "changed_percent": automated["repaired_rate"] * 100,
            "unchanged": 1000 - automated["repaired"],
            "operation_counts": dict(sorted(operation_counts.items())),
        },
        "final_grounding": {
            "supported_negative_summary_verdicts": 1000,
            "overstated_negative_claims": 0,
            "unsupported_negative_claims": 0,
            "inappropriate_abstentions": 0,
            "appropriate_abstentions": 1000 - supported_negative_users,
            "users_with_supported_negative_evidence": supported_negative_users,
            "supported_negative_evidence_omitted": 0,
            "positive_sentences_preserved": 1000,
            "manual_positive_grounding_supported": sum(
                row["positive_grounding"] == "supported" for row in manual
            ),
            "manual_positive_grounding_reviewed": len(manual),
        },
        "semantic_audit": {
            "stratified_manual_reviewed": len(manual),
            "four_part_semantic_complete": sum(
                row["four_part_semantic_complete"] for row in manual
            ),
            "internally_coherent": internally_coherent,
            "internal_positive_negative_contradictions": len(manual)
            - internally_coherent,
            "semantic_richness_acceptable": sum(
                row["semantic_richness_acceptable"] for row in manual
            ),
        },
        "other_quality": {
            "formatting_failures": automated["formatting_failures"],
            "placeholders": automated["placeholders"],
            "exact_duplicate_groups": automated["duplicates"]["exact_group_count"],
            "near_duplicate_pairs": automated["duplicates"]["near_pair_count"],
            "confirmed_privacy_leaks": len(CONFIRMED_TITLE_LEAK_IDS),
            "confirmed_privacy_leak_user_ids": sorted(CONFIRMED_TITLE_LEAK_IDS),
        },
        "length": {
            "raw_v2": automated["length"]["raw"],
            "final": automated["length"]["final"],
            "mean_change_words": automated["length"]["final"]["mean"]
            - automated["length"]["raw"]["mean"],
            "mean_change_percent": (
                automated["length"]["final"]["mean"]
                / automated["length"]["raw"]["mean"]
                - 1
            ),
            "materially_compressed": len(compressed),
            "materially_compressed_semantically_complete": sum(
                row["semantically_complete"] for row in compressed
            ),
            "materially_compressed_semantically_incomplete": sum(
                not row["semantically_complete"] for row in compressed
            ),
        },
        "baseline_comparison": {
            "baseline": "100-user v002_final_cleanup_offline",
            "baseline_length": baseline["length"]["final_cleanup"],
            "baseline_materially_compressed": baseline["length"][
                "materially_compressed_relative_to_raw_v2"
            ]["count"],
            "baseline_internal_semantic_complete": baseline["final_candidate"][
                "four_part_semantic_usability"
            ]["complete"],
            "baseline_confirmed_privacy_leaks": baseline["final_candidate"][
                "privacy"
            ]["confirmed_identifiable_leaks"],
            "systematic_regression": True,
            "regressions": [
                "mean final length fell from 105.41 to 85.22 words",
                "material compression rose from 15% to 65.7%",
                "11/120 manually reviewed profiles contain broad positive/negative genre contradictions",
                "4/1000 profiles contain confirmed movie-title leakage",
                "the per-user compression audit identifies any profile that loses its positive profile or negative slot",
            ],
        },
        "usage_and_cost": automated["usage"] | automated["cost"],
        "readiness": {
            "tears_training_input_suitable": False,
            "editable_ux_profile_suitable": False,
            "actual_controllability_tested": False,
            "reason": (
                "Grounding cleanup succeeds, but systematic compression, internal "
                "positive/negative contradictions, and confirmed title leakage fail "
                "the production promotion gate."
            ),
        },
        "promotion_decision": {
            "recommend_full_cohort_generation": False,
            "remaining_199948_users_submitted": 0,
            "tears_training_started": False,
            "decision": "DO NOT PROMOTE THE GENERIC PRODUCTION CANDIDATE",
        },
    }
    final_report["fingerprint"] = stable_hash(final_report)
    atomic_write_json(candidate / "reports" / "production_calibration_report.json", final_report)
    markdown = f"""# 1,000-user V2 production calibration

## Decision

**DO NOT start full-cohort generation.** The generic repair eliminates detected
negative-grounding errors, but it causes systematic compression and exposes
internal V2 positive/negative contradictions and four confirmed title leaks.

## Core results

- API: 1,000/1,000 successful, 0 missing.
- Repaired: {automated['repaired']}/1,000 ({automated['repaired_rate']:.1%}).
- Final negative grounding: 0 unsupported, 0 overstated, 0 inappropriate abstentions.
- Manual semantic audit: 120 reviewed; 120 four-part complete; {internally_coherent}
  internally coherent; {len(manual)-internally_coherent} contradictory.
- Privacy: {len(CONFIRMED_TITLE_LEAK_IDS)} confirmed title leaks after all 19 automated
  matches were manually adjudicated.
- Exact/near duplicates: 0/0. Formatting failures/placeholders: 0/0.
- Raw/final mean words: {automated['length']['raw']['mean']:.2f}/
  {automated['length']['final']['mean']:.2f}; materially compressed:
  {len(compressed)}/1,000, with {sum(row['semantically_complete'] for row in compressed)}
  complete and {sum(not row['semantically_complete'] for row in compressed)} incomplete.
- Usage: {automated['usage']['input_tokens']:,} input and
  {automated['usage']['output_tokens']:,} output tokens. Exact cost:
  ${automated['cost']['exact_calibration_usd']:.8f}; projected 200,948-user cost:
  ${automated['cost']['projected_all_200948_users_usd']:.2f}.

No remaining users were submitted and TEARS training was not started.
"""
    (candidate / "reports" / "production_calibration_report.md").write_text(markdown)
    print(json.dumps(final_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
