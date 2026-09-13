"""Freeze the manual semantic audit for the v12 paired calibration.

This file is reporting-only.  Its explicit audit labels are never imported by
the inference, prompt, validation, retry, or full-cohort code paths.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from tears_training.evidence_gated_calibration_v11 import _log_wandb


ROOT = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "production_calibration_1000/v012_20260816_evidence_gated_genre_contract_final"
)
V10 = ROOT.parent / "v010_20260816_evidence_gated_emiliano"


POSITIVE_ON_NEGATIVE = {
    8742: ["Crime"],
    13279: ["Comedy", "Drama", "Fantasy", "Mystery", "Sci-Fi"],
    24796: ["Drama", "Mystery", "Sci-Fi"],
    31715: ["Fantasy"],
    83310: ["Adventure", "Animation", "Fantasy"],
    98331: ["Drama", "Fantasy", "Horror", "Mystery", "Thriller"],
    114176: ["Action", "Comedy", "Drama", "Sci-Fi"],
    124152: ["Adventure"],
    124194: ["Comedy", "Drama", "Romance"],
    157895: ["Animation", "Drama"],
    176397: ["Action", "Comedy"],
    189979: ["Drama", "Romance"],
}

POSITIVE_ON_MIXED = {
    111189: ["Drama", "Romance"],
    115286: ["Drama"],
    152245: ["Comedy"],
    176592: ["Drama"],
}

POSITIVE_ON_INSUFFICIENT = {
    24796: ["Adventure"],
    31715: ["Animation"],
    71650: ["Drama"],
    111189: ["Action", "Adventure", "Sci-Fi", "Western"],
    114176: ["Crime"],
    115286: ["Musical", "Romance"],
    124152: ["Animation", "Crime", "Thriller"],
    152245: ["Crime", "Thriller"],
    175668: ["Adventure", "Fantasy"],
    176397: ["Drama"],
    176592: ["Adventure", "Fantasy"],
    184219: ["Action", "Adventure"],
    185663: ["Crime", "Mystery", "Thriller"],
}

SUPPORTED_NEGATIVE_OMISSIONS = {24796: ["Fantasy", "Mystery"]}


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    automated = load_json(ROOT / "reports/automated_validation_report.json")
    v10 = load_json(V10 / "reports/final_audit_report.json")
    queue = load_jsonl(ROOT / "reports/manual_audit_queue.jsonl")
    selected = load_jsonl(ROOT / "validated/final_selected_summaries.jsonl")
    sample_ids = [int(row["user_id"]) for row in queue]
    sample_set = set(sample_ids)
    assert len(sample_ids) == 122 == len(sample_set)

    strict_positive_failures = (
        set(POSITIVE_ON_NEGATIVE)
        | set(POSITIVE_ON_MIXED)
        | set(POSITIVE_ON_INSUFFICIENT)
    )
    assert strict_positive_failures <= sample_set
    assert set(SUPPORTED_NEGATIVE_OMISSIONS) <= sample_set

    labels = []
    for row in queue:
        user_id = int(row["user_id"])
        labels.append(
            {
                "user_id": user_id,
                "negative_evidence_status": row["negative_evidence_status"],
                "mixed_genre_user": bool(row["genre_classifications"]["MIXED"]),
                "unsupported_negative_claim": False,
                "overstated_negative_claim": False,
                "inappropriate_abstention": False,
                "supported_negative_genres_omitted": SUPPORTED_NEGATIVE_OMISSIONS.get(user_id, []),
                "positive_preference_traceable_to_positive_examples_or_appropriately_omitted": True,
                "strict_positive_genre_grounding": user_id not in strict_positive_failures,
                "categorical_positive_on_negative_genres": POSITIVE_ON_NEGATIVE.get(user_id, []),
                "categorical_positive_on_mixed_genres": POSITIVE_ON_MIXED.get(user_id, []),
                "categorical_positive_on_insufficient_genres": POSITIVE_ON_INSUFFICIENT.get(user_id, []),
                "substantive_positive_negative_contradiction": user_id in POSITIVE_ON_NEGATIVE,
                "four_part_semantic_usability": True,
                "strict_semantic_completeness": (
                    user_id not in strict_positive_failures
                    and user_id not in SUPPORTED_NEGATIVE_OMISSIONS
                ),
                "expressive_editable_tears_profile": True,
            }
        )

    status_counts = Counter(row["negative_evidence_status"] for row in queue)
    final_missing = {
        int(row["user_id"]): row["contract"]["missing_required_negative_genres"]
        for row in selected
        if row["contract"].get("missing_required_negative_genres")
    }
    assert final_missing == {24796: ["Fantasy", "Mystery"], 46654: ["Crime", "Drama"]}
    required_instances = sum(
        len(row["evidence"]["required_negative_genres"]) for row in queue
    )
    sample_missing_instances = sum(map(len, SUPPORTED_NEGATIVE_OMISSIONS.values()))

    previous_six = {5282, 26328, 64863, 83310, 111189, 198747}
    previous_fifteen = {
        5282, 8742, 13279, 24796, 26328, 31715, 64863, 83310,
        98331, 111189, 114176, 124152, 157895, 176397, 198747,
    }
    previous_four_omissions = {13279, 83310, 124194, 189979}

    report = {
        "scope": {
            "users": 1000,
            "same_paired_users_as_v10": True,
            "model": "gpt-5-mini-2025-08-07",
            "tmdb_semantic_metadata_used": False,
            "semantic_post_generation_repair_users": 0,
            "full_cohort_started": False,
            "tears_training_started": False,
            "weave_enabled": False,
        },
        "offline_preflight": load_json(ROOT / "reports/preflight_report.json"),
        "api_and_retry": {
            "initial_api_success": 1000,
            "initial_api_missing_or_failed": 0,
            "all_paid_requests_success": 1056,
            "all_paid_requests_missing_or_failed": 0,
            "users_retried_at_least_once": 45,
            "users_retried_twice": 11,
            "coverage_retry_resolved_users": 43,
            "residual_coverage_omission_users": final_missing,
        },
        "automated_final_audit": automated,
        "automated_contract_screen_interpretation": {
            "raw_contract_pass": automated["contract"]["final_contract_pass"],
            "raw_contract_fail": automated["contract"]["final_contract_fail"],
            "warning_fields": automated["contract"]["failure_fields"],
            "note": (
                "These are deterministic lexical-screen warnings, not adjudicated semantic-error counts. "
                "The initial overinclusive validator probes are preserved under reports/validator_probe_*; "
                "manual review distinguishes contextual theme wording and explicit uncertainty from categorical claims."
            ),
            "all_nine_NONE_dislike_flags_manually_checked_as_abstentions": True,
            "adjudicated_NONE_fabricated_dislikes": 0,
            "required_negative_genre_instances": 683,
            "required_negative_genre_instances_covered": 679,
            "required_negative_genre_instances_omitted": 4,
        },
        "manual_semantic_audit": {
            "reviewed": 122,
            "same_v10_stratified_120_plus_two_prior_omission_users": True,
            "sample_user_ids_sha256": hashlib.sha256(
                json.dumps(sample_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "strata": dict(status_counts),
            "mixed_genre_users": sum(bool(row["genre_classifications"]["MIXED"]) for row in queue),
            "unsupported_negative_claims": 0,
            "overstated_negative_claims": 0,
            "inappropriate_abstentions": 0,
            "NONE_no_fabricated_dislike": {"passed": status_counts["NONE"], "total": status_counts["NONE"]},
            "WEAK_narrow_and_qualified": {"passed": status_counts["WEAK"], "total": status_counts["WEAK"]},
            "STRONG_expressed_supported_pattern": {"passed": status_counts["STRONG"], "total": status_counts["STRONG"]},
            "supported_negative_omission_users": SUPPORTED_NEGATIVE_OMISSIONS,
            "sample_required_negative_instances": required_instances,
            "sample_required_negative_instances_covered": required_instances - sample_missing_instances,
            "positive_preference_traceable_or_appropriately_omitted": 122,
            "strict_positive_genre_grounding_pass": 122 - len(strict_positive_failures),
            "strict_positive_genre_grounding_fail": len(strict_positive_failures),
            "strict_positive_failure_user_ids": sorted(strict_positive_failures),
            "categorical_positive_on_negative_genre_users": POSITIVE_ON_NEGATIVE,
            "categorical_positive_on_mixed_genre_users": POSITIVE_ON_MIXED,
            "categorical_positive_on_insufficient_genre_users": POSITIVE_ON_INSUFFICIENT,
            "substantive_positive_negative_contradictions": len(POSITIVE_ON_NEGATIVE),
            "substantive_positive_negative_contradiction_user_ids": sorted(POSITIVE_ON_NEGATIVE),
            "four_part_semantic_usability": 122,
            "strict_semantic_completeness": sum(row["strict_semantic_completeness"] for row in labels),
            "expressive_editable_profile": 122,
            "actual_controllability_tested": False,
        },
        "regression_case_outcomes": {
            "previous_15_strict_positive_failures_resolved": sorted(previous_fifteen - strict_positive_failures),
            "previous_15_strict_positive_failures_remaining": sorted(previous_fifteen & strict_positive_failures),
            "previous_6_mixed_failures_resolved": sorted(previous_six - strict_positive_failures),
            "previous_6_mixed_failures_remaining": sorted(previous_six & strict_positive_failures),
            "previous_4_negative_omission_users_all_resolved": not bool(previous_four_omissions & set(final_missing)),
            "new_residual_negative_omission_users": final_missing,
        },
        "v10_comparison": {
            "v10_length": v10["length"]["final"],
            "v12_length": automated["length"],
            "mean_word_change": automated["length"]["mean"] - v10["length"]["final"]["mean"],
            "v10_strict_positive_grounding": "105/120",
            "v12_strict_positive_grounding": f"{122 - len(strict_positive_failures)}/122",
            "v10_supported_negative_omission_users_full_1000": 4,
            "v12_supported_negative_omission_users_full_1000": len(final_missing),
            "v10_cleanup_users": v10["privacy_and_format_cleanup"]["total_users"],
            "v12_cleanup_users": automated["allowed_cleanup"]["users"],
            "v10_exact_cost_usd": v10["usage_and_cost"]["exact_calibration_usd"],
            "v12_exact_cost_usd": automated["cost"]["exact_calibration_usd"],
        },
        "training_and_ux_readiness": {
            "structurally_suitable_as_tears_training_text": True,
            "semantically_recommended_for_tears_training": False,
            "editable_profile_richness_preserved": True,
            "recommended_for_final_ux_study": False,
            "reason": (
                "Natural-language richness and editability are preserved, but strict aggregate positive-genre "
                "grounding regressed in the manual sample, substantive positive/negative contradictions remain, "
                "and two STRONG users still omit four required negative genres after the retry limit."
            ),
        },
        "promotion_decision": {
            "recommend_full_cohort_generation": False,
            "failed_gates": [
                "20/122 manually audited profiles fail strict positive genre grounding",
                "12/122 contain a substantive positive/negative genre contradiction",
                "2/204 STRONG users omit four required negative genres after targeted retries",
                "2 of the six prior mixed-positive regression exemplars remain unresolved",
            ],
            "full_cohort_submitted": False,
            "tears_training_started": False,
        },
    }

    (ROOT / "reports/manual_audit_labels.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in labels)
    )
    (ROOT / "reports/final_audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    runtime_dir = ROOT / "code/final_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        Path("tears_training/evidence_gated_summary_harness_v11.py"),
        Path("tears_training/evidence_gated_calibration_v11.py"),
        Path("tests/test_evidence_gated_summary_harness_v11.py"),
        Path("scripts/finalize_evidence_gated_genre_contract_calibration.py"),
    ]
    runtime_manifest = {}
    for source in sources:
        target = runtime_dir / source.name
        shutil.copy2(source, target)
        runtime_manifest[source.name] = {
            "workspace_path": str(source.resolve()),
            "artifact_path": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    (runtime_dir / "manifest.json").write_text(
        json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n"
    )
    _log_wandb(
        ROOT,
        load_json(ROOT / "wandb_monitor_config.json"),
        {
            "manual/reviewed": 122,
            "manual/unsupported_negative_claims": 0,
            "manual/overstated_negative_claims": 0,
            "manual/inappropriate_abstentions": 0,
            "manual/strict_positive_grounding_pass": 102,
            "manual/strict_positive_grounding_fail": 20,
            "manual/positive_negative_contradictions": 12,
            "manual/semantic_completeness": 102,
            "manual/expressive_editable": 122,
            "validation/final_required_negative_omission_users": 2,
            "promotion/recommend_full_cohort": 0,
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
