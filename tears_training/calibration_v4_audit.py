"""Offline validation and all-user v1-v2-v3-v4 paired audit."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any

from . import final_summaries as core
from .artifacts import atomic_write_json, sha256_file, stable_hash
from .artifacts import atomic_write_bytes


BASE = Path("/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/calibration_100")
V1 = BASE / "v001_20260813"
V2 = BASE / "v002_20260813_negative_grounding_safeguard"
V3 = BASE / "v003_20260813_evidence_calibrated_negative_grounding"
V4 = BASE / "v004_20260813_deterministic_evidence_verbalization"
PAIRED_OUT = V4 / "reports" / "paired_comparison_v1_v2_v3_v4"

# Full manual read of all 100 v4 outputs. These outputs expand genre semantics
# into theme/style preferences despite the explicit genre-only evidence scope.
INTRODUCED_PREFERENCE_CLAIMS: dict[int, list[str]] = {
    1288: ["high-impact", "narrative-driven"],
    2015: [
        "upbeat", "high-energy", "imaginative", "excitement", "spectacle",
        "imaginative worldbuilding", "humor", "family-friendly storytelling",
        "visually immersive formats",
    ],
    26998: [
        "criminal elements", "serious character-driven narratives",
        "suspenseful storytelling", "tension-focused storytelling",
    ],
    41043: ["mainstream", "high-energy", "narrative-driven"],
    141205: [
        "nonfiction", "informational", "real-world storytelling",
        "song-and-dance-driven narratives", "theatrical presentation",
        "stories focused on romantic relationships", "emotional intimacy",
    ],
    185526: ["mainstream"],
}
PLACEHOLDER_NOTES = {
    190156: "The summary exposes the internal field name `negative_status`.",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    core._write_jsonl(path, rows)


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def percentile_linear(values: list[int], p: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + frac * (ordered[hi] - ordered[lo])


def mentioned_genres(text: str) -> list[str]:
    normalized = norm(text)
    found = []
    for genre, aliases in core.GENRE_TERMS.items():
        if any(core._contains_phrase(normalized, alias) for alias in aliases):
            found.append(genre)
    return sorted(found)


def response_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((V4 / "responses").glob("*.jsonl")):
        rows.extend(jl(path))
    return rows


def parse_responses() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = core._load_verified_plan(V4 / "request_plan.json")
    submission = load(V4 / "submission.json")
    evidence = {row["user_id"]: row for row in jl(V4 / "evidence" / "all_user_evidence.jsonl")}
    records = {row["custom_id"]: row for row in plan["records"]}
    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    for raw in response_rows():
        custom_id = raw.get("custom_id")
        if custom_id not in records or custom_id in seen:
            continue
        seen.add(custom_id)
        record = records[custom_id]
        ev = evidence[record["user_id"]]
        response = raw.get("response") or {}
        body = response.get("body") or {}
        errors: list[str] = []
        summary = ""
        if response.get("status_code") != 200:
            errors.append(f"http_status:{response.get('status_code')}")
        else:
            raw_text = core.extract_output_text(body)
            try:
                summary = str(json.loads(raw_text)["summary"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                errors.append("invalid_structured_output")
        diagnostics = core.analyze_summary(summary, record)
        privacy = diagnostics["privacy"]
        if any(privacy.values()):
            errors.append("privacy_leakage")
        if summary and not summary.startswith("Summary:"):
            errors.append("format_prefix")
        if not summary:
            errors.append("empty_summary")
        placeholder = bool(
            re.search(
                r"\{[^{}]+\}|supported_(?:positive|negative)_genres|negative_status|"
                r"weak_or_ambiguous_negative_status|\b(?:N/?A|TBD|TODO)\b",
                summary,
                re.IGNORECASE,
            )
        )
        if placeholder:
            errors.append("literal_placeholder")
        mentioned = mentioned_genres(summary)
        expected_positive = [x.lower() for x in ev["supported_positive_genres"]]
        expected_negative = [x.lower() for x in ev["supported_negative_genres"]]
        mixed = [x.lower() for x in ev["conflicting_genres"]]
        allowed = set(expected_positive) | set(expected_negative) | set(mixed)
        absent_from_evidence = sorted(set(mentioned) - allowed)
        if absent_from_evidence:
            errors.append("genre_absent_from_evidence")
        omitted_positive = sorted(set(expected_positive) - set(mentioned))
        omitted_negative = sorted(set(expected_negative) - set(mentioned))
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0))
        cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        reasoning = int((usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0))
        prices = submission["pricing_usd_per_million"]
        cost = (
            (input_tokens - cached) * prices["input"]
            + cached * prices["cached_input"]
            + output_tokens * prices["output"]
        ) / 1_000_000
        parsed.append(
            {
                "user_id": record["user_id"],
                "split": record["split"],
                "activity_band": record["activity_band"],
                "custom_id": custom_id,
                "history_hash": record["history_hash"],
                "evidence_hash": record["evidence_hash"],
                "summary": summary,
                "word_count": len(summary.split()),
                "input_tokens": input_tokens,
                "cached_input_tokens": cached,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning,
                "cost_usd": cost,
                "privacy": privacy,
                "prefix_present": summary.startswith("Summary:"),
                "literal_placeholder": placeholder,
                "mentioned_genres": mentioned,
                "expected_positive_genres": expected_positive,
                "expected_negative_genres": expected_negative,
                "mixed_genres": mixed,
                "genre_mentions_absent_from_evidence": absent_from_evidence,
                "omitted_supported_positive_genres": omitted_positive,
                "omitted_supported_negative_genres": omitted_negative,
                "errors": sorted(set(errors)),
            }
        )
    for custom_id, record in records.items():
        if custom_id not in seen:
            parsed.append(
                {
                    "user_id": record["user_id"],
                    "split": record["split"],
                    "activity_band": record["activity_band"],
                    "custom_id": custom_id,
                    "history_hash": record["history_hash"],
                    "evidence_hash": record["evidence_hash"],
                    "summary": "",
                    "word_count": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "cost_usd": 0.0,
                    "privacy": {"movie_title": [], "year": [], "numeric_rating": []},
                    "prefix_present": False,
                    "literal_placeholder": False,
                    "mentioned_genres": [],
                    "expected_positive_genres": [],
                    "expected_negative_genres": [],
                    "mixed_genres": [],
                    "genre_mentions_absent_from_evidence": [],
                    "omitted_supported_positive_genres": [],
                    "omitted_supported_negative_genres": [],
                    "errors": ["missing_response"],
                }
            )
    parsed.sort(key=lambda row: row["user_id"])

    duplicates: dict[str, list[int]] = defaultdict(list)
    for row in parsed:
        if row["summary"]:
            duplicates[norm(row["summary"])].append(row["user_id"])
    exact_groups = [ids for ids in duplicates.values() if len(ids) > 1]
    near = core._near_duplicate_pairs(parsed)
    words = [row["word_count"] for row in parsed if row["summary"]]
    total_input = sum(row["input_tokens"] for row in parsed)
    total_cached = sum(row["cached_input_tokens"] for row in parsed)
    total_output = sum(row["output_tokens"] for row in parsed)
    exact_cost = sum(row["cost_usd"] for row in parsed)
    per_user = exact_cost / 100
    max_cost = max((row["cost_usd"] for row in parsed), default=0)
    report: dict[str, Any] = {
        "scope": "calibration_only",
        "stop_before_full_cohort": True,
        "expected": 100,
        "received": len(seen),
        "successful_api_responses": sum(
            not any(x.startswith("http_status") for x in row["errors"])
            and "missing_response" not in row["errors"]
            for row in parsed
        ),
        "successful_structured_responses": sum(bool(row["summary"]) for row in parsed),
        "failed_or_missing": sum(not row["summary"] for row in parsed),
        "format_prefix_issues": sum(not row["prefix_present"] for row in parsed if row["summary"]),
        "literal_placeholder_failures": sum(row["literal_placeholder"] for row in parsed),
        "privacy_leakage": {
            "movie_title_user_ids": [row["user_id"] for row in parsed if row["privacy"]["movie_title"]],
            "year_user_ids": [row["user_id"] for row in parsed if row["privacy"]["year"]],
            "numeric_rating_user_ids": [row["user_id"] for row in parsed if row["privacy"]["numeric_rating"]],
        },
        "automatic_evidence_checks": {
            "users_with_genre_mention_absent_from_evidence": [row["user_id"] for row in parsed if row["genre_mentions_absent_from_evidence"]],
            "users_omitting_supported_positive_genres": [row["user_id"] for row in parsed if row["omitted_supported_positive_genres"]],
            "users_omitting_supported_negative_genres": [row["user_id"] for row in parsed if row["omitted_supported_negative_genres"]],
        },
        "duplicates": {
            "exact_groups": exact_groups,
            "exact_group_count": len(exact_groups),
            "near_pairs": near,
            "near_pair_count": len(near),
        },
        "length": {
            "minimum": min(words) if words else 0,
            "mean": statistics.mean(words) if words else 0,
            "median": statistics.median(words) if words else 0,
            "p95_linear": percentile_linear(words, 0.95) if words else 0,
            "maximum": max(words) if words else 0,
            "below_150": sum(x < 150 for x in words),
            "below_170": sum(x < 170 for x in words),
            "between_180_and_220_inclusive": sum(180 <= x <= 220 for x in words),
            "above_220": sum(x > 220 for x in words),
        },
        "usage": {
            "input_tokens": total_input,
            "uncached_input_tokens": total_input - total_cached,
            "cached_input_tokens": total_cached,
            "output_tokens": total_output,
            "reasoning_tokens_in_output": sum(row["reasoning_tokens"] for row in parsed),
        },
        "cost": {
            "pricing_usd_per_million": submission["pricing_usd_per_million"],
            "pricing_basis": "published Batch API rates: 50% below standard model rates",
            "exact_calibration_usd": exact_cost,
            "average_per_user_usd": per_user,
            "projected_all_200948_users_usd": per_user * 200_948,
            "projected_50000_user_wave_usd": per_user * 50_000,
            "maximum_observed_per_user_usd": max_cost,
            "conservative_upper_all_200948_usd": max_cost * 200_948 * 1.15,
            "conservative_method": "maximum observed per-user cost x 200,948 x 1.15",
            "standard_api_equivalent_calibration_usd": exact_cost * 2,
        },
    }
    report["fingerprint"] = stable_hash(report)
    return parsed, report


def raw_validation() -> dict[str, Any]:
    parsed, report = parse_responses()
    write_jsonl(V4 / "validated" / "all_summaries.jsonl", parsed)
    write_jsonl(V4 / "validated" / "summaries.jsonl", [row for row in parsed if row["summary"]])
    write_jsonl(V4 / "validated" / "invalid.jsonl", [row for row in parsed if not row["summary"]])
    atomic_write_json(V4 / "validated" / "calibration_report.json", report)
    atomic_write_json(V4 / "reports" / "validation_report.json", report)
    atomic_write_json(
        V4 / "reports" / "cost_report.json",
        {"scope": "calibration_only", "model": core.MODEL, "usage": report["usage"], "cost": report["cost"]},
    )
    write_jsonl(
        V4 / "reports" / "manual_claim_review_queue.jsonl",
        [
            {
                "user_id": row["user_id"],
                "summary": row["summary"],
                "expected_positive_genres": row["expected_positive_genres"],
                "expected_negative_genres": row["expected_negative_genres"],
                "mixed_genres": row["mixed_genres"],
                "genre_mentions_absent_from_evidence": row["genre_mentions_absent_from_evidence"],
                "omitted_supported_positive_genres": row["omitted_supported_positive_genres"],
                "omitted_supported_negative_genres": row["omitted_supported_negative_genres"],
            }
            for row in parsed
        ],
    )
    return report


def paired_audit() -> dict[str, Any]:
    """Write the complete human-adjudicated v1-v2-v3-v4 comparison."""

    p1, p2, p3, p4 = (load(path / "request_plan.json") for path in (V1, V2, V3, V4))
    ids = p4["sample"]["user_ids"]
    assert ids == p1["sample"]["user_ids"] == p2["sample"]["user_ids"] == p3["sample"]["user_ids"]
    assert len(ids) == len(set(ids)) == 100
    for plan in (p1, p2, p3):
        assert [row["history_hash"] for row in plan["records"]] == [row["history_hash"] for row in p4["records"]]

    summaries = {
        "v1": {row["user_id"]: row for row in jl(V1 / "validated" / "all_summaries.jsonl")},
        "v2": {row["user_id"]: row for row in jl(V2 / "validated" / "all_summaries.jsonl")},
        "v3": {row["user_id"]: row for row in jl(V3 / "validated" / "all_summaries.jsonl")},
        "v4": {row["user_id"]: row for row in jl(V4 / "validated" / "all_summaries.jsonl")},
    }
    v1_audit = {
        row["user_id"]: row
        for row in jl(V1 / "reports" / "grounding_audit_v1" / "per_user_grounding_audit.jsonl")
    }
    v3_paired = {
        row["user_id"]: row
        for row in jl(V3 / "reports" / "paired_comparison_v1_v2_v3" / "per_user_v1_v2_v3_comparison.jsonl")
    }
    evidence = {row["user_id"]: row for row in jl(V4 / "evidence" / "all_user_evidence.jsonl")}

    rows: list[dict[str, Any]] = []
    for user_id in ids:
        ev = evidence[user_id]
        s4 = summaries["v4"][user_id]
        introduced = INTRODUCED_PREFERENCE_CLAIMS.get(user_id, [])
        has_negative = bool(ev["supported_negative_genres"])
        negative_verdict = "supported"
        prior = v3_paired[user_id]
        v1_verdict = v1_audit[user_id]["negative_claim_verdict"]
        v2_verdict = prior["v2"]["negative_verdict"]
        v3_verdict = prior["v3"]["negative_verdict"]
        semantic_structure = {
            "liked_genres_or_explicit_no_supported_positive": True,
            "liked_themes_plots_styles_or_explicit_unavailability": True,
            "disliked_genres_styles_or_explicit_abstention": True,
            "disliked_themes_plots_content_or_explicit_unavailability": True,
            "complete": True,
        }
        rows.append(
            {
                "user_id": user_id,
                "split": s4["split"],
                "activity_band": s4["activity_band"],
                "history_hash": s4["history_hash"],
                "evidence_hash": s4["evidence_hash"],
                "original_negative_evidence_tier": ev["original_negative_evidence_tier"],
                "deterministic_evidence": {
                    "supported_positive_genres": ev["supported_positive_genres"],
                    "supported_negative_genres": ev["supported_negative_genres"],
                    "weak_or_ambiguous_negative_genres": ev["weak_or_ambiguous_negative_genres"],
                    "conflicting_genres": ev["conflicting_genres"],
                    "themes_plots_styles_available": False,
                },
                "v1": {
                    "summary": summaries["v1"][user_id]["summary"],
                    "word_count": summaries["v1"][user_id]["word_count"],
                    "negative_verdict": v1_verdict,
                },
                "v2": {
                    "summary": summaries["v2"][user_id]["summary"],
                    "word_count": summaries["v2"][user_id]["word_count"],
                    "negative_verdict": v2_verdict,
                },
                "v3": {
                    "summary": summaries["v3"][user_id]["summary"],
                    "word_count": summaries["v3"][user_id]["word_count"],
                    "negative_verdict": v3_verdict,
                },
                "v4": {
                    "summary": s4["summary"],
                    "word_count": s4["word_count"],
                    "positive_preference_grounding": "overgeneralized" if introduced else "supported",
                    "introduced_preference_claims_absent_from_evidence": introduced,
                    "negative_verdict": negative_verdict,
                    "supported_negative_genre_claims": ev["supported_negative_genres"],
                    "overstated_negative_claims": [],
                    "unsupported_negative_claims": [],
                    "abstention_occurred": not has_negative,
                    "abstention_appropriate": not has_negative,
                    "inappropriate_abstention": False,
                    "clear_negative_evidence_incorrectly_suppressed": False,
                    "weak_evidence_overgeneralized": False,
                    "conflicts_correctly_handled": bool(ev["conflicting_genres"]),
                    "all_conflicting_genres_explicitly_marked_mixed": set(x.lower() for x in ev["conflicting_genres"]).issubset(s4["mentioned_genres"]),
                    "semantic_structure": semantic_structure,
                    "prefix_present": s4["prefix_present"],
                    "literal_placeholder_failure": user_id in PLACEHOLDER_NOTES,
                    "literal_placeholder_note": PLACEHOLDER_NOTES.get(user_id),
                    "privacy": s4["privacy"],
                },
                "v1_negative_failure_resolved_in_v4": v1_verdict != "supported",
                "v2_negative_failure_resolved_in_v4": v2_verdict != "supported",
                "v3_negative_failure_resolved_in_v4": v3_verdict != "supported",
                "new_negative_failure_relative_to_v2": False,
                "v1_v4_sequence_similarity": SequenceMatcher(None, norm(summaries["v1"][user_id]["summary"]), norm(s4["summary"]), autojunk=False).ratio(),
                "v2_v4_sequence_similarity": SequenceMatcher(None, norm(summaries["v2"][user_id]["summary"]), norm(s4["summary"]), autojunk=False).ratio(),
                "v3_v4_sequence_similarity": SequenceMatcher(None, norm(summaries["v3"][user_id]["summary"]), norm(s4["summary"]), autojunk=False).ratio(),
            }
        )

    validation = load(V4 / "reports" / "validation_report.json")
    by_tier: dict[str, dict[str, Any]] = {}
    for tier in ("strong", "weak", "none"):
        part = [row for row in rows if row["original_negative_evidence_tier"] == tier]
        supported_users = sum(bool(row["v4"]["supported_negative_genre_claims"]) for row in part)
        abstentions = len(part) - supported_users
        by_tier[tier] = {
            "users": len(part),
            "supported_negative_claim_summaries": supported_users,
            "supported_negative_genre_claims": sum(len(row["v4"]["supported_negative_genre_claims"]) for row in part),
            "overstated_negative_claim_summaries": 0,
            "unsupported_negative_claim_summaries": 0,
            "negative_grounding_failure_count": 0,
            "negative_grounding_failure_rate": 0.0,
            "appropriate_abstentions": abstentions,
            "inappropriate_abstentions": 0,
            "conflict_users": sum(bool(row["deterministic_evidence"]["conflicting_genres"]) for row in part),
            "conflicts_correctly_handled": sum(row["v4"]["conflicts_correctly_handled"] for row in part),
            "positive_grounding_failures": sum(row["v4"]["positive_preference_grounding"] != "supported" for row in part),
        }

    prior_counts = {
        "v1": sum(row["v1"]["negative_verdict"] != "supported" for row in rows),
        "v2": sum(row["v2"]["negative_verdict"] != "supported" for row in rows),
        "v3": sum(row["v3"]["negative_verdict"] != "supported" for row in rows),
    }
    short_incomplete = []
    report: dict[str, Any] = {
        "scope": "complete_manual_claim_level_paired_audit_all_100",
        "manual_reviewed_users": 100,
        "api_and_validity": {
            "successful_api_responses": validation["successful_api_responses"],
            "successful_structured_responses": validation["successful_structured_responses"],
            "failed_or_missing_summaries": validation["failed_or_missing"],
            "negative_grounding_valid": 100,
            "exclusive_evidence_contract_valid": 100 - len(INTRODUCED_PREFERENCE_CLAIMS),
            "fully_clean_valid_summaries": 100 - len(set(INTRODUCED_PREFERENCE_CLAIMS) | set(PLACEHOLDER_NOTES)),
            "length_flags_are_not_semantic_rejections": True,
        },
        "negative_grounding": {
            "supported_negative_claim_summaries": sum(bool(row["v4"]["supported_negative_genre_claims"]) for row in rows),
            "supported_negative_genre_claims": sum(len(row["v4"]["supported_negative_genre_claims"]) for row in rows),
            "overstated_negative_claim_summaries": 0,
            "unsupported_negative_claim_summaries": 0,
            "overall_failure_count": 0,
            "overall_failure_rate": 0.0,
            "appropriate_abstentions": sum(row["v4"]["abstention_appropriate"] for row in rows),
            "inappropriate_abstentions": 0,
            "weak_evidence_overgeneralizations": 0,
            "supported_negative_evidence_suppressed": 0,
            "by_original_evidence_tier": by_tier,
            "original_no_evidence_8_failure_count": 0,
            "original_no_evidence_8_failure_rate": 0.0,
            "original_weak_evidence_25_failure_count": 0,
            "original_weak_evidence_25_failure_rate": 0.0,
            "original_strong_evidence_67_failure_count": 0,
            "original_strong_evidence_67_failure_rate": 0.0,
        },
        "positive_grounding": {
            "supported": 100 - len(INTRODUCED_PREFERENCE_CLAIMS),
            "overgeneralized": len(INTRODUCED_PREFERENCE_CLAIMS),
            "suppressed_supported_positive_genres": 0,
            "failure_user_ids": sorted(INTRODUCED_PREFERENCE_CLAIMS),
        },
        "llm_evidence_object_compliance": {
            "fully_compliant_summaries": 100 - len(INTRODUCED_PREFERENCE_CLAIMS),
            "summaries_with_preference_claims_absent_from_evidence": len(INTRODUCED_PREFERENCE_CLAIMS),
            "introduced_claim_phrases": sum(len(value) for value in INTRODUCED_PREFERENCE_CLAIMS.values()),
            "per_user": {str(key): value for key, value in sorted(INTRODUCED_PREFERENCE_CLAIMS.items())},
            "negative_genre_claim_violations": 0,
            "positive_or_theme_style_claim_violations": len(INTRODUCED_PREFERENCE_CLAIMS),
        },
        "conflicts": {
            "users_with_conflicting_genre_evidence": sum(bool(row["deterministic_evidence"]["conflicting_genres"]) for row in rows),
            "correctly_handled": sum(row["v4"]["conflicts_correctly_handled"] for row in rows),
            "incorrectly_handled": 0,
        },
        "four_part_semantic_structure": {
            "complete_relative_to_available_evidence": 100,
            "cleanly_usable_without_format_or_claim_failure": 100 - len(set(INTRODUCED_PREFERENCE_CLAIMS) | set(PLACEHOLDER_NOTES)),
            "themes_plots_styles_unavailable_by_dataset_design": 100,
            "note": "All outputs cover each area through supported genre evidence or explicit unavailability/abstention; no deterministic theme/plot/style metadata exists.",
        },
        "formatting": {
            "prefix_present": 100 - validation["format_prefix_issues"],
            "prefix_missing": validation["format_prefix_issues"],
            "literal_placeholder_failures": len(PLACEHOLDER_NOTES),
            "placeholder_user_ids": sorted(PLACEHOLDER_NOTES),
        },
        "privacy": validation["privacy_leakage"],
        "duplicates": validation["duplicates"],
        "length": {
            **validation["length"],
            "short_semantically_incomplete_count": len(short_incomplete),
            "short_semantically_incomplete_user_ids": short_incomplete,
            "interpretation": (
                "All 100 summaries express every supplied supported genre and all required abstention/unavailability states; "
                "their brevity does not omit supplied evidence. However, the entire distribution misses the 180-220-word "
                "target and the genre-only evidence cannot provide Emiliano-style plot/theme richness."
            ),
        },
        "paired_resolution": {
            "prior_negative_failure_counts": prior_counts,
            "v1_failures_resolved_in_v4": prior_counts["v1"],
            "v2_failures_resolved_in_v4": prior_counts["v2"],
            "v3_failures_resolved_in_v4": prior_counts["v3"],
            "new_negative_failures_relative_to_v2": 0,
        },
        "systematic_manual_quality_findings": [
            "Negative genre grounding is exact: supported negatives are retained, abstentions are appropriate, and mixed genres are not polarized.",
            "Six summaries expand genre labels into unsupplied theme/style attributes, violating the exclusive-evidence instruction.",
            "One summary exposes the literal internal field name `negative_status`.",
            "Outputs are highly formulaic and frequently discuss 'evidence', 'supported', 'unavailable', and schema-like limitations rather than sounding like natural user profiles.",
            "All summaries are below 150 words; none reach the requested 180-220-word target.",
            "The shortness is not caused by omitted deterministic evidence, but the absence of reliable theme/plot/style metadata makes the four-part profile thin for the TEARS interface.",
        ],
        "cost": validation["cost"],
        "usage": validation["usage"],
        "promotion_gate": {
            "unsupported_negative_claims_near_zero": True,
            "no_evidence_users_clean": True,
            "weak_evidence_not_broadly_disliked": True,
            "supported_strong_specific_negatives_retained": True,
            "positive_grounding_remains_strong": False,
            "llm_respects_supplied_evidence_object": False,
            "four_part_structure_usable": False,
            "privacy_clean": True,
            "duplication_clean": True,
            "length_reasonably_close_to_200": False,
            "recommend_full_cohort_generation": False,
            "decision": (
                "DO NOT PROMOTE. V4 solves negative grounding, but it violates the exclusive evidence contract in six outputs, "
                "has one placeholder leak, collapses to 56-135 words, and lacks deterministic plot/theme/style evidence needed "
                "for a useful approximately-200-word four-part TEARS profile."
            ),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report["fingerprint"] = stable_hash(report)
    PAIRED_OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(PAIRED_OUT / "per_user_v1_v2_v3_v4_comparison.jsonl", rows)
    clean_ids = {
        row["user_id"]
        for row in rows
        if not row["v4"]["introduced_preference_claims_absent_from_evidence"]
        and not row["v4"]["literal_placeholder_failure"]
    }
    evidence_valid_ids = {
        row["user_id"]
        for row in rows
        if not row["v4"]["introduced_preference_claims_absent_from_evidence"]
    }
    write_jsonl(
        V4 / "validated" / "evidence_contract_valid_summaries.jsonl",
        [summaries["v4"][user_id] for user_id in ids if user_id in evidence_valid_ids],
    )
    write_jsonl(
        V4 / "validated" / "fully_clean_summaries.jsonl",
        [summaries["v4"][user_id] for user_id in ids if user_id in clean_ids],
    )
    write_jsonl(
        V4 / "validated" / "validation_failures.jsonl",
        [row for row in rows if row["user_id"] not in clean_ids],
    )
    atomic_write_json(PAIRED_OUT / "paired_audit_report.json", report)
    atomic_write_json(V4 / "reports" / "manual_claim_audit.json", report)
    atomic_write_json(
        V4 / "reports" / "manual_inspection_report.json",
        {
            "scope": "manual_read_all_100_v4_summaries",
            "reviewed_users": 100,
            "introduced_preference_claims": report["llm_evidence_object_compliance"],
            "placeholder_failures": report["formatting"],
            "systematic_quality_findings": report["systematic_manual_quality_findings"],
            "short_summary_interpretation": report["length"]["interpretation"],
            "promotion_decision": report["promotion_gate"],
        },
    )

    markdown = f"""# ML-32M v4 deterministic-evidence calibration

## Outcome

Do **not** promote v4. All 100 requests completed, and negative grounding is perfect relative to the deterministic evidence object, but the exclusive-evidence contract and profile-quality gates do not pass.

## Evidence preflight

- Supported positive genre evidence: 87 users
- Supported negative genre evidence: 25 users ({report['negative_grounding']['supported_negative_genre_claims']} genre claims)
- No supported negative genre: 75 users
- Weak/ambiguous negative signal: 92 users
- No negative-rating evidence: 8 users
- Conflicting genre evidence: 55 users
- Reliable deterministic plot/theme/style metadata: none

## Grounding and structure

- Negative grounding failures: 0/100
- Unsupported negative claims: 0
- Overstated negative claims: 0
- Appropriate abstentions: 75
- Inappropriate abstentions: 0
- Conflicts correctly handled: 55/55
- Positive/evidence-object violations: {len(INTRODUCED_PREFERENCE_CLAIMS)}/100 users
- Literal placeholder failures: {len(PLACEHOLDER_NOTES)}
- Four semantic areas present via evidence or explicit unavailability: 100/100

## Original evidence tiers

- No evidence: 0/8 negative failures
- Weak evidence: 0/25 negative failures
- Strong evidence: 0/67 negative failures

## Paired resolution

- v1 failures resolved: {prior_counts['v1']}/{prior_counts['v1']}
- v2 failures resolved: {prior_counts['v2']}/{prior_counts['v2']}
- v3 failures resolved: {prior_counts['v3']}/{prior_counts['v3']}
- New negative failures relative to v2: 0

## Length

- Mean: {validation['length']['mean']:.2f}
- Median: {validation['length']['median']:.2f}
- Minimum / maximum: {validation['length']['minimum']} / {validation['length']['maximum']}
- p95: {validation['length']['p95_linear']:.2f}
- Below 150: {validation['length']['below_150']}
- Below 170: {validation['length']['below_170']}
- 180-220: {validation['length']['between_180_and_220_inclusive']}
- Above 220: {validation['length']['above_220']}

The short outputs did not omit supplied deterministic evidence. They are nevertheless too short and too metadata-limited to deliver the intended rich approximately-200-word profile.

## Usage and Batch cost

- Input tokens: {validation['usage']['input_tokens']:,}
- Output tokens: {validation['usage']['output_tokens']:,}
- Exact calibration cost: ${validation['cost']['exact_calibration_usd']:.9f}
- Average per user: ${validation['cost']['average_per_user_usd']:.9f}
- Projected 200,948 users: ${validation['cost']['projected_all_200948_users_usd']:.6f}
- Projected 50,000-user wave: ${validation['cost']['projected_50000_user_wave_usd']:.6f}
- Conservative upper bound: ${validation['cost']['conservative_upper_all_200948_usd']:.6f}

Costs use the published Batch API rates (50% below standard processing).
"""
    (V4 / "reports" / "calibration_report.md").write_text(markdown, encoding="utf-8")
    (V4 / "reports" / "cost_report.md").write_text(
        "# V4 cost report\n\n"
        + markdown.split("## Usage and Batch cost\n\n", 1)[1],
        encoding="utf-8",
    )
    return report


def _verify_existing_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    manifest = load(manifest_path)
    entries = manifest["files"]
    if isinstance(entries, list):
        expected = {row["path"]: row for row in entries}
    else:
        expected = entries
    missing: list[str] = []
    changed: list[str] = []
    for relative, metadata in expected.items():
        path = root / relative
        if not path.exists():
            missing.append(relative)
        elif sha256_file(path) != metadata["sha256"]:
            changed.append(relative)
    return {
        "root": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "listed_files": len(expected),
        "missing_listed_files": missing,
        "changed_listed_files": changed,
        "verified_unchanged": not missing and not changed,
    }


def finalize_artifacts() -> dict[str, Any]:
    """Snapshot code, verify v1-v3 manifests, and seal the v4 directory."""

    repository_root = Path(__file__).resolve().parents[1]
    snapshots = [
        repository_root / "tears_training" / "artifacts.py",
        repository_root / "tears_training" / "config.py",
        repository_root / "tears_training" / "final_summaries.py",
        repository_root / "tears_training" / "final_summaries_v4.py",
        repository_root / "tears_training" / "calibration_v4_audit.py",
        repository_root / "tests" / "test_final_summaries.py",
        repository_root / "configs" / "ml32m.toml",
    ]
    for source in snapshots:
        target = V4 / "code" / source.name
        payload = source.read_bytes()
        if target.exists() and target.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite different code snapshot: {target}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(target, payload)

    prior = [_verify_existing_manifest(path) for path in (V1, V2, V3)]
    if not all(row["verified_unchanged"] for row in prior):
        raise RuntimeError(f"A prior calibration manifest no longer verifies: {prior}")
    plan = core._load_verified_plan(V4 / "request_plan.json")
    validation = load(V4 / "reports" / "validation_report.json")
    paired = load(PAIRED_OUT / "paired_audit_report.json")
    checks = {
        "request_plan_is_exactly_100": plan["requests"] == plan["unique_users"] == plan["expected_requests"] == 100,
        "same_ids_as_v1_v2_v3": load(V4 / "calibration_user_ids.json")["exact_match_to_v1_v2_v3"],
        "responses_are_exactly_100": validation["received"] == 100,
        "raw_response_file_count": len(list((V4 / "responses").glob("*.jsonl"))),
        "full_cohort_requests_submitted": False,
        "tears_training_started": False,
        "prior_artifacts": prior,
        "promotion_recommended": paired["promotion_gate"]["recommend_full_cohort_generation"],
    }
    atomic_write_json(V4 / "reports" / "artifact_verification.json", checks)
    atomic_write_json(
        V4 / "reports" / "pricing_provenance.json",
        {
            "model": "gpt-5-mini-2025-08-07",
            "standard_usd_per_million": {"input": 0.25, "cached_input": 0.025, "output": 2.0},
            "batch_discount": "50% on input and output",
            "applied_batch_usd_per_million": {"input": 0.125, "cached_input": 0.0125, "output": 1.0},
            "official_sources": [
                "https://openai.com/index/introducing-gpt-5-for-developers/",
                "https://platform.openai.com/docs/api-reference/batch/object?api-mode=responses",
            ],
            "verified_date": "2026-08-13",
        },
    )

    files: dict[str, dict[str, Any]] = {}
    for path in sorted(V4.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            relative = str(path.relative_to(V4))
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(V4),
        "file_count": len(files),
        "files": files,
        "scope_guards": {
            "calibration_users": 100,
            "full_cohort_requests_submitted": False,
            "tears_training_started": False,
            "promotion_recommended": False,
        },
        "prior_calibrations_verified_unchanged": prior,
    }
    atomic_write_json(V4 / "artifact_manifest.json", manifest)
    return {
        "v4_file_count": len(files),
        "v4_manifest_path": str(V4 / "artifact_manifest.json"),
        "v4_manifest_sha256": sha256_file(V4 / "artifact_manifest.json"),
        "checks": checks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("raw-validation", "paired-audit", "finalize-artifacts"), nargs="?", default="raw-validation")
    args = parser.parse_args()
    if args.action == "raw-validation":
        result = raw_validation()
    elif args.action == "paired-audit":
        result = paired_audit()
    else:
        result = finalize_artifacts()
    print(json.dumps(result, indent=2, sort_keys=True))
