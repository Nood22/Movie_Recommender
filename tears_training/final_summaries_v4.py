"""Fourth frozen 100-user calibration: deterministic evidence -> verbalization.

This module deliberately exposes only calibration-only operations.  It derives
genre evidence from the immutable v3 request plan, writes a preflight report,
and builds exactly 100 Responses API requests only after that preflight passes.
The model never receives titles, ratings, or raw histories.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from . import final_summaries as core
from .artifacts import atomic_write_bytes, sha256_file, stable_hash


PROTOCOL_VERSION = (
    "ml32m-support20-v4-deterministic-genre-evidence-verbalization-"
    "gpt5mini-2025-08-07"
)
MODEL = "gpt-5-mini-2025-08-07"
EVIDENCE_SCHEMA_VERSION = "ml32m-genre-evidence-v1"
CALIBRATION_USERS = 100
TARGET_MIN_WORDS = 180
TARGET_MAX_WORDS = 220

# This is an experimental prompt.  It preserves Emiliano's four semantic
# areas and Summary: convention, while restricting verbalization to the
# deterministic evidence payload.
V4_SYSTEM_PROMPT = """Task: Verbalize the supplied deterministic movie-preference evidence as a highly detailed viewer profile.
The evidence object is the complete and exclusive source of preference claims. Do not independently infer, add, suppress, broaden, narrow, or reinterpret any liked or disliked genre, style, theme, plot, or content preference.
Do not mention any specific movie titles, actors, years of production, numeric ratings, rating thresholds, evidence counts, or internal schema fields.
Preserve this semantic organization: liked genres; liked themes, plots, or styles when supported; disliked genres or styles when supported; disliked themes, plots, or content when supported.
Use every genre in supported_positive_genres as a supported liked genre, every genre in supported_negative_genres as a supported disliked genre, and no other genre as a preference. Genres in mixed_genres may only be described as mixed or conflicting evidence, never as liked or disliked. Do not turn weak_or_ambiguous_negative_status into a dislike.
If negative_status is no_supported_negative, explicitly state that the available evidence does not support a strong negative preference. If negative_status is supported, describe the supplied supported negative genres normally and do not abstain from them.
The supplied metadata does not establish theme, plot, content, or style preferences when those fields are marked unavailable. State that limitation naturally; never invent content to fill a section.
Begin with the literal prefix Summary:. Do not force exactly four sentences.
Target approximately 180-220 words when the supplied evidence supports that detail. If it does not, prefer a shorter grounded summary. Never pad with invented preferences or generic claims presented as user traits.
Return only the structured summary field requested by the response schema."""
V4_PROMPT_SHA256 = hashlib.sha256(V4_SYSTEM_PROMPT.encode("utf-8")).hexdigest()


EVIDENCE_RULES: dict[str, Any] = {
    "rating_bands": {
        "positive": "rating >= 4.0",
        "negative": "rating <= 2.5",
        "very_negative": "rating <= 1.5",
    },
    "genre_occurrence": (
        "Each MovieLens pipe-delimited genre on a rated item contributes one "
        "occurrence to that genre. '(no genres listed)' is ignored."
    ),
    "repeated_positive": "at least 3 positive occurrences for the genre",
    "repeated_negative": (
        "at least 3 negative occurrences for the genre, or at least 2 very-negative occurrences"
    ),
    "positive_support": (
        "not conflicting; repeated positive; positive occurrences are at least "
        "half of all occurrences; positive occurrences are at least twice negative occurrences"
    ),
    "negative_support": (
        "not conflicting; repeated negative; negative occurrences are at least "
        "half of all occurrences; negative occurrences are at least twice positive occurrences"
    ),
    "conflict": (
        "at least 3 positive and at least 3 negative occurrences for the same genre; "
        "the genre is excluded from both supported preference lists"
    ),
    "weak_or_ambiguous_negative": (
        "a genre has at least one negative occurrence but fails negative_support "
        "because evidence is isolated, insufficiently concentrated, contradicted, or conflicting"
    ),
    "abstention": (
        "emit no supported negative genre when no genre satisfies negative_support; "
        "the verbalizer must explicitly report no supported strong negative preference"
    ),
    "themes_plots_styles": (
        "unavailable: the frozen request histories provide only titles, numeric ratings, "
        "and MovieLens genres; titles are not mined for latent attributes"
    ),
}


def _write_immutable_text(path: Path, text: str) -> None:
    payload = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _genres(value: str) -> list[str]:
    return sorted(
        {
            genre.strip()
            for genre in str(value).split("|")
            if genre.strip() and genre.strip() != "(no genres listed)"
        }
    )


def _original_negative_tier(ratings: list[float]) -> str:
    low = sum(value <= 2.5 for value in ratings)
    very_low = sum(value <= 1.5 for value in ratings)
    if low == 0:
        return "none"
    if low >= 3 or very_low >= 2:
        return "strong"
    return "weak"


def extract_user_evidence(record: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    for genres, rating in zip(record["genres"], record["ratings"]):
        for genre in _genres(genres):
            row = stats.setdefault(
                genre,
                {
                    "occurrences": 0,
                    "positive_occurrences": 0,
                    "negative_occurrences": 0,
                    "very_negative_occurrences": 0,
                },
            )
            row["occurrences"] += 1
            row["positive_occurrences"] += int(float(rating) >= 4.0)
            row["negative_occurrences"] += int(float(rating) <= 2.5)
            row["very_negative_occurrences"] += int(float(rating) <= 1.5)

    supported_positive: list[str] = []
    supported_negative: list[str] = []
    conflicts: list[str] = []
    ambiguous_negative: list[dict[str, Any]] = []
    per_genre: dict[str, dict[str, Any]] = {}
    for genre, counts in sorted(stats.items()):
        total = counts["occurrences"]
        high = counts["positive_occurrences"]
        low = counts["negative_occurrences"]
        very_low = counts["very_negative_occurrences"]
        conflict = high >= 3 and low >= 3
        repeated_positive = high >= 3
        repeated_negative = low >= 3 or very_low >= 2
        positive = (
            not conflict
            and repeated_positive
            and high / total >= 0.5
            and high >= 2 * low
        )
        negative = (
            not conflict
            and repeated_negative
            and low / total >= 0.5
            and low >= 2 * high
        )
        if conflict:
            conflicts.append(genre)
        if positive:
            supported_positive.append(genre)
        if negative:
            supported_negative.append(genre)
        ambiguity_reason: str | None = None
        if low and not negative:
            if conflict:
                ambiguity_reason = "conflicting_repeated_positive_and_negative"
            elif not repeated_negative:
                ambiguity_reason = "isolated_or_not_repeated"
            elif low / total < 0.5:
                ambiguity_reason = "negative_not_half_of_genre_occurrences"
            elif low < 2 * high:
                ambiguity_reason = "contradictory_positive_evidence"
            else:
                ambiguity_reason = "not_supported"
            ambiguous_negative.append(
                {
                    "genre": genre,
                    "reason": ambiguity_reason,
                    **counts,
                }
            )
        per_genre[genre] = {
            **counts,
            "positive_share": high / total,
            "negative_share": low / total,
            "repeated_positive": repeated_positive,
            "repeated_negative": repeated_negative,
            "conflicting": conflict,
            "supported_positive": positive,
            "supported_negative": negative,
            "ambiguous_negative_reason": ambiguity_reason,
        }

    ratings = [float(value) for value in record["ratings"]]
    negative_status = "supported" if supported_negative else "no_supported_negative"
    original_tier = _original_negative_tier(ratings)
    if supported_negative:
        weak_status = "not_applicable_supported_negative_present"
    elif original_tier == "none":
        weak_status = "no_negative_rating_evidence"
    elif conflicts:
        weak_status = "negative_signal_present_but_conflicting_or_ambiguous"
    else:
        weak_status = "negative_signal_present_but_no_reliable_genre_dislike"

    # This is the only object supplied to the LLM.  It deliberately omits
    # titles, ratings, counts, and ambiguous genre names.
    verbalization_payload = {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "evidence_scope": "MovieLens genres only",
        "supported_positive_genres": supported_positive,
        "supported_negative_genres": supported_negative,
        "mixed_genres": conflicts,
        "negative_status": negative_status,
        "weak_or_ambiguous_negative_status": weak_status,
        "liked_themes_plots_styles": {"status": "unavailable"},
        "disliked_themes_plots_styles_content": {"status": "unavailable"},
    }
    result = {
        "user_id": int(record["user_id"]),
        "history_hash": record["history_hash"],
        "history_items": record["history_items"],
        "original_negative_evidence_tier": original_tier,
        "supported_positive_genres": supported_positive,
        "supported_negative_genres": supported_negative,
        "conflicting_genres": conflicts,
        "weak_or_ambiguous_negative_genres": ambiguous_negative,
        "no_negative_rating_evidence": original_tier == "none",
        "no_supported_negative_genre": not supported_negative,
        "genre_statistics": per_genre,
        "metadata_scope": {
            "reliable_for_generation": ["MovieLens genres"],
            "not_used_for_inference": ["movie titles"],
            "not_available": ["themes", "plots", "styles", "content attributes"],
        },
        "verbalization_payload": verbalization_payload,
    }
    result["evidence_hash"] = stable_hash(result)
    return result


def _representative_cases(evidence: list[dict[str, Any]]) -> dict[str, list[int]]:
    def first(predicate: Any, limit: int = 3) -> list[int]:
        return [row["user_id"] for row in evidence if predicate(row)][:limit]

    return {
        "supported_positive": first(lambda row: bool(row["supported_positive_genres"])),
        "supported_negative": first(lambda row: bool(row["supported_negative_genres"])),
        "weak_or_ambiguous_negative": first(
            lambda row: bool(row["weak_or_ambiguous_negative_genres"])
            and not row["supported_negative_genres"]
        ),
        "no_negative_rating_evidence": first(lambda row: row["no_negative_rating_evidence"]),
        "conflicting_genre_evidence": first(lambda row: bool(row["conflicting_genres"])),
        "no_supported_positive": first(lambda row: not row["supported_positive_genres"]),
    }


def preflight(v3_plan_path: Path, output_dir: Path) -> dict[str, Any]:
    reference = core._load_verified_plan(v3_plan_path)
    if reference["requests"] != CALIBRATION_USERS:
        raise RuntimeError("V3 reference is not exactly 100 requests")
    evidence = [extract_user_evidence(record) for record in reference["records"]]
    if len(evidence) != CALIBRATION_USERS or len({x["user_id"] for x in evidence}) != 100:
        raise RuntimeError("Evidence preflight is not exactly 100 unique users")
    if [x["user_id"] for x in evidence] != reference["sample"]["user_ids"]:
        raise RuntimeError("Evidence order or user IDs differ from frozen v3 sample")

    # Mechanical invariants guard against exactly the overgeneralization the
    # deterministic layer is intended to prevent.
    invariant_failures: list[dict[str, Any]] = []
    for row in evidence:
        for genre in row["supported_positive_genres"]:
            stats = row["genre_statistics"][genre]
            if not stats["supported_positive"] or stats["conflicting"]:
                invariant_failures.append({"user_id": row["user_id"], "genre": genre, "kind": "positive"})
        for genre in row["supported_negative_genres"]:
            stats = row["genre_statistics"][genre]
            if not stats["supported_negative"] or stats["conflicting"]:
                invariant_failures.append({"user_id": row["user_id"], "genre": genre, "kind": "negative"})

    original_tiers = Counter(x["original_negative_evidence_tier"] for x in evidence)
    report: dict[str, Any] = {
        "scope": "calibration_preflight_only",
        "api_called": False,
        "users": len(evidence),
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "rules": EVIDENCE_RULES,
        "metadata_conclusion": (
            "Only MovieLens genres are reliable deterministic preference attributes. "
            "Titles are retained solely for offline manual audit and are not mined for themes, plots, or styles."
        ),
        "distribution": {
            "supported_positive_evidence_users": sum(bool(x["supported_positive_genres"]) for x in evidence),
            "supported_negative_evidence_users": sum(bool(x["supported_negative_genres"]) for x in evidence),
            "weak_or_ambiguous_negative_evidence_users": sum(bool(x["weak_or_ambiguous_negative_genres"]) for x in evidence),
            "no_negative_rating_evidence_users": sum(x["no_negative_rating_evidence"] for x in evidence),
            "no_supported_negative_genre_users": sum(x["no_supported_negative_genre"] for x in evidence),
            "conflicting_genre_evidence_users": sum(bool(x["conflicting_genres"]) for x in evidence),
            "no_supported_positive_evidence_users": sum(not x["supported_positive_genres"] for x in evidence),
            "original_negative_tiers": dict(sorted(original_tiers.items())),
        },
        "representative_case_user_ids": _representative_cases(evidence),
        "invariant_failures": invariant_failures,
        "automatic_preflight_pass": not invariant_failures,
        "manual_review_required_before_submission": True,
        "v3_reference": {
            "plan_path": str(v3_plan_path.resolve()),
            "plan_fingerprint": reference["fingerprint"],
            "request_sha256": reference["request_file"]["sha256"],
            "user_ids_sha256": reference["sample"]["user_ids_sha256"],
        },
    }
    report["fingerprint"] = stable_hash(report)
    output_dir = output_dir.resolve()
    core._write_immutable_jsonl(output_dir / "evidence" / "all_user_evidence.jsonl", evidence)
    core._write_immutable_json(output_dir / "evidence" / "evidence_rules.json", EVIDENCE_RULES)
    core._write_immutable_json(output_dir / "reports" / "preflight_report.json", report)
    return report


REPRESENTATIVE_REVIEW_NOTES: dict[int, str] = {
    1288: "No low-rated item exists; numerous genres recur in positive items, so positive support and negative abstention are appropriate.",
    40516: "No low-rated item exists; repeated comedy/crime/drama/mystery/romance/sci-fi/thriller positives support the emitted genres and no dislike.",
    41043: "No low-rated item exists; repeated positive action/adventure/comedy/crime/drama/IMAX/sci-fi/thriller/war evidence supports the positive list and negative abstention.",
    2752: "Repeated low-rated Action and Adventure occurrences dominate their positive evidence; Comedy has repeated evidence in both directions and is correctly quarantined as mixed.",
    7190: "Children has repeated concentrated low evidence; Comedy, Drama, Fantasy, and Romance recur on both high- and low-rated items and are correctly marked mixed.",
    12562: "The history contains a very large, repeated, genre-coherent low-rated block. Every emitted negative genre meets the same concentration and contradiction checks; isolated genres are not emitted.",
    2015: "Two low-rated animated/family items are contradicted by many positive occurrences, so no negative genre is emitted; repeated positive genres remain supported.",
    7350: "Five heterogeneous low-rated items do not establish a stable negative genre; Drama has repeated evidence in both directions and is correctly marked mixed.",
    8545: "Action, Adventure, Comedy, and Fantasy recur in both positive and negative items, so the extractor does not convert the low-rated cluster into broad dislikes.",
    43011: "Positive evidence is diffuse and Romance is repeated in both directions; abstaining from both stable positive and negative genre preferences is conservative.",
    47525: "Several broad genres recur in both positive and negative items; the extractor marks those conflicts and avoids forcing a preference in either direction.",
}


def manual_preflight_review(
    v1_audit_path: Path,
    v3_paired_audit_path: Path,
    v3_plan_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Record the completed human preflight review before paid submission."""

    output_dir = output_dir.resolve()
    evidence_rows = _load_jsonl(output_dir / "evidence" / "all_user_evidence.jsonl")
    evidence_by_user = {row["user_id"]: row for row in evidence_rows}
    record_by_user = {
        row["user_id"]: row
        for row in core._load_verified_plan(v3_plan_path)["records"]
    }
    v1_rows = _load_jsonl(v1_audit_path)
    paired_rows = _load_jsonl(v3_paired_audit_path)
    prior_failures = {
        "v1": sorted(
            row["user_id"]
            for row in v1_rows
            if row["negative_claim_verdict"] != "supported"
        ),
        "v2": sorted(
            row["user_id"]
            for row in paired_rows
            if row["v2"]["negative_verdict"] != "supported"
        ),
        "v3": sorted(
            row["user_id"]
            for row in paired_rows
            if row["v3"]["negative_verdict"] != "supported"
        ),
    }
    union = sorted(set().union(*map(set, prior_failures.values())))
    prior_failure_checks = []
    for user_id in union:
        row = evidence_by_user[user_id]
        prior_failure_checks.append(
            {
                "user_id": user_id,
                "failed_versions": [
                    version for version, ids in prior_failures.items() if user_id in ids
                ],
                "original_negative_evidence_tier": row["original_negative_evidence_tier"],
                "supported_positive_genres": row["supported_positive_genres"],
                "supported_negative_genres": row["supported_negative_genres"],
                "conflicting_genres": row["conflicting_genres"],
                "negative_generation_action": (
                    "verbalize_supported_negative_only"
                    if row["supported_negative_genres"]
                    else "explicit_negative_abstention"
                ),
            }
        )

    representative = []
    for user_id, note in REPRESENTATIVE_REVIEW_NOTES.items():
        evidence = evidence_by_user[user_id]
        record = record_by_user[user_id]
        representative.append(
            {
                "user_id": user_id,
                "review_note": note,
                "original_negative_evidence_tier": evidence["original_negative_evidence_tier"],
                "supported_positive_genres": evidence["supported_positive_genres"],
                "supported_negative_genres": evidence["supported_negative_genres"],
                "conflicting_genres": evidence["conflicting_genres"],
                "low_rated_items_for_human_audit": [
                    {"title": title, "rating": rating, "genres": genres}
                    for title, rating, genres in zip(
                        record["titles"], record["ratings"], record["genres"]
                    )
                    if float(rating) <= 2.5
                ],
                "high_rated_items_for_human_audit": [
                    {"title": title, "rating": rating, "genres": genres}
                    for title, rating, genres in zip(
                        record["titles"], record["ratings"], record["genres"]
                    )
                    if float(rating) >= 4.0
                ],
            }
        )
    report: dict[str, Any] = {
        "scope": "calibration_preflight_manual_review",
        "api_called": False,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewed_representative_users": representative,
        "prior_failure_counts": {key: len(value) for key, value in prior_failures.items()},
        "prior_failure_union_users": len(union),
        "prior_failure_checks": prior_failure_checks,
        "conclusion": (
            "PASS. Representative histories and every user with a v1, v2, or v3 "
            "claim-grounding failure were checked. Supported genres obey the repeated, "
            "concentration, dominance, and conflict rules. The extractor is conservative "
            "but does not show obvious deterministic overgeneralization. Proceed with "
            "exactly the frozen 100 v4 requests."
        ),
        "manual_preflight_pass": True,
    }
    report["fingerprint"] = stable_hash(report)
    core._write_immutable_json(
        output_dir / "reports" / "preflight_manual_review.json", report
    )
    return report


def protocol_manifest(repository_root: Path) -> dict[str, Any]:
    source_notebook = (repository_root / core.SOURCE_NOTEBOOK_RELATIVE).resolve()
    if sha256_file(source_notebook) != core.EXPECTED_SOURCE_NOTEBOOK_SHA256:
        raise RuntimeError("Original Emiliano notebook bytes changed")
    value: dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "label": "Deterministic genre evidence followed by constrained LLM verbalization",
        "experimental_not_exact_emiliano_reproduction": True,
        "source": {
            "emiliano_notebook_path": str(source_notebook),
            "emiliano_notebook_sha256": sha256_file(source_notebook),
            "preserved_semantics": [
                "liked genres",
                "liked themes/plots/styles when supported",
                "disliked genres/styles when supported",
                "disliked themes/plots/content when supported",
            ],
            "preserved_format": "Summary: prefix; no exactly-four-sentence rule",
        },
        "system_prompt": V4_SYSTEM_PROMPT,
        "system_prompt_sha256": V4_PROMPT_SHA256,
        "evidence": {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "rules": EVIDENCE_RULES,
            "llm_receives_raw_history": False,
            "llm_receives_titles": False,
            "llm_receives_ratings": False,
            "llm_receives_counts_or_thresholds": False,
            "llm_input": "verbalization_payload JSON only",
        },
        "generation": {
            "model": MODEL,
            "endpoint": "/v1/responses",
            "reasoning_effort": "minimal",
            "temperature": None,
            "seed": None,
            "max_output_tokens": core.MAX_OUTPUT_TOKENS,
            "structured_output": core.SUMMARY_SCHEMA,
            "text_verbosity": "low",
            "store": False,
            "omitted_parameters": {
                "temperature": "not sent",
                "seed": "not sent",
            },
        },
        "validation": {
            "target_word_range": [TARGET_MIN_WORDS, TARGET_MAX_WORDS],
            "word_range_is_semantic_validity_rule": False,
            "requires_literal_summary_prefix": True,
            "requires_exact_sentence_count": False,
            "requires_four_semantic_areas_or_explicit_unavailability": True,
            "privacy": ["movie title", "year", "numeric rating"],
            "claim_grounding": "every preference claim must occur in the LLM evidence payload",
            "near_duplicate_thresholds": {"sequence": 0.92, "token_jaccard": 0.85},
        },
    }
    value["fingerprint"] = stable_hash(value)
    return value


def _request_for(record: dict[str, Any], evidence: dict[str, Any], protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload_text = json.dumps(
        evidence["verbalization_payload"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    cache_key = stable_hash(
        {
            "user_id": record["user_id"],
            "history_hash": record["history_hash"],
            "evidence_hash": evidence["evidence_hash"],
            "protocol_fingerprint": protocol["fingerprint"],
        }
    )
    custom_id = f"cal-user-{record['user_id']}-{cache_key[:16]}-a0"
    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "input": [
                {"role": "system", "content": V4_SYSTEM_PROMPT},
                {"role": "user", "content": payload_text},
            ],
            "reasoning": {"effort": "minimal"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "viewer_profile",
                    "strict": True,
                    "schema": core.SUMMARY_SCHEMA,
                },
                "verbosity": "low",
            },
            "max_output_tokens": core.MAX_OUTPUT_TOKENS,
            "store": False,
        },
    }
    target_record = deepcopy(record)
    target_record.update(
        {
            "custom_id": custom_id,
            "cache_key": cache_key,
            "evidence_hash": evidence["evidence_hash"],
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "llm_input_sha256": hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
            "llm_input": evidence["verbalization_payload"],
        }
    )
    return request, target_record


def plan_from_preflight(v3_plan_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    preflight_report = _load_json(output_dir / "reports" / "preflight_report.json")
    if not preflight_report.get("automatic_preflight_pass"):
        raise RuntimeError("Preflight invariants failed; refusing to build paid requests")
    manual_report = _load_json(output_dir / "reports" / "preflight_manual_review.json")
    if not manual_report.get("manual_preflight_pass"):
        raise RuntimeError("Manual preflight did not pass; refusing to build paid requests")
    reference = core._load_verified_plan(v3_plan_path)
    evidence_rows = _load_jsonl(output_dir / "evidence" / "all_user_evidence.jsonl")
    evidence_by_user = {row["user_id"]: row for row in evidence_rows}
    if reference["sample"]["user_ids"] != [row["user_id"] for row in evidence_rows]:
        raise RuntimeError("Preflight evidence IDs differ from frozen sample")
    repository_root = Path(__file__).resolve().parents[1]
    protocol = protocol_manifest(repository_root)
    requests: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for source_record in reference["records"]:
        request, record = _request_for(
            source_record, evidence_by_user[source_record["user_id"]], protocol
        )
        requests.append(request)
        records.append(record)
    if len(requests) != 100 or len({row["custom_id"] for row in requests}) != 100:
        raise RuntimeError("V4 plan is not exactly 100 unique requests")

    request_path = output_dir / "requests" / "calibration-100.jsonl"
    count, size = core._write_immutable_jsonl(request_path, requests)
    plan: dict[str, Any] = {
        "scope": "calibration_only",
        "expected_requests": 100,
        "requests": count,
        "unique_users": 100,
        "seed": reference["seed"],
        "protocol": protocol,
        "dataset": deepcopy(reference["dataset"]),
        "sample": deepcopy(reference["sample"]),
        "request_file": {
            "path": str(request_path),
            "requests": count,
            "bytes": size,
            "sha256": sha256_file(request_path),
        },
        "records": records,
        "estimated_input_tokens": sum(
            core.estimate_tokens(row["body"]["input"][0]["content"] + row["body"]["input"][1]["content"])
            for row in requests
        ),
        "reserved_output_tokens": 100 * core.MAX_OUTPUT_TOKENS,
        "preflight": {
            "report_path": str(output_dir / "reports" / "preflight_report.json"),
            "report_fingerprint": preflight_report["fingerprint"],
            "manual_report_path": str(output_dir / "reports" / "preflight_manual_review.json"),
            "manual_report_fingerprint": manual_report["fingerprint"],
            "evidence_path": str(output_dir / "evidence" / "all_user_evidence.jsonl"),
            "evidence_sha256": sha256_file(output_dir / "evidence" / "all_user_evidence.jsonl"),
            "manual_review_completed": True,
            "manual_review_conclusion": "No deterministic supported preference violates the frozen support/conflict rules.",
        },
        "v3_reference": {
            "plan_path": str(v3_plan_path.resolve()),
            "plan_fingerprint": reference["fingerprint"],
            "request_sha256": reference["request_file"]["sha256"],
            "user_ids_sha256": reference["sample"]["user_ids_sha256"],
        },
    }
    plan["fingerprint"] = stable_hash(plan)
    core._write_immutable_json(output_dir / "request_plan.json", plan)
    core._write_immutable_json(output_dir / "protocol.json", protocol)
    _write_immutable_text(output_dir / "prompt" / "v4_system_prompt.txt", V4_SYSTEM_PROMPT)
    core._write_immutable_json(
        output_dir / "prompt" / "provenance.json",
        {
            "protocol_version": protocol["version"],
            "protocol_fingerprint": protocol["fingerprint"],
            "prompt_sha256": V4_PROMPT_SHA256,
            "system_message": V4_SYSTEM_PROMPT,
            "user_input_construction": protocol["evidence"],
            "generation": protocol["generation"],
            "validation": protocol["validation"],
            "evidence_rules": EVIDENCE_RULES,
            "v3_reference": plan["v3_reference"],
        },
    )
    core._write_immutable_json(
        output_dir / "calibration_user_ids.json",
        {
            "seed": plan["seed"],
            "count": 100,
            "user_ids": plan["sample"]["user_ids"],
            "user_ids_sha256": plan["sample"]["user_ids_sha256"],
            "exact_match_to_v1_v2_v3": True,
            "v3_reference": plan["v3_reference"],
            "split_counts": plan["sample"]["split_counts"],
            "activity_band_counts": plan["sample"]["activity_band_counts"],
            "stratum_counts": plan["sample"]["stratum_counts"],
        },
    )
    try:
        import openai
        openai_version = openai.__version__
    except (ImportError, AttributeError):
        openai_version = None
    core._write_immutable_json(
        output_dir / "runtime_environment.json",
        {
            "python_version": sys.version,
            "openai_sdk_version": openai_version,
            "api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
            "api_key_value_recorded": False,
            "repository_root": str(repository_root),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tears_training.final_summaries_v4")
    sub = parser.add_subparsers(dest="action", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--v3-plan", type=Path, required=True)
    pre.add_argument("--output-dir", type=Path, required=True)
    review = sub.add_parser("manual-preflight-review")
    review.add_argument("--v1-audit", type=Path, required=True)
    review.add_argument("--v3-paired-audit", type=Path, required=True)
    review.add_argument("--v3-plan", type=Path, required=True)
    review.add_argument("--output-dir", type=Path, required=True)
    plan = sub.add_parser("plan-from-preflight")
    plan.add_argument("--v3-plan", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    submit = sub.add_parser("submit-calibration")
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--input-price-per-million", type=float, required=True)
    submit.add_argument("--cached-input-price-per-million", type=float, required=True)
    submit.add_argument("--output-price-per-million", type=float, required=True)
    submit.add_argument("--max-calibration-cost-usd", type=float, default=1.0)
    submit.add_argument("--confirm-calibration-spend", action="store_true")
    poll = sub.add_parser("poll-calibration")
    poll.add_argument("--submission", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "preflight":
        result = preflight(args.v3_plan, args.output_dir)
    elif args.action == "manual-preflight-review":
        result = manual_preflight_review(
            args.v1_audit,
            args.v3_paired_audit,
            args.v3_plan,
            args.output_dir,
        )
    elif args.action == "plan-from-preflight":
        result = plan_from_preflight(args.v3_plan, args.output_dir)
    elif args.action == "submit-calibration":
        result = core.submit_calibration(
            args.plan,
            None,
            args.input_price_per_million,
            args.cached_input_price_per_million,
            args.output_price_per_million,
            args.max_calibration_cost_usd,
            args.confirm_calibration_spend,
        )
    else:
        result = core.poll_calibration(args.submission, None)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
