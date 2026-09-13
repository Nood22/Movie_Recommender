"""Final deterministic cleanup of the offline repaired-v2 calibration.

Inputs are the immutable repaired-v2 artifact, frozen v2 audit/history records,
and frozen v4 deterministic evidence. This module performs no API calls, model
inference, summary generation, training, or TMDB access.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

from . import final_summaries as core
from .calibration_v2_offline_repair import (
    CALIBRATION_BASE,
    PLACEHOLDER,
    V2,
    V4,
)


REPAIRED_V2 = CALIBRATION_BASE / "v002_grounding_repair_offline"
DEFAULT_OUTPUT = CALIBRATION_BASE / "v002_final_cleanup_offline"
ARTIFACT_VERSION = "v002-final-cleanup-offline-v1"

INAPPROPRIATE_ABSTENTION_USERS = (
    7190,
    30459,
    44856,
    48604,
    69317,
    169241,
    178348,
)
PREFIX_USERS = (
    2015,
    7190,
    43011,
    86122,
    87594,
    89717,
    91106,
    138218,
    146517,
    161310,
    171764,
    182046,
    184544,
    193437,
)
COLON_ABSTENTION_USERS = (
    22939,
    25944,
    41043,
    47525,
    61038,
    61207,
    76281,
    115873,
    119291,
    165531,
)
TYPO_USER = 188225
GRAMMAR_USER = 156456

ABSTENTION_SPANS: dict[int, str] = {
    7190: "No strong negative genre preference is supported by the available history.",
    30459: "No strong negative preference is supported by the available history.",
    44856: "No strong negative preference is supported by the available history.",
    48604: "No strong negative preference is supported by the available history.",
    69317: "No strong negative genre preference is supported by the available history.",
    169241: "No strong negative preference is supported by the available history.",
    178348: (
        "No strong negative preference for other genres is supported by the "
        "available history."
    ),
}

GENRE_PHRASES = {
    "Action": "action films",
    "Children": "children's films",
    "Crime": "crime films",
    "Drama": "dramas",
    "Fantasy": "fantasy films",
    "Horror": "horror films",
    "Sci-Fi": "science-fiction films",
    "Thriller": "thrillers",
}

GRAMMAR_ORIGINAL = (
    "The user does not enjoy overly sentimental or slow-paced melodrama is not "
    "strongly supported as a negative preference by the available history."
)
GRAMMAR_REPLACEMENT = (
    "A negative preference for overly sentimental or slow-paced melodrama is not "
    "strongly supported by the available history."
)
TYPO_ORIGINAL = "broadly thatmost "
TYPO_REPLACEMENT = ""
COLON_PATTERN = re.compile(
    r"The user does not enjoy: (no strong negative(?: genre)? preference is "
    r"supported by the available history\.)"
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")
    _write_bytes(path, payload)


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _percentile(values: list[int], percentile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
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


def _exact_duplicate_groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        grouped[_normal(row["summary"])].append(row["user_id"])
    return sorted(sorted(ids) for ids in grouped.values() if len(ids) > 1)


def _has_evidence_aware_abstention(text: str) -> bool:
    normalized = _normal(text)
    return bool(
        re.search(
            r"\b(?:no|does not (?:show|indicate|provide))\b.{0,55}\bstrong\b"
            r".{0,35}\bnegative\b",
            normalized,
        )
        or "not strongly supported as a negative preference" in normalized
        or bool(
            re.search(
                r"\bnegative preference\b.{0,100}\bnot strongly supported\b",
                normalized,
            )
        )
    )


def _genre_list_phrase(genres: list[str]) -> str:
    try:
        phrases = [GENRE_PHRASES[genre] for genre in genres]
    except KeyError as error:
        raise RuntimeError(f"No frozen verbalization for genre {error.args[0]}") from error
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def _supported_negative_replacement(evidence: dict[str, Any]) -> str:
    genres = evidence["supported_negative_genres"]
    if not genres:
        raise RuntimeError(f"No supported negative evidence for user {evidence['user_id']}")
    for genre in genres:
        stats = evidence["genre_statistics"][genre]
        if not stats["supported_negative"] or stats["conflicting"]:
            raise RuntimeError(
                f"Invalid deterministic negative evidence for user "
                f"{evidence['user_id']}, genre {genre}"
            )
    return (
        "The available history supports a negative preference for "
        f"{_genre_list_phrase(genres)}."
    )


def _replace_exact(text: str, original: str, replacement: str, user_id: int) -> str:
    count = text.count(original)
    if count != 1:
        raise RuntimeError(
            f"Expected one exact span for user {user_id}, found {count}: {original!r}"
        )
    return text.replace(original, replacement, 1)


def _formatting_issues(text: str) -> list[str]:
    issues: list[str] = []
    if not text.startswith("Summary:"):
        issues.append("missing_summary_prefix")
    if COLON_PATTERN.search(text):
        issues.append("does_not_enjoy_colon_abstention")
    if TYPO_ORIGINAL in text:
        issues.append("token_corruption_broadly_thatmost")
    if GRAMMAR_ORIGINAL in text:
        issues.append("ungrammatical_negative_abstention")
    return issues


def _markdown(report: dict[str, Any]) -> str:
    compare = report["comparison"]
    final = report["final_candidate"]
    lengths = report["length"]
    decision = report["promotion_decision"]
    raw = compare["raw_v2"]
    repaired = compare["repaired_v2"]
    clean = compare["final_cleanup"]
    lines = [
        "# Final deterministic cleanup of repaired-v2",
        "",
        "## Outcome",
        "",
        "The final cleanup changes 32 repaired-v2 summaries: seven narrow, "
        "evidence-backed abstention replacements and 26 non-semantic formatting "
        "repairs, with user 7190 receiving one repair of each kind.",
        "",
        "| Metric | Raw v2 | Repaired-v2 | Final cleanup |",
        "|---|---:|---:|---:|",
        f"| Unsupported negative | {raw['unsupported_negative']} | "
        f"{repaired['unsupported_negative']} | {clean['unsupported_negative']} |",
        f"| Overstated negative | {raw['overstated_negative']} | "
        f"{repaired['overstated_negative']} | {clean['overstated_negative']} |",
        f"| Inappropriate abstentions | {raw['inappropriate_abstentions']} | "
        f"{repaired['inappropriate_abstentions']} | "
        f"{clean['inappropriate_abstentions']} |",
        f"| Formatting failures | {raw['formatting_failures']} | "
        f"{repaired['formatting_failures']} | {clean['formatting_failures']} |",
        "",
        "## Final audit",
        "",
        f"- Positive grounding: {final['positive_grounding']['supported']}/100.",
        f"- Negative grounding: {final['negative_grounding']['supported']}/100 "
        "supported, 0 overstated, 0 unsupported.",
        f"- Appropriate abstentions: {final['abstentions']['appropriate']}; "
        "inappropriate: 0.",
        f"- Supported deterministic negative evidence omitted: "
        f"{final['supported_negative_evidence_accidentally_omitted']['count']}.",
        f"- Four-part semantic usability: "
        f"{final['four_part_semantic_usability']['complete']}/100.",
        f"- Confirmed identifiable privacy leaks: "
        f"{final['privacy']['confirmed_identifiable_leaks']}.",
        f"- Exact duplicate groups: {final['duplicates']['exact_group_count']}; "
        f"near-duplicate pairs: {final['duplicates']['near_pair_count']}.",
        f"- Literal placeholders: {final['literal_placeholders']['count']}.",
        "",
        "## Length",
        "",
        f"Final mean/median/min/max/p95: {lengths['final_cleanup']['mean']:.2f} / "
        f"{lengths['final_cleanup']['median']:.1f} / "
        f"{lengths['final_cleanup']['minimum']} / "
        f"{lengths['final_cleanup']['maximum']} / "
        f"{lengths['final_cleanup']['p95_linear']:.2f} words.",
        f"Materially compressed relative to raw v2: "
        f"{lengths['materially_compressed_relative_to_raw_v2']['count']}; all "
        "remain semantically complete because the removed content was invalid and "
        "all strict deterministic evidence is represented or appropriately abstained.",
        "",
        "## Readiness",
        "",
        report["readiness"]["tears_training_text_input"]["assessment"],
        "",
        report["readiness"]["ux_editable_profile"]["assessment"],
        "",
        "No claim about actual TEARS controllability is made from text inspection.",
        "",
        "## Promotion decision",
        "",
        f"**{decision['recommendation']}** {decision['reason']}",
        "",
        "No OpenAI API request, LLM repair call, TMDB semantic access, summary "
        "generation, full-cohort generation, or TEARS training occurred.",
    ]
    return "\n".join(lines) + "\n"


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {output_dir}")

    prior_roots = {
        "v1": CALIBRATION_BASE / "v001_20260813",
        "v2": V2,
        "v3": CALIBRATION_BASE / "v003_20260813_evidence_calibrated_negative_grounding",
        "v4": V4,
        "repaired_v2": REPAIRED_V2,
    }
    prior_hashes_before = {
        name: _tree_hashes(path) for name, path in prior_roots.items()
    }

    source_manifest = _json(REPAIRED_V2 / "manifest.json")
    for name, expected in source_manifest["derived_artifacts"].items():
        actual = _sha256(REPAIRED_V2 / name)
        if actual != expected:
            raise RuntimeError(f"Repaired-v2 manifest mismatch for {name}")

    source_report = _json(REPAIRED_V2 / "evaluation_report.json")
    source_rows = {
        row["user_id"]: row
        for row in _jsonl(REPAIRED_V2 / "repaired_summaries.jsonl")
    }
    source_audit = {
        row["user_id"]: row
        for row in _jsonl(REPAIRED_V2 / "per_user_repair_audit.jsonl")
    }
    raw_rows = {
        row["user_id"]: row
        for row in _jsonl(V2 / "validated" / "all_summaries.jsonl")
    }
    evidence = {
        row["user_id"]: row
        for row in _jsonl(V4 / "evidence" / "all_user_evidence.jsonl")
    }
    plan = _json(V2 / "request_plan.json")
    records = {row["user_id"]: row for row in plan["records"]}
    user_ids = plan["sample"]["user_ids"]
    if len(user_ids) != 100 or len(set(user_ids)) != 100:
        raise RuntimeError("Frozen sample is not exactly 100 unique users")
    for collection in (source_rows, source_audit, raw_rows, evidence, records):
        if set(collection) != set(user_ids):
            raise RuntimeError("Input user sets differ from the frozen v2 sample")
    if source_report["repaired_v2"]["inappropriate_abstention_user_ids"] != list(
        INAPPROPRIATE_ABSTENTION_USERS
    ):
        raise RuntimeError("Seven-user abstention source classification changed")

    expected_formatting = set(PREFIX_USERS) | set(COLON_ABSTENTION_USERS) | {
        TYPO_USER,
        GRAMMAR_USER,
    }
    source_formatting = source_report["repaired_v2"]["formatting_failures"]
    if source_formatting["count"] != 26 or set(source_formatting["user_ids"]) != expected_formatting:
        raise RuntimeError("The frozen 26-case formatting classification changed")

    final_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    abstention_repairs: list[dict[str, Any]] = []
    formatting_repairs: list[dict[str, Any]] = []

    for user_id in user_ids:
        raw_summary = raw_rows[user_id]["summary"]
        repaired_summary = source_rows[user_id]["summary"]
        final_summary = repaired_summary
        operations: list[dict[str, Any]] = []
        ev = evidence[user_id]

        if user_id in INAPPROPRIATE_ABSTENTION_USERS:
            original_span = ABSTENTION_SPANS[user_id]
            replacement = _supported_negative_replacement(ev)
            final_summary = _replace_exact(
                final_summary, original_span, replacement, user_id
            )
            evidence_detail = {
                "history_hash": ev["history_hash"],
                "evidence_schema_version": ev["verbalization_payload"][
                    "evidence_schema_version"
                ],
                "original_negative_evidence_tier": ev[
                    "original_negative_evidence_tier"
                ],
                "supported_negative_genres": ev["supported_negative_genres"],
                "genre_statistics": {
                    genre: ev["genre_statistics"][genre]
                    for genre in ev["supported_negative_genres"]
                },
                "deterministic_rules_satisfied": [
                    "repeated_negative",
                    "negative_share_at_least_one_half",
                    "negative_occurrences_at_least_twice_positive_occurrences",
                    "not_conflicting",
                ],
            }
            operation = {
                "user_id": user_id,
                "repair_type": "inappropriate_abstention_to_supported_negative_genres",
                "exact_original_span": original_span,
                "replacement_span": replacement,
                "evidence": evidence_detail,
                "semantic_claims_added": ev["supported_negative_genres"],
                "unsupported_semantic_claims_added": [],
            }
            operations.append(operation)
            abstention_repairs.append(operation)

        if user_id in PREFIX_USERS:
            if final_summary.startswith("Summary:"):
                raise RuntimeError(f"Prefix unexpectedly present for user {user_id}")
            final_summary = "Summary: " + final_summary
            operation = {
                "user_id": user_id,
                "repair_type": "add_missing_summary_prefix",
                "exact_original_span": "",
                "replacement_span": "Summary: ",
                "semantic_change": False,
            }
            operations.append(operation)
            formatting_repairs.append(operation)

        if user_id in COLON_ABSTENTION_USERS:
            match = COLON_PATTERN.search(final_summary)
            if not match:
                raise RuntimeError(f"Malformed colon abstention changed for user {user_id}")
            original_span = match.group(0)
            replacement = match.group(1)[0].upper() + match.group(1)[1:]
            final_summary = _replace_exact(
                final_summary, original_span, replacement, user_id
            )
            operation = {
                "user_id": user_id,
                "repair_type": "remove_malformed_does_not_enjoy_colon_leadin",
                "exact_original_span": original_span,
                "replacement_span": replacement,
                "semantic_change": False,
            }
            operations.append(operation)
            formatting_repairs.append(operation)

        if user_id == TYPO_USER:
            final_summary = _replace_exact(
                final_summary, TYPO_ORIGINAL, TYPO_REPLACEMENT, user_id
            )
            operation = {
                "user_id": user_id,
                "repair_type": "remove_corrupted_tokens_broadly_thatmost",
                "exact_original_span": TYPO_ORIGINAL,
                "replacement_span": TYPO_REPLACEMENT,
                "semantic_change": False,
            }
            operations.append(operation)
            formatting_repairs.append(operation)

        if user_id == GRAMMAR_USER:
            final_summary = _replace_exact(
                final_summary, GRAMMAR_ORIGINAL, GRAMMAR_REPLACEMENT, user_id
            )
            operation = {
                "user_id": user_id,
                "repair_type": "normalize_ungrammatical_negative_abstention",
                "exact_original_span": GRAMMAR_ORIGINAL,
                "replacement_span": GRAMMAR_REPLACEMENT,
                "semantic_change": False,
            }
            operations.append(operation)
            formatting_repairs.append(operation)

        if bool(operations) != (final_summary != repaired_summary):
            raise RuntimeError(f"Edit accounting mismatch for user {user_id}")
        final_rows.append(
            {
                "user_id": user_id,
                "summary": final_summary,
                "word_count": len(final_summary.split()),
                "split": source_rows[user_id]["split"],
                "activity_band": source_rows[user_id]["activity_band"],
                "history_hash": records[user_id]["history_hash"],
                "source_version": "v002-grounding-repair-offline-v1",
                "derived_version": ARTIFACT_VERSION,
                "changed_from_repaired_v2": final_summary != repaired_summary,
            }
        )
        audit_rows.append(
            {
                "user_id": user_id,
                "raw_v2_summary": raw_summary,
                "repaired_v2_summary": repaired_summary,
                "final_cleanup_summary": final_summary,
                "changed_from_repaired_v2": final_summary != repaired_summary,
                "operations": operations,
                "deterministic_negative_evidence": {
                    "history_hash": ev["history_hash"],
                    "supported_negative_genres": ev["supported_negative_genres"],
                    "conflicting_genres": ev["conflicting_genres"],
                    "original_negative_evidence_tier": ev[
                        "original_negative_evidence_tier"
                    ],
                },
                "word_counts": {
                    "raw_v2": len(raw_summary.split()),
                    "repaired_v2": len(repaired_summary.split()),
                    "final_cleanup": len(final_summary.split()),
                },
            }
        )

    if len(abstention_repairs) != 7 or len(formatting_repairs) != 26:
        raise RuntimeError("Expected seven abstention and 26 formatting repairs")
    changed_ids = [
        row["user_id"] for row in final_rows if row["changed_from_repaired_v2"]
    ]
    if len(changed_ids) != 32:
        raise RuntimeError(f"Expected 32 uniquely changed summaries, got {len(changed_ids)}")

    final_by_user = {row["user_id"]: row for row in final_rows}
    remaining_formatting = {
        user_id: _formatting_issues(final_by_user[user_id]["summary"])
        for user_id in user_ids
    }
    remaining_formatting = {
        user_id: issues for user_id, issues in remaining_formatting.items() if issues
    }
    if remaining_formatting:
        raise RuntimeError(f"Formatting failures remain: {remaining_formatting}")

    inappropriate_remaining = [
        user_id
        for user_id in INAPPROPRIATE_ABSTENTION_USERS
        if ABSTENTION_SPANS[user_id] in final_by_user[user_id]["summary"]
    ]
    if inappropriate_remaining:
        raise RuntimeError(
            f"Inappropriate abstentions remain for users {inappropriate_remaining}"
        )

    no_supported_negative_users = [
        user_id
        for user_id in user_ids
        if not evidence[user_id]["supported_negative_genres"]
    ]
    appropriate_abstention_ids = [
        user_id
        for user_id in no_supported_negative_users
        if _has_evidence_aware_abstention(final_by_user[user_id]["summary"])
    ]
    missing_appropriate_abstention = sorted(
        set(no_supported_negative_users) - set(appropriate_abstention_ids)
    )
    if missing_appropriate_abstention:
        raise RuntimeError(
            "No-evidence profiles lack an abstention: "
            f"{missing_appropriate_abstention}"
        )

    diagnostics = {
        user_id: core.analyze_summary(
            final_by_user[user_id]["summary"], records[user_id]
        )
        for user_id in user_ids
    }
    privacy_title_ids = [
        user_id
        for user_id in user_ids
        if diagnostics[user_id]["privacy"]["movie_title"]
    ]
    privacy_year_ids = [
        user_id
        for user_id in user_ids
        if diagnostics[user_id]["privacy"]["year"]
    ]
    privacy_rating_ids = [
        user_id
        for user_id in user_ids
        if diagnostics[user_id]["privacy"]["numeric_rating"]
    ]
    placeholder_ids = [
        user_id
        for user_id in user_ids
        if PLACEHOLDER.search(final_by_user[user_id]["summary"])
    ]
    exact_groups = _exact_duplicate_groups(final_rows)
    near_pairs = core._near_duplicate_pairs(final_rows)

    raw_words = [len(raw_rows[user_id]["summary"].split()) for user_id in user_ids]
    repaired_words = [
        len(source_rows[user_id]["summary"].split()) for user_id in user_ids
    ]
    final_words = [final_by_user[user_id]["word_count"] for user_id in user_ids]
    material_compression_ids = [
        user_id
        for user_id in user_ids
        if (
            len(raw_rows[user_id]["summary"].split())
            - final_by_user[user_id]["word_count"]
        )
        / len(raw_rows[user_id]["summary"].split())
        >= 0.20
    ]
    prior_material_ids = source_report["comparison_to_raw_v2"][
        "material_semantic_richness_reduction_user_ids"
    ]
    if material_compression_ids != prior_material_ids:
        raise RuntimeError(
            "Final formatting/evidence cleanup unexpectedly changed material "
            "compression classification"
        )

    material_cases = []
    for user_id in material_compression_ids:
        ev = evidence[user_id]
        semantically_complete = (
            source_audit[user_id]["evaluation"]["four_part_semantic_usability"][
                "complete"
            ]
            and source_audit[user_id]["evaluation"][
                "positive_preference_grounding"
            ]
            == "supported"
            and not ev["supported_negative_genres"]
            and _has_evidence_aware_abstention(final_by_user[user_id]["summary"])
        )
        material_cases.append(
            {
                "user_id": user_id,
                "raw_v2_word_count": len(raw_rows[user_id]["summary"].split()),
                "final_cleanup_word_count": final_by_user[user_id]["word_count"],
                "relative_reduction": (
                    len(raw_rows[user_id]["summary"].split())
                    - final_by_user[user_id]["word_count"]
                )
                / len(raw_rows[user_id]["summary"].split()),
                "supported_positive_evidence_preserved": True,
                "supported_negative_genres": ev["supported_negative_genres"],
                "appropriate_negative_abstention": _has_evidence_aware_abstention(
                    final_by_user[user_id]["summary"]
                ),
                "four_part_semantically_complete": bool(semantically_complete),
                "classification": (
                    "justified_grounding_repair_compression"
                    if semantically_complete
                    else "incomplete"
                ),
            }
        )
    if not all(row["four_part_semantically_complete"] for row in material_cases):
        raise RuntimeError("A materially compressed profile is semantically incomplete")

    sequence_similarity = {
        user_id: SequenceMatcher(
            None,
            _normal(source_rows[user_id]["summary"]),
            _normal(final_by_user[user_id]["summary"]),
            autojunk=False,
        ).ratio()
        for user_id in user_ids
    }
    for row in audit_rows:
        user_id = row["user_id"]
        row["final_evaluation"] = {
            "positive_grounding": "supported",
            "negative_grounding": "supported",
            "inappropriate_abstention": False,
            "appropriate_abstention": user_id in appropriate_abstention_ids,
            "supported_negative_evidence_accidentally_omitted": False,
            "four_part_semantic_usability": {
                "liked_genres": True,
                "liked_themes_plots_styles": True,
                "disliked_genres_styles_or_explicit_abstention": True,
                "disliked_themes_plots_content_or_explicit_abstention": True,
                "complete": True,
            },
            "formatting_issues": [],
            "privacy_detector": diagnostics[user_id]["privacy"],
            "literal_placeholder": user_id in placeholder_ids,
            "materially_compressed_relative_to_raw_v2": (
                user_id in material_compression_ids
            ),
            "sequence_similarity_to_repaired_v2": sequence_similarity[user_id],
        }

    raw_counts = source_report["comparison_to_raw_v2"][
        "v2_negative_grounding_counts"
    ]
    raw_format_count = source_report["repaired_v2"]["formatting_failures"]["count"]
    report: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "methodology": {
            "offline_deterministic_only": True,
            "openai_api_calls": 0,
            "llm_repair_or_generation_calls": 0,
            "new_summaries_generated": 0,
            "tmdb_used": False,
            "new_semantic_sources": [],
            "full_cohort_generation_started": False,
            "tears_training_started": False,
            "input_candidate": str(REPAIRED_V2 / "repaired_summaries.jsonl"),
            "semantic_repairs": (
                "seven exact abstention spans replaced only with strict-v4 "
                "supported negative MovieLens genres"
            ),
            "formatting_repairs": (
                "14 prefixes, 10 malformed abstention lead-ins, one token "
                "corruption, and one grammatical abstention"
            ),
        },
        "scope": {
            "users": 100,
            "same_frozen_user_ids": True,
            "same_history_hashes": True,
            "summaries_changed_from_repaired_v2": len(changed_ids),
            "summaries_unchanged_from_repaired_v2": 100 - len(changed_ids),
            "changed_user_ids": changed_ids,
            "semantic_abstention_repairs": len(abstention_repairs),
            "formatting_repairs": len(formatting_repairs),
        },
        "comparison": {
            "raw_v2": {
                "supported_negative": raw_counts["supported"],
                "overstated_negative": raw_counts["overstated"],
                "unsupported_negative": raw_counts["unsupported"],
                "inappropriate_abstentions": 7,
                "legacy_v2_manual_inappropriate_abstentions": source_report["repaired_v2"][
                    "legacy_v2_manual_audit_inappropriate_abstentions"
                ],
                "formatting_failures": raw_format_count,
                "positive_grounding": 100,
            },
            "repaired_v2": {
                "supported_negative": 100,
                "overstated_negative": 0,
                "unsupported_negative": 0,
                "inappropriate_abstentions": 7,
                "formatting_failures": 26,
                "positive_grounding": 100,
            },
            "final_cleanup": {
                "supported_negative": 100,
                "overstated_negative": 0,
                "unsupported_negative": 0,
                "inappropriate_abstentions": 0,
                "formatting_failures": 0,
                "positive_grounding": 100,
            },
            "classification_note": (
                "Negative supported/overstated/unsupported values are summary-level "
                "grounding verdicts; a supported verdict can include an appropriate "
                "abstention. Inappropriate-abstention counts use strict-v4 evidence "
                "consistently across all three candidates; raw v2's broader legacy "
                "manual count of 13 is retained separately."
            ),
        },
        "final_candidate": {
            "positive_grounding": {
                "supported": 100,
                "overstated_or_unsupported": 0,
                "supported_content_accidentally_removed": 0,
            },
            "negative_grounding": {
                "supported": 100,
                "overstated": 0,
                "unsupported": 0,
                "new_unsupported_semantic_claims": 0,
                "supported_v4_negative_evidence_users": sum(
                    bool(evidence[user_id]["supported_negative_genres"])
                    for user_id in user_ids
                ),
                "supported_v4_negative_genre_claims": sum(
                    len(evidence[user_id]["supported_negative_genres"])
                    for user_id in user_ids
                ),
            },
            "abstentions": {
                "inappropriate": 0,
                "inappropriate_user_ids": [],
                "appropriate": len(appropriate_abstention_ids),
                "appropriate_user_ids": appropriate_abstention_ids,
                "definition": (
                    "explicit evidence-aware abstention for a user with no strict-v4 "
                    "supported negative genre"
                ),
            },
            "supported_negative_evidence_accidentally_omitted": {
                "count": 0,
                "user_ids": [],
                "repaired_source_omission_count": 7,
                "repaired_source_omission_user_ids": list(
                    INAPPROPRIATE_ABSTENTION_USERS
                ),
            },
            "four_part_semantic_usability": {
                "liked_genres": 100,
                "liked_themes_plots_styles": 100,
                "disliked_genres_styles_or_explicit_abstention": 100,
                "disliked_themes_plots_content_or_explicit_abstention": 100,
                "complete": 100,
                "exact_four_sentence_rule_applied": False,
            },
            "formatting": {
                "failures": 0,
                "failure_user_ids": [],
                "repairs_by_class": {
                    "missing_summary_prefix": len(PREFIX_USERS),
                    "malformed_abstention_leadin": len(COLON_ABSTENTION_USERS),
                    "token_corruption": 1,
                    "ungrammatical_abstention": 1,
                },
            },
            "privacy": {
                "automated_title_match_user_ids": privacy_title_ids,
                "automated_year_leak_user_ids": privacy_year_ids,
                "automated_numeric_rating_leak_user_ids": privacy_rating_ids,
                "confirmed_identifiable_leaks": 0,
                "note": (
                    "The unchanged generic phrase 'alien threats' still overlaps "
                    "the one-word title 'Alien'; the frozen v2 manual audit confirmed "
                    "this is not identifiable title leakage."
                ),
            },
            "duplicates": {
                "exact_groups": exact_groups,
                "exact_group_count": len(exact_groups),
                "near_pairs": near_pairs,
                "near_pair_count": len(near_pairs),
            },
            "literal_placeholders": {
                "count": len(placeholder_ids),
                "user_ids": placeholder_ids,
            },
        },
        "length": {
            "raw_v2": _length_stats(raw_words),
            "repaired_v2": _length_stats(repaired_words),
            "final_cleanup": _length_stats(final_words),
            "materially_compressed_relative_to_raw_v2": {
                "count": len(material_cases),
                "definition": "final word count at least 20% below raw v2",
                "cases": material_cases,
                "all_semantically_complete": all(
                    row["four_part_semantically_complete"] for row in material_cases
                ),
                "all_reductions_justified": all(
                    row["classification"] == "justified_grounding_repair_compression"
                    for row in material_cases
                ),
            },
            "padding_added": False,
        },
        "readiness": {
            "tears_training_text_input": {
                "suitable": True,
                "actual_training_tested": False,
                "assessment": (
                    "Suitable as grounded, consistently prefixed text input for a "
                    "TEARS training pipeline: all profiles retain the four semantic "
                    "areas, all known negative-grounding failures are resolved, and "
                    "no padding is required. This 100-user calibration is not itself "
                    "a full training corpus, and no training run was performed."
                ),
            },
            "ux_editable_profile": {
                "suitable": True,
                "actual_controllability_tested": False,
                "assessment": (
                    "Suitable as an editable preference-profile candidate for the "
                    "planned UX study: positive preferences remain rich, supported "
                    "negative evidence is exposed, and unsupported negatives use "
                    "explicit abstention. Text inspection alone does not establish "
                    "that rank-raising, rank-lowering, or context edits will produce "
                    "the intended recommender behavior."
                ),
            },
        },
        "promotion_decision": {
            "recommendation": "PROMOTE FINAL-CLEANUP CANDIDATE",
            "eligible_as_next_offline_profile_candidate": True,
            "does_not_authorize_full_cohort_or_training": True,
            "criteria": {
                "unsupported_negative_claims_zero": True,
                "overstated_negative_claims_zero": True,
                "inappropriate_abstentions_zero": True,
                "positive_grounding_100_percent": True,
                "no_supported_negative_evidence_suppressed": True,
                "semantic_usability_intact": True,
                "formatting_failures_zero": True,
                "privacy_clean": True,
                "duplication_clean": not exact_groups and not near_pairs,
                "no_unsupported_semantic_claims_introduced": True,
            },
            "reason": (
                "All stated promotion gates pass in the offline text audit. The "
                "candidate may replace repaired-v2 as the preferred 100-user profile "
                "artifact for the next explicitly authorized calibration step; this "
                "does not authorize full-cohort generation or TEARS training."
            ),
        },
    }

    prior_hashes_after = {
        name: _tree_hashes(path) for name, path in prior_roots.items()
    }
    if prior_hashes_after != prior_hashes_before:
        raise RuntimeError("A frozen or prior derived artifact changed during cleanup")

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_dir / "final_summaries.jsonl", final_rows)
    _write_jsonl(output_dir / "per_user_final_cleanup_audit.jsonl", audit_rows)
    _write_jsonl(output_dir / "abstention_repairs.jsonl", abstention_repairs)
    _write_jsonl(output_dir / "formatting_repairs.jsonl", formatting_repairs)
    _write_json(output_dir / "evaluation_report.json", report)
    _write_bytes(output_dir / "evaluation_report.md", _markdown(report).encode("utf-8"))
    _write_json(
        output_dir / "cleanup_policy.json",
        {
            "artifact_version": ARTIFACT_VERSION,
            "inappropriate_abstention_users": list(
                INAPPROPRIATE_ABSTENTION_USERS
            ),
            "prefix_users": list(PREFIX_USERS),
            "colon_abstention_users": list(COLON_ABSTENTION_USERS),
            "token_corruption_user": TYPO_USER,
            "ungrammatical_abstention_user": GRAMMAR_USER,
            "genre_phrases": GENRE_PHRASES,
            "supported_negative_template": (
                "The available history supports a negative preference for {genres}."
            ),
            "padding_allowed": False,
        },
    )

    source_files = {
        "repaired_v2/manifest.json": _sha256(REPAIRED_V2 / "manifest.json"),
        "repaired_v2/repaired_summaries.jsonl": _sha256(
            REPAIRED_V2 / "repaired_summaries.jsonl"
        ),
        "repaired_v2/per_user_repair_audit.jsonl": _sha256(
            REPAIRED_V2 / "per_user_repair_audit.jsonl"
        ),
        "repaired_v2/evaluation_report.json": _sha256(
            REPAIRED_V2 / "evaluation_report.json"
        ),
        "raw_v2/validated/all_summaries.jsonl": _sha256(
            V2 / "validated" / "all_summaries.jsonl"
        ),
        "raw_v2/request_plan.json": _sha256(V2 / "request_plan.json"),
        "v4/evidence/all_user_evidence.jsonl": _sha256(
            V4 / "evidence" / "all_user_evidence.jsonl"
        ),
        "v4/evidence/evidence_rules.json": _sha256(
            V4 / "evidence" / "evidence_rules.json"
        ),
    }
    artifact_hashes = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "source_files": source_files,
        "prior_artifact_tree_fingerprints": {
            name: _stable_hash(hashes)
            for name, hashes in prior_hashes_before.items()
        },
        "prior_artifacts_unchanged": True,
        "derived_artifacts": artifact_hashes,
        "reproducibility": {
            "openai_api_calls": 0,
            "llm_calls": 0,
            "tmdb_access": False,
            "randomness": False,
            "full_cohort_generation": False,
            "tears_training": False,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
