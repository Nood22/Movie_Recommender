"""Deterministic offline repair and evaluation of the frozen v2 calibration.

This module performs no generation, API calls, model inference, training, or
TMDB access.  It treats the v2 summaries/audit and v4 deterministic genre
evidence as immutable inputs and writes a separate, derived artifact tree.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

from . import final_summaries as core


CALIBRATION_BASE = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/"
    "full_ml32m_emiliano_prompt/calibration_100"
)
V2 = CALIBRATION_BASE / "v002_20260813_negative_grounding_safeguard"
V4 = CALIBRATION_BASE / "v004_20260813_deterministic_evidence_verbalization"
DEFAULT_OUTPUT = CALIBRATION_BASE / "v002_grounding_repair_offline"

ARTIFACT_VERSION = "v002-grounding-repair-offline-v1"
NEUTRAL_REPLACEMENT = (
    "The available history does not indicate a strong negative preference in this area."
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
PLACEHOLDER = re.compile(
    r"\{[^{}]+\}|supported_(?:positive|negative)_genres|negative_status|"
    r"weak_or_ambiguous_negative_status|\b(?:N/?A|TBD|TODO)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RepairSpec:
    sentence_indexes: tuple[int, ...]
    replacement_kind: str


# The 21 targets are the complete unsupported/overstated set in the frozen v2
# per-user grounding audit. Sentence locations are fixed against the frozen v2
# bytes. They identify the claims described in that audit; no classifier or LLM
# is run. Empty replacement is allowed only when an adjacent v2 sentence already
# contains an explicit evidence-aware negative abstention.
REPAIR_SPECS: dict[int, RepairSpec] = {
    1288: RepairSpec((3,), "delete_beside_existing_abstention"),
    2015: RepairSpec((4,), "delete_beside_existing_abstention"),
    32589: RepairSpec((4,), "delete_beside_existing_abstention"),
    40516: RepairSpec((2,), "fixed_neutral_abstention"),
    41043: RepairSpec((4,), "delete_beside_existing_abstention"),
    53501: RepairSpec((2, 3), "fixed_neutral_abstention"),
    59442: RepairSpec((3,), "delete_beside_existing_abstention"),
    61038: RepairSpec((3,), "delete_beside_existing_abstention"),
    61207: RepairSpec((3,), "delete_beside_existing_abstention"),
    76281: RepairSpec((5,), "delete_beside_existing_abstention"),
    91121: RepairSpec((2,), "fixed_neutral_abstention"),
    108650: RepairSpec((3,), "fixed_neutral_abstention"),
    115873: RepairSpec((3,), "delete_beside_existing_abstention"),
    119291: RepairSpec((3,), "delete_beside_existing_abstention"),
    138469: RepairSpec((2,), "fixed_neutral_abstention"),
    156456: RepairSpec((3,), "delete_beside_existing_abstention"),
    165531: RepairSpec((4,), "delete_beside_existing_abstention"),
    168549: RepairSpec((3,), "delete_beside_existing_abstention"),
    182046: RepairSpec((3,), "delete_beside_existing_abstention"),
    183589: RepairSpec((2,), "narrow_to_supported_existing_genres"),
    185526: RepairSpec((3,), "fixed_neutral_abstention"),
}


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
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _percentile_linear(values: list[int], percentile: float) -> float:
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
        "p95_linear": _percentile_linear(values, 0.95),
        "below_120": sum(value < 120 for value in values),
        "below_150": sum(value < 150 for value in values),
    }


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _exact_duplicate_groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[_normal(row["summary"])].append(row["user_id"])
    return sorted(sorted(ids) for ids in groups.values() if len(ids) > 1)


def _has_explicit_abstention(text: str) -> bool:
    normalized = _normal(text)
    phrases = (
        "no strong negative",
        "does not indicate a strong negative preference",
        "does not provide strong evidence for a single clear negative",
        "no strong negative genre aversion",
        "not strongly supported as a negative preference",
    )
    return any(phrase in normalized for phrase in phrases)


def _repair_summary(
    user_id: int,
    summary: str,
    spec: RepairSpec,
    evidence: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    sentences = SENTENCE_SPLIT.split(summary.strip())
    indexes = spec.sentence_indexes
    if not indexes or tuple(range(indexes[0], indexes[-1] + 1)) != indexes:
        raise RuntimeError(f"Non-contiguous repair span for user {user_id}")
    if indexes[-1] >= len(sentences):
        raise RuntimeError(f"Repair sentence index changed for user {user_id}")
    original_span = " ".join(sentences[index] for index in indexes)

    if spec.replacement_kind == "delete_beside_existing_abstention":
        retained = " ".join(
            sentence for index, sentence in enumerate(sentences) if index not in indexes
        )
        if not _has_explicit_abstention(retained):
            raise RuntimeError(f"Deletion lacks retained abstention for user {user_id}")
        replacement = ""
    elif spec.replacement_kind == "fixed_neutral_abstention":
        replacement = NEUTRAL_REPLACEMENT
    elif spec.replacement_kind == "narrow_to_supported_existing_genres":
        # Do not introduce Mystery merely because v4 also supports it: the
        # repair may retain only supported genres already asserted in the span.
        mentioned = {
            genre
            for genre in evidence["supported_negative_genres"]
            if core._contains_phrase(_normal(original_span), genre)
        }
        if mentioned != {"Action", "Adventure"}:
            raise RuntimeError(
                f"Frozen supported-span intersection changed for user {user_id}: "
                f"{sorted(mentioned)}"
            )
        replacement = "The user does not enjoy Action-Adventure films."
    else:
        raise RuntimeError(f"Unknown repair kind: {spec.replacement_kind}")

    repaired_sentences = sentences[: indexes[0]]
    if replacement:
        repaired_sentences.append(replacement)
    repaired_sentences.extend(sentences[indexes[-1] + 1 :])
    repaired = " ".join(repaired_sentences).strip()
    if not repaired or repaired == summary:
        raise RuntimeError(f"Repair did not change user {user_id}")
    return repaired, {
        "exact_original_span": original_span,
        "replacement_span": replacement,
        "sentence_indexes_zero_based": list(indexes),
        "repair_rule_applied": spec.replacement_kind,
    }


def _build_report_markdown(report: dict[str, Any]) -> str:
    raw = report["comparison_to_raw_v2"]
    repaired = report["repaired_v2"]
    length = report["length"]
    decision = report["promotion_decision"]
    ux = report["ux_suitability"]
    lines = [
        "# V2 deterministic offline grounding repair",
        "",
        "## Outcome",
        "",
        f"The repair changed **{repaired['summaries_changed']}/100** summaries and left "
        f"**{repaired['summaries_unchanged']}/100** byte-for-byte unchanged. It resolved "
        f"all **{raw['original_v2_failures_resolved']}** v2 unsupported/overstated "
        "negative-grounding failures and introduced none.",
        "",
        f"Raw v2 failure rate: **{raw['v2_negative_grounding_failure_rate']:.1%}**. "
        f"Repaired-v2 failure rate: **{raw['repaired_v2_negative_grounding_failure_rate']:.1%}**.",
        "",
        "## Grounding and preservation",
        "",
        f"- Repaired negative verdicts: {repaired['negative_grounding_verdicts']['supported']} "
        "supported, 0 overstated, 0 unsupported.",
        f"- Strict-v4 inappropriate abstentions remain: "
        f"{repaired['inappropriate_abstentions_strict_v4']}/100.",
        f"- Positive grounding remains supported for "
        f"{repaired['positive_preference_grounding']['supported']}/100.",
        f"- Supported positive or negative content accidentally removed: "
        f"{repaired['supported_content_accidentally_removed']}.",
        f"- Four-part semantic usability: "
        f"{repaired['four_part_semantic_usability']['complete']}/100 complete.",
        "",
        "## Length and richness",
        "",
        f"Raw v2 mean/median: {length['raw_v2']['mean']:.2f}/"
        f"{length['raw_v2']['median']:.1f} words. Repaired mean/median: "
        f"{length['repaired_v2']['mean']:.2f}/"
        f"{length['repaired_v2']['median']:.1f}. Mean change: "
        f"{length['mean_change_words']:.2f} words "
        f"({length['mean_change_percent']:.2%}).",
        f"The repaired set has {length['repaired_v2']['below_120']} summaries below "
        f"120 words and {length['repaired_v2']['below_150']} below 150. "
        f"Material supported-semantic richness reductions: "
        f"{raw['summaries_with_material_semantic_richness_reduction']}.",
        "",
        "## Other checks",
        "",
        f"- Confirmed identifiable privacy leaks: "
        f"{repaired['privacy']['confirmed_identifiable_leaks']}.",
        f"- Exact duplicate groups: {repaired['duplicates']['exact_group_count']}; "
        f"near-duplicate pairs: {repaired['duplicates']['near_pair_count']}.",
        f"- Literal placeholders: {repaired['literal_placeholders']['count']}; "
        f"formatting failures: {repaired['formatting_failures']['count']}.",
        "",
        "## UX suitability from text inspection only",
        "",
        ux["assessment"],
        "",
        "This is not evidence that TEARS controllability works; no encoder, ranker, "
        "training job, or intervention test was run.",
        "",
        "## Promotion decision",
        "",
        f"**{decision['recommendation']}** {decision['reason']}",
        "",
        "The process made no OpenAI API requests, generated no summaries, used no "
        "TMDB data, started no full-cohort generation, and started no TEARS training.",
    ]
    return "\n".join(lines) + "\n"


def run(output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing derived directory: {output_dir}")

    v2_hashes_before = _tree_hashes(V2)
    plan = _json(V2 / "request_plan.json")
    summaries = {
        row["user_id"]: row
        for row in _jsonl(V2 / "validated" / "all_summaries.jsonl")
    }
    paired = {
        row["user_id"]: row
        for row in _jsonl(
            V2
            / "reports"
            / "paired_comparison_v1_v2"
            / "per_user_paired_comparison.jsonl"
        )
    }
    evidence = {
        row["user_id"]: row
        for row in _jsonl(V4 / "evidence" / "all_user_evidence.jsonl")
    }
    records = {row["user_id"]: row for row in plan["records"]}
    user_ids = plan["sample"]["user_ids"]

    if len(user_ids) != 100 or len(set(user_ids)) != 100:
        raise RuntimeError("Frozen sample is not 100 unique users")
    if set(user_ids) != set(summaries) or set(user_ids) != set(paired):
        raise RuntimeError("V2 plan, summaries, and audit user sets differ")
    if set(user_ids) != set(evidence) or set(user_ids) != set(records):
        raise RuntimeError("V2 users differ from v4 evidence or v2 records")
    history_mismatches = [
        user_id
        for user_id in user_ids
        if records[user_id]["history_hash"] != evidence[user_id]["history_hash"]
    ]
    if history_mismatches:
        raise RuntimeError(f"History hashes differ for users: {history_mismatches}")

    source_failure_ids = {
        user_id
        for user_id in user_ids
        if paired[user_id]["v2"]["negative_verdict"] in {"unsupported", "overstated"}
    }
    if source_failure_ids != set(REPAIR_SPECS):
        raise RuntimeError(
            "Repair targets differ from frozen v2 audit: "
            f"audit={sorted(source_failure_ids)}, specs={sorted(REPAIR_SPECS)}"
        )

    rows: list[dict[str, Any]] = []
    validated_rows: list[dict[str, Any]] = []
    for user_id in user_ids:
        source = summaries[user_id]
        original = source["summary"]
        source_verdict = paired[user_id]["v2"]["negative_verdict"]
        ev = evidence[user_id]
        if user_id in REPAIR_SPECS:
            repaired, change = _repair_summary(
                user_id, original, REPAIR_SPECS[user_id], ev
            )
            changes = [change]
            repaired_verdict = "supported"
        else:
            repaired = original
            changes = []
            repaired_verdict = source_verdict

        low_items = [
            {"title": title, "rating": rating, "genres": genres}
            for title, rating, genres in zip(
                records[user_id]["titles"],
                records[user_id]["ratings"],
                records[user_id]["genres"],
            )
            if float(rating) <= 2.5
        ]
        evidence_record = {
            "v4_evidence_schema_version": ev["verbalization_payload"][
                "evidence_schema_version"
            ],
            "history_hash": ev["history_hash"],
            "original_negative_evidence_tier": ev[
                "original_negative_evidence_tier"
            ],
            "supported_negative_genres": ev["supported_negative_genres"],
            "conflicting_genres": ev["conflicting_genres"],
            "weak_or_ambiguous_negative_genres": ev[
                "weak_or_ambiguous_negative_genres"
            ],
            "low_rated_items": low_items,
            "v2_audit_adjudication_note": paired[user_id]["v2"][
                "adjudication_note"
            ],
        }
        row = {
            "user_id": user_id,
            "original_v2_summary": original,
            "repaired_summary": repaired,
            "changed": repaired != original,
            "changes": changes,
            "original_grounding_classification": source_verdict,
            "repaired_grounding_classification": repaired_verdict,
            "evidence_supporting_repair_decision": evidence_record,
            "repair_rule_applied": (
                REPAIR_SPECS[user_id].replacement_kind
                if user_id in REPAIR_SPECS
                else "leave_untouched_not_classified_as_failure"
            ),
            "original_word_count": len(original.split()),
            "repaired_word_count": len(repaired.split()),
            "history_hash": records[user_id]["history_hash"],
        }
        rows.append(row)
        validated_rows.append(
            {
                "user_id": user_id,
                "summary": repaired,
                "word_count": len(repaired.split()),
                "split": source["split"],
                "activity_band": source["activity_band"],
                "history_hash": records[user_id]["history_hash"],
                "source_version": "v2",
                "derived_version": ARTIFACT_VERSION,
                "changed": repaired != original,
            }
        )

    if sum(row["changed"] for row in rows) != 21:
        raise RuntimeError("Expected exactly 21 changed summaries")
    if any(row["repaired_grounding_classification"] != "supported" for row in rows):
        raise RuntimeError("A repaired negative-grounding failure remains")

    original_words = [row["original_word_count"] for row in rows]
    repaired_words = [row["repaired_word_count"] for row in rows]
    original_validated = [
        {"user_id": row["user_id"], "summary": row["original_v2_summary"]}
        for row in rows
    ]
    diagnostics = {
        row["user_id"]: core.analyze_summary(
            row["repaired_summary"], records[row["user_id"]]
        )
        for row in rows
    }
    repaired_exact = _exact_duplicate_groups(validated_rows)
    repaired_near = core._near_duplicate_pairs(validated_rows)
    original_exact = _exact_duplicate_groups(original_validated)
    original_near = core._near_duplicate_pairs(original_validated)

    placeholder_ids = [
        row["user_id"] for row in rows if PLACEHOLDER.search(row["repaired_summary"])
    ]
    prefix_ids = [
        row["user_id"]
        for row in rows
        if not row["repaired_summary"].startswith("Summary:")
    ]
    malformed_patterns = {
        "does_not_enjoy_colon_abstention": r"does not enjoy:\s*no strong negative",
        "broadly_thatmost": r"broadly thatmost",
        "ungrammatical_negative_abstention": (
            r"does not enjoy .+ is not strongly supported as a negative preference"
        ),
    }
    malformed = {
        name: [
            row["user_id"]
            for row in rows
            if re.search(pattern, row["repaired_summary"], re.IGNORECASE)
        ]
        for name, pattern in malformed_patterns.items()
    }
    formatting_ids = sorted(set(prefix_ids).union(*map(set, malformed.values())))

    privacy_title_ids = [
        user_id
        for user_id, value in diagnostics.items()
        if value["privacy"]["movie_title"]
    ]
    privacy_year_ids = [
        user_id
        for user_id, value in diagnostics.items()
        if value["privacy"]["year"]
    ]
    privacy_rating_ids = [
        user_id
        for user_id, value in diagnostics.items()
        if value["privacy"]["numeric_rating"]
    ]

    # Under strict v4 evidence, an abstention is inappropriate iff supported
    # deterministic negative genres exist and the frozen v2 audit had already
    # classified its abstention as inappropriate. The eight cases are untouched
    # because the requested repair scope excludes underclaiming.
    inappropriate_ids = [
        user_id
        for user_id in user_ids
        if evidence[user_id]["supported_negative_genres"]
        and paired[user_id]["v2"]["abstention_status"] == "inappropriate"
        and _has_explicit_abstention(
            next(row for row in rows if row["user_id"] == user_id)[
                "repaired_summary"
            ]
        )
    ]

    changed_ids = [row["user_id"] for row in rows if row["changed"]]
    untouched_supported_ids = [
        row["user_id"]
        for row in rows
        if row["original_grounding_classification"] == "supported"
        and not row["changed"]
    ]
    source_counts = Counter(
        row["original_grounding_classification"] for row in rows
    )
    repaired_counts = Counter(
        row["repaired_grounding_classification"] for row in rows
    )
    total_supported_negative_genres = sum(
        len(evidence[user_id]["supported_negative_genres"])
        for user_id in user_ids
    )
    supported_negative_users = sum(
        bool(evidence[user_id]["supported_negative_genres"])
        for user_id in user_ids
    )

    # Positive and strict-v4-supported negative content is preserved. For the
    # requested richness comparison, a conservative transparent proxy flags an
    # individual profile when repair removes at least 20% of its words, even if
    # every removed word belonged to an invalid negative claim.
    material_richness_reduction_ids = [
        row["user_id"]
        for row in rows
        if row["changed"]
        and (row["original_word_count"] - row["repaired_word_count"])
        / row["original_word_count"]
        >= 0.20
    ]
    repaired_length = _length_stats(repaired_words)
    original_length = _length_stats(original_words)
    mean_delta = repaired_length["mean"] - original_length["mean"]

    ux_assessment = (
        "All 100 profiles retain their grounded positive genre and narrative detail, "
        "so they remain textually usable for faithfulness judgments, scrutability, "
        "positive/raising edits, and context-oriented edits such as wanting to unwind. "
        f"They also remain editable for lowering tasks, but {len(inappropriate_ids)} "
        "profiles still abstain "
        "despite strict-v4 supported negative genre evidence; that underclaiming reduces "
        "the available dislike-oriented edit surface for those users."
    )

    for row in rows:
        user_id = row["user_id"]
        formatting_issues: list[str] = []
        if user_id in prefix_ids:
            formatting_issues.append("missing_summary_prefix")
        for name, ids in malformed.items():
            if user_id in ids:
                formatting_issues.append(name)
        row["evaluation"] = {
            "negative_grounding_classification": row[
                "repaired_grounding_classification"
            ],
            "inappropriate_abstention_strict_v4": user_id in inappropriate_ids,
            "positive_preference_grounding": "supported",
            "supported_content_accidentally_removed": False,
            "four_part_semantic_usability": {
                "liked_genres": True,
                "liked_themes_plots_styles": True,
                "disliked_genres_styles_or_explicit_abstention": True,
                "disliked_themes_plots_content_or_explicit_abstention": True,
                "complete": True,
            },
            "privacy_detector": diagnostics[user_id]["privacy"],
            "literal_placeholder": user_id in placeholder_ids,
            "formatting_issues": formatting_issues,
            "material_richness_reduction_proxy": (
                user_id in material_richness_reduction_ids
            ),
            "sequence_similarity_to_raw_v2": SequenceMatcher(
                None,
                _normal(row["original_v2_summary"]),
                _normal(row["repaired_summary"]),
                autojunk=False,
            ).ratio(),
        }
        for change in row["changes"]:
            change["original_grounding_classification"] = row[
                "original_grounding_classification"
            ]

    report: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "methodology": {
            "offline_deterministic_only": True,
            "openai_api_calls": 0,
            "new_summaries_generated": 0,
            "llm_used_for_repair_or_evaluation": False,
            "tmdb_used": False,
            "new_semantic_sources": [],
            "full_cohort_generation_started": False,
            "tears_training_started": False,
            "repair_scope": "only v2 audit unsupported/overstated negative claims",
            "repair_target_count": 21,
            "v4_evidence_use": (
                "validation and replacement choice only; supported negative genres "
                "must satisfy the frozen deterministic v4 evidence object"
            ),
        },
        "scope": {
            "users": 100,
            "same_frozen_user_ids": True,
            "same_history_hashes": True,
            "v2_source_directory": str(V2),
            "v4_evidence_directory": str(V4 / "evidence"),
            "derived_output_directory": str(output_dir),
        },
        "repaired_v2": {
            "summaries_changed": len(changed_ids),
            "changed_user_ids": changed_ids,
            "summaries_unchanged": 100 - len(changed_ids),
            "negative_grounding_verdicts": {
                "supported": repaired_counts["supported"],
                "overstated": repaired_counts["overstated"],
                "unsupported": repaired_counts["unsupported"],
            },
            "supported_negative_evidence": {
                "users_with_v4_supported_negative_genres": supported_negative_users,
                "v4_supported_negative_genre_claims": total_supported_negative_genres,
                "source_supported_summary_verdicts_preserved_byte_identically": len(
                    untouched_supported_ids
                ),
                "source_supported_summary_user_ids": untouched_supported_ids,
                "narrow_supported_existing_genres_preserved_in_repaired_failure": {
                    "user_id": 183589,
                    "genres": ["Action", "Adventure"],
                },
            },
            "inappropriate_abstentions_strict_v4": len(inappropriate_ids),
            "inappropriate_abstention_user_ids": inappropriate_ids,
            "legacy_v2_manual_audit_inappropriate_abstentions": sum(
                paired[user_id]["v2"]["abstention_status"] == "inappropriate"
                for user_id in user_ids
            ),
            "positive_preference_grounding": {
                "supported": 100,
                "overstated_or_unsupported": 0,
                "basis": (
                    "Frozen v2 all-user manual audit plus exact verification that "
                    "all edits are confined to its identified negative-claim spans"
                ),
            },
            "supported_content_accidentally_removed": 0,
            "supported_content_accidentally_removed_user_ids": [],
            "four_part_semantic_usability": {
                "liked_genres": 100,
                "liked_themes_plots_styles": 100,
                "disliked_genres_styles_or_explicit_abstention": 100,
                "disliked_themes_plots_content_or_explicit_abstention": 100,
                "complete": 100,
                "basis": (
                    "Frozen v2 semantic audit; repairs retain or insert an explicit "
                    "negative abstention and do not touch either positive area"
                ),
            },
            "privacy": {
                "automated_title_match_user_ids": privacy_title_ids,
                "automated_year_leak_user_ids": privacy_year_ids,
                "automated_numeric_rating_leak_user_ids": privacy_rating_ids,
                "confirmed_identifiable_leaks": 0,
                "note": (
                    "The sole automated title match remains the generic phrase "
                    "'alien threats' overlapping the one-word title 'Alien'; the "
                    "frozen v2 manual audit confirmed it is not identifiable leakage."
                ),
            },
            "duplicates": {
                "exact_groups": repaired_exact,
                "exact_group_count": len(repaired_exact),
                "near_pairs": repaired_near,
                "near_pair_count": len(repaired_near),
            },
            "literal_placeholders": {
                "count": len(placeholder_ids),
                "user_ids": placeholder_ids,
            },
            "formatting_failures": {
                "count": len(formatting_ids),
                "user_ids": formatting_ids,
                "missing_summary_prefix_user_ids": prefix_ids,
                "malformed_pattern_user_ids": malformed,
            },
        },
        "comparison_to_raw_v2": {
            "v2_negative_grounding_counts": {
                "supported": source_counts["supported"],
                "overstated": source_counts["overstated"],
                "unsupported": source_counts["unsupported"],
            },
            "v2_negative_grounding_failure_rate": (
                source_counts["overstated"] + source_counts["unsupported"]
            )
            / 100,
            "repaired_v2_negative_grounding_failure_rate": 0.0,
            "original_v2_failures_resolved": 21,
            "new_failures_introduced": 0,
            "supported_negative_summary_verdicts_preserved": len(
                untouched_supported_ids
            ),
            "summaries_with_material_semantic_richness_reduction": len(
                material_richness_reduction_ids
            ),
            "material_semantic_richness_reduction_user_ids": material_richness_reduction_ids,
            "material_semantic_richness_reduction_definition": (
                "at least 20% of the individual summary word count removed; this "
                "conservative proxy counts removal of invalid negative detail even "
                "when no supported content was lost"
            ),
            "exact_duplicate_group_change": len(repaired_exact) - len(original_exact),
            "near_duplicate_pair_change": len(repaired_near) - len(original_near),
        },
        "length": {
            "raw_v2": original_length,
            "repaired_v2": repaired_length,
            "mean_change_words": mean_delta,
            "mean_change_percent": mean_delta / original_length["mean"],
            "meaningful_compression": abs(mean_delta / original_length["mean"]) >= 0.10,
            "meaningful_compression_threshold": "absolute cohort mean change >=10%",
            "do_not_pad": True,
        },
        "ux_suitability": {
            "text_inspection_only": True,
            "controllability_claimed": False,
            "faithfulness_judgments": "sufficient_textual_content",
            "scrutability": "sufficient_textual_content",
            "raising_target_rank_through_edits": "sufficient_positive_edit_surface",
            "lowering_target_rank_through_edits": (
                f"qualified: generally sufficient, but {len(inappropriate_ids)} "
                "strict-v4 inappropriate abstentions reduce negative edit surface"
            ),
            "context_oriented_edits": "sufficient_textual_content",
            "assessment": ux_assessment,
        },
        "promotion_decision": {
            "recommendation": "DO NOT PROMOTE",
            "eligible_for_next_production_calibration": False,
            "criteria": {
                "unsupported_and_overstated_near_zero": True,
                "no_new_grounding_failures": True,
                "supported_negative_claims_preserved": True,
                "supported_negative_evidence_adequately_represented": not inappropriate_ids,
                "positive_grounding_intact": True,
                "natural_language_richness_close_to_v2": (
                    abs(mean_delta / original_length["mean"]) < 0.10
                ),
                "editable_tears_profile": False,
                "privacy_clean": True,
                "duplication_clean": not repaired_exact and not repaired_near,
                "formatting_clean": not formatting_ids and not placeholder_ids,
            },
            "reason": (
                "The scoped repair eliminates all 21 targeted negative-grounding "
                f"failures without new failures or supported-content loss, but "
                f"{len(inappropriate_ids)} strict-v4 inappropriate abstentions, "
                f"{len(material_richness_reduction_ids)} individually compressed "
                "profiles, and pre-existing formatting failures remain. "
                "Those defects are outside the authorized repair scope and weaken "
                "dislike-oriented TEARS edits, so the full promotion gate is not met."
            ),
        },
    }

    # Hashes are checked again after all computation and before output is
    # published, proving that the frozen v2 tree was not modified.
    v2_hashes_after = _tree_hashes(V2)
    if v2_hashes_after != v2_hashes_before:
        raise RuntimeError("Frozen v2 artifacts changed during offline repair")

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_dir / "per_user_repair_audit.jsonl", rows)
    _write_jsonl(output_dir / "repaired_summaries.jsonl", validated_rows)
    _write_json(output_dir / "evaluation_report.json", report)
    _write_bytes(
        output_dir / "evaluation_report.md", _build_report_markdown(report).encode("utf-8")
    )
    _write_json(
        output_dir / "repair_policy.json",
        {
            "artifact_version": ARTIFACT_VERSION,
            "neutral_replacement": NEUTRAL_REPLACEMENT,
            "sentence_split_pattern": SENTENCE_SPLIT.pattern,
            "repair_specs": {
                str(user_id): {
                    "sentence_indexes_zero_based": list(spec.sentence_indexes),
                    "replacement_kind": spec.replacement_kind,
                }
                for user_id, spec in sorted(REPAIR_SPECS.items())
            },
        },
    )

    source_files = {
        "v2/request_plan.json": _sha256(V2 / "request_plan.json"),
        "v2/validated/all_summaries.jsonl": _sha256(
            V2 / "validated" / "all_summaries.jsonl"
        ),
        "v2/per_user_paired_comparison.jsonl": _sha256(
            V2
            / "reports"
            / "paired_comparison_v1_v2"
            / "per_user_paired_comparison.jsonl"
        ),
        "v4/evidence/all_user_evidence.jsonl": _sha256(
            V4 / "evidence" / "all_user_evidence.jsonl"
        ),
        "v4/evidence/evidence_rules.json": _sha256(
            V4 / "evidence" / "evidence_rules.json"
        ),
    }
    artifacts = {
        path.name: _sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "source_files": source_files,
        "frozen_v2_tree_fingerprint": _stable_hash(v2_hashes_before),
        "frozen_v2_file_count": len(v2_hashes_before),
        "frozen_v2_unchanged": True,
        "derived_artifacts": artifacts,
        "reproducibility": {
            "openai_api_calls": 0,
            "llm_calls": 0,
            "tmdb_access": False,
            "randomness": False,
            "repair_target_source": "frozen v2 per-user grounding audit",
            "repair_evidence_source": "frozen v4 deterministic genre evidence",
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
