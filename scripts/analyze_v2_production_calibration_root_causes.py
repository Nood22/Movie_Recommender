#!/usr/bin/env python3
"""Offline root-cause analysis for the immutable 1,000-user V2 calibration.

This script reads only completed calibration artifacts. It performs no model/API
calls and writes a separate diagnostic artifact under the repository.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys


SOURCE = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "production_calibration_1000/v002_20260816_v2_generic_repair"
)
FINAL = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "production_calibration_1000/v008_20260816_v2_generic_repair_final"
)
OUTPUT = Path(
    "artifacts/production_calibration_1000/"
    "v008_root_cause_analysis_offline"
)

sys.path.insert(0, str(FINAL / "code"))
import production_summary_protocol as protocol  # noqa: E402


SEMANTIC_OPS = {
    "remove_additional_negative_or_abstention_span",
    "grounding_narrow_or_neutralize",
    "abstention_to_supported_negative",
}
FORMATTING_OPS = {
    "format_add_summary_prefix",
    "format_malformed_abstention_leadin",
    "format_remove_literal_instruction_or_placeholder",
}
CONTRADICTION_IDS = {
    8742,
    18515,
    50674,
    60875,
    65242,
    83310,
    92810,
    118834,
    159382,
    166368,
    199226,
}
TITLE_LEAK_IDS = {35148, 39004, 92423, 127899}
OTHER_VIEWER_RE = re.compile(r"\b(?:other|some) (?:viewers|users|audiences)\b", re.I)
RATING_INFERENCE_RE = re.compile(
    r"\b(?:rating|rated|low score|lower score|very low|few low|single low|"
    r"isolated low|mixed response|weaker response)\b",
    re.I,
)
BROAD_RE = re.compile(
    r"\b(?:broad(?:ly)?|generally|tends? to|especially|across|overall|"
    r"straightforward|conventional|formulaic|generic|unoriginal)\b",
    re.I,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def words(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def count_record(count: int, *, all_count: int = 1000, changed_count: int = 867) -> dict:
    return {
        "count": count,
        "percent_of_all_1000": pct(count, all_count),
        "percent_of_changed_867": pct(count, changed_count),
    }


def span_flags(text: str, supported: set[str]) -> dict:
    mentioned = protocol._mentioned_genres(text)
    return {
        "mentioned_genres": sorted(mentioned),
        "supported_genres_mentioned": sorted(mentioned & supported),
        "unsupported_or_unlicensed_genres_mentioned": sorted(mentioned - supported),
        "mentions_non_genre_plot_theme_style_attribute": bool(
            protocol.NON_GENRE_NEGATIVE_ATTRIBUTE_PATTERN.search(text)
        ),
        "is_abstention": protocol._is_abstention(text),
        "mentions_other_viewers": bool(OTHER_VIEWER_RE.search(text)),
        "contains_rating_evidence_language": bool(RATING_INFERENCE_RE.search(text)),
        "has_broad_generalization_language": bool(BROAD_RE.search(text))
        or len(mentioned) > 1,
    }


def exclusive_span_category(flags: dict) -> str:
    if flags["mentions_non_genre_plot_theme_style_attribute"]:
        return "inferred_negative_plot_theme_style_or_tone"
    if flags["mentioned_genres"]:
        return "negative_genre_generalization"
    if flags["is_abstention"]:
        return "abstention_or_evidence_caveat"
    return "generic_negative_assertion"


def primary_failure(op_types: set[str], supported: tuple[str, ...]) -> str:
    if "abstention_to_supported_negative" in op_types:
        return "inappropriate_abstention_despite_supported_negative_genre"
    if "grounding_narrow_or_neutralize" in op_types:
        if supported:
            return "overbroad_negative_claim_narrowed_to_supported_genres"
        return "unsupported_negative_claim_neutralized_no_supported_genre"
    if "remove_additional_negative_or_abstention_span" in op_types:
        return "extra_forced_negative_or_duplicate_abstention_removed"
    return "formatting_only"


def reason_text(primary: str, evidence_category: str, supported: tuple[str, ...]) -> str:
    if primary == "inappropriate_abstention_despite_supported_negative_genre":
        return (
            "Raw V2 abstained even though deterministic repeated rating evidence "
            f"supported the narrow negative genre set {list(supported)}."
        )
    if primary == "overbroad_negative_claim_narrowed_to_supported_genres":
        return (
            "Raw V2 made a negative claim broader than the licensed genre evidence; "
            f"the repair retained only {list(supported)}."
        )
    if primary == "unsupported_negative_claim_neutralized_no_supported_genre":
        return (
            "Raw V2 asserted a negative preference although deterministic evidence "
            f"was {evidence_category} and supported no negative genre."
        )
    if primary == "extra_forced_negative_or_duplicate_abstention_removed":
        return (
            "Raw V2 filled an additional negative/other-viewer slot after the one "
            "licensed negative statement; the extra span was removed."
        )
    return "Only deterministic prefix/placeholder/grammar normalization was needed."


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(FINAL / "validated/all_raw_and_final_summaries.jsonl")
    evidence_rows = read_jsonl(SOURCE / "evidence/all_user_evidence.jsonl")
    evidence = {int(row["user_id"]): row for row in evidence_rows}
    plans = {
        int(row["user_id"]): row
        for row in json.loads((SOURCE / "request_plan.json").read_text())["records"]
    }
    privacy = json.loads((FINAL / "reports/privacy_adjudication.json").read_text())
    compression = json.loads((FINAL / "reports/material_compression_audit.json").read_text())

    changed = [row for row in rows if row["changed"]]
    assert len(rows) == 1000 and len(changed) == 867

    per_user: list[dict] = []
    operation_occurrences = Counter()
    operation_users = Counter()
    primary_counts = Counter()
    evidence_counts_all = Counter(row["evidence_category"] for row in rows)
    evidence_counts_changed = Counter(row["evidence_category"] for row in changed)
    semantic_users: set[int] = set()
    formatting_only_users: set[int] = set()
    content_flag_users: dict[str, set[int]] = defaultdict(set)
    exclusive_category_users: dict[str, set[int]] = defaultdict(set)
    exclusive_category_ops = Counter()
    exclusive_category_word_delta = Counter()
    supported_semantics_touched_users: set[int] = set()

    for row in changed:
        uid = int(row["user_id"])
        supported = tuple(row["supported_negative_genres"])
        supported_set = set(supported)
        op_types = {op["type"] for op in row["operations"]}
        for op_type in op_types:
            operation_users[op_type] += 1
        for op in row["operations"]:
            operation_occurrences[op["type"]] += 1

        semantic_ops = [op for op in row["operations"] if op["type"] in SEMANTIC_OPS]
        formatting_ops = [op for op in row["operations"] if op["type"] in FORMATTING_OPS]
        if semantic_ops:
            semantic_users.add(uid)
        else:
            formatting_only_users.add(uid)

        primary = primary_failure(op_types, supported)
        primary_counts[primary] += 1
        repaired_spans = []
        for op in semantic_ops:
            original = op.get("original_span", "")
            replacement = op.get("replacement_span", "")
            flags = span_flags(original, supported_set)
            category = exclusive_span_category(flags)
            delta = max(0, words(original) - words(replacement))
            exclusive_category_users[category].add(uid)
            exclusive_category_ops[category] += 1
            exclusive_category_word_delta[category] += delta
            for flag in (
                "mentions_non_genre_plot_theme_style_attribute",
                "is_abstention",
                "mentions_other_viewers",
                "contains_rating_evidence_language",
                "has_broad_generalization_language",
            ):
                if flags[flag]:
                    content_flag_users[flag].add(uid)
            if flags["mentioned_genres"]:
                content_flag_users["mentions_any_genre"].add(uid)
            if flags["supported_genres_mentioned"]:
                supported_semantics_touched_users.add(uid)
            repaired_spans.append(
                {
                    "operation": op["type"],
                    "original_span": original,
                    "replacement_span": replacement,
                    "word_count_before": words(original),
                    "word_count_after": words(replacement),
                    "word_reduction": words(original) - words(replacement),
                    "content_category": category,
                    "content_flags": flags,
                }
            )

        ev = evidence[uid]
        genre_evidence = {
            genre: ev["genre_statistics"][genre]
            for genre in supported
        }
        per_user.append(
            {
                "user_id": uid,
                "evidence_category": row["evidence_category"],
                "supported_negative_genres": list(supported),
                "supported_negative_genre_statistics": genre_evidence,
                "primary_failure_type": primary,
                "why_repair_was_required": reason_text(
                    primary, row["evidence_category"], supported
                ),
                "semantic_repair": bool(semantic_ops),
                "formatting_only": not semantic_ops,
                "raw_word_count": row["raw_word_count"],
                "final_word_count": row["final_word_count"],
                "absolute_word_reduction": row["raw_word_count"] - row["final_word_count"],
                "relative_word_reduction": round(
                    (row["raw_word_count"] - row["final_word_count"])
                    / row["raw_word_count"],
                    6,
                ),
                "materially_compressed": row["materially_compressed"],
                "raw_summary": row["raw_summary"],
                "final_summary": row["final_summary"],
                "semantic_repair_spans": repaired_spans,
                "formatting_operations": formatting_ops,
                "all_operations": row["operations"],
                "positive_sentences_preserved": row["positive_sentences_preserved"],
                "supported_negative_evidence_omitted": row[
                    "supported_negative_evidence_omitted"
                ],
            }
        )

    # Mutually exclusive failure accounting.
    assert sum(primary_counts.values()) == 867
    assert len(semantic_users) == 819
    assert len(formatting_only_users) == 48

    materially_compressed = [row for row in rows if row["materially_compressed"]]
    assert len(materially_compressed) == 657
    compressed_ids = {int(row["user_id"]) for row in materially_compressed}
    compressed_flag_counts = {
        key: len(users & compressed_ids) for key, users in content_flag_users.items()
    }
    compressed_category_counts = {
        key: len(users & compressed_ids) for key, users in exclusive_category_users.items()
    }
    conflict_ids = {
        int(row["user_id"]) for row in evidence_rows if row["conflicting_genres"]
    }
    compressed_category_operation_counts = Counter()
    compressed_category_word_delta = Counter()
    for record in per_user:
        if record["user_id"] not in compressed_ids:
            continue
        for span in record["semantic_repair_spans"]:
            category = span["content_category"]
            compressed_category_operation_counts[category] += 1
            compressed_category_word_delta[category] += max(
                0, span["word_reduction"]
            )

    # Explain the eleven sample contradictions with the exact positive-side text
    # and the deterministic genre statistics that drove the negative slot.
    contradiction_records = []
    for uid in sorted(CONTRADICTION_IDS):
        row = next(item for item in rows if int(item["user_id"]) == uid)
        marker = "The available history supports a negative preference for"
        positive_text = row["final_summary"].split(marker, 1)[0].strip()
        lexical_overlap = sorted(
            protocol._mentioned_genres(positive_text)
            & set(row["supported_negative_genres"])
        )
        contradiction_records.append(
            {
                "user_id": uid,
                "positive_text_preserved_from_raw_v2": positive_text,
                "supported_negative_genres": row["supported_negative_genres"],
                "positive_negative_lexical_overlap": lexical_overlap,
                "overlap_genre_statistics": {
                    genre: evidence[uid]["genre_statistics"][genre]
                    for genre in lexical_overlap
                },
                "mechanism": (
                    "Raw V2 inferred a broad positive genre preference from selected "
                    "liked examples/subgenres. Generic repair independently rendered "
                    "the aggregate strict negative genre set and was prohibited from "
                    "editing the preserved positive prose."
                ),
            }
        )

    # Preserve the human privacy adjudication and add the exact raw/final text and
    # whether negative repair could have affected the title-bearing positive span.
    privacy_cases = {
        int(case["user_id"]): case for case in privacy["cases"]
    }
    title_leak_records = []
    for uid in sorted(TITLE_LEAK_IDS):
        row = next(item for item in rows if int(item["user_id"]) == uid)
        plan = plans[uid]
        title_leak_records.append(
            {
                "user_id": uid,
                "automated_exact_history_title_matches": privacy_cases[uid][
                    "automated_matches"
                ],
                "manual_rationale": privacy_cases[uid]["manual_rationale"],
                "raw_summary": row["raw_summary"],
                "final_summary": row["final_summary"],
                "summary_changed_by_negative_repair": row["changed"],
                "supplied_history_titles": plan["titles"],
                "mechanism": (
                    "The title appeared verbatim in the private title-bearing history "
                    "and was copied into positive/example prose despite the prohibition. "
                    "Negative-only repair intentionally did not alter that span."
                ),
            }
        )

    report = {
        "scope": {
            "offline_only": True,
            "openai_calls": 0,
            "tmdb_used": False,
            "source_artifacts_modified": False,
            "users": len(rows),
            "changed_users": len(changed),
            "unchanged_users": len(rows) - len(changed),
        },
        "primary_failure_types_mutually_exclusive": {
            key: count_record(value) for key, value in primary_counts.items()
        },
        "semantic_vs_formatting": {
            "semantic_repair_users": count_record(len(semantic_users)),
            "formatting_only_users": count_record(len(formatting_only_users)),
        },
        "operation_counts_overlapping": {
            key: {
                "operation_occurrences": operation_occurrences[key],
                **count_record(operation_users[key]),
            }
            for key in sorted(operation_users)
        },
        "evidence_strata": {
            "all_users": dict(sorted(evidence_counts_all.items())),
            "changed_users": dict(sorted(evidence_counts_changed.items())),
        },
        "conflicting_rating_evidence": {
            "users": len(conflict_ids),
            "semantic_repair_users": len(conflict_ids & semantic_users),
            "semantic_repair_rate_percent": pct(
                len(conflict_ids & semantic_users), len(conflict_ids)
            ),
            "nonconflict_users": len(rows) - len(conflict_ids),
            "nonconflict_semantic_repair_users": len(semantic_users - conflict_ids),
            "nonconflict_semantic_repair_rate_percent": pct(
                len(semantic_users - conflict_ids), len(rows) - len(conflict_ids)
            ),
            "interpretation": (
                "Mixed genre evidence raises the failure rate, but is not sufficient "
                "to explain it: semantic repair remains common without conflicts."
            ),
        },
        "repaired_span_content_overlapping_user_counts": {
            key: count_record(len(users)) for key, users in content_flag_users.items()
        },
        "repaired_span_content_exclusive_operation_category": {
            key: {
                "users": len(exclusive_category_users[key]),
                "operation_occurrences": exclusive_category_ops[key],
                "positive_word_reduction_across_operations": exclusive_category_word_delta[
                    key
                ],
            }
            for key in sorted(exclusive_category_ops)
        },
        "length": {
            "raw_mean_words": 112.44,
            "final_mean_words": 85.22,
            "mean_reduction_words": 27.22,
            "mean_reduction_percent": round(100 * 27.22 / 112.44, 2),
            "aggregate_raw_words": sum(row["raw_word_count"] for row in rows),
            "aggregate_final_words": sum(row["final_word_count"] for row in rows),
            "aggregate_word_reduction": sum(
                row["raw_word_count"] - row["final_word_count"] for row in rows
            ),
        },
        "material_compression": {
            "users": len(materially_compressed),
            "percent_of_all": pct(len(materially_compressed), len(rows)),
            "aggregate_raw_words": sum(row["raw_word_count"] for row in materially_compressed),
            "aggregate_final_words": sum(
                row["final_word_count"] for row in materially_compressed
            ),
            "aggregate_word_reduction": sum(
                row["raw_word_count"] - row["final_word_count"]
                for row in materially_compressed
            ),
            "semantic_content_flags_lost_or_replaced_user_counts_overlapping": compressed_flag_counts,
            "exclusive_operation_category_user_counts_overlapping": compressed_category_counts,
            "exclusive_operation_category_operation_counts": dict(
                sorted(compressed_category_operation_counts.items())
            ),
            "exclusive_operation_category_positive_word_reduction": dict(
                sorted(compressed_category_word_delta.items())
            ),
            "semantically_complete": compression["semantically_complete"],
            "semantically_incomplete": compression["semantically_incomplete"],
            "incomplete_user_ids": [
                case["user_id"]
                for case in compression["cases"]
                if not case["semantically_complete"]
            ],
        },
        "supported_content_preservation": {
            "positive_sentences_preserved_users": sum(
                bool(row["positive_sentences_preserved"]) for row in rows
            ),
            "supported_negative_evidence_users": sum(
                bool(row["supported_negative_genres"]) for row in rows
            ),
            "supported_negative_evidence_omitted_users": sum(
                bool(row["supported_negative_evidence_omitted"]) for row in rows
            ),
            "users_where_repaired_span_also_mentioned_a_supported_genre": len(
                supported_semantics_touched_users
            ),
            "interpretation": (
                "Some replaced broad spans contained supported genre tokens, but the "
                "generic replacement retained the complete licensed genre set. No "
                "supported positive sentence or supported negative genre was omitted."
            ),
        },
        "manual_sample_contradictions": {
            "count": len(contradiction_records),
            "manual_sample_denominator": 120,
            "percent": pct(len(contradiction_records), 120),
            "records_file": "contradiction_cases.jsonl",
        },
        "confirmed_title_leaks": {
            "count": len(title_leak_records),
            "percent_of_all": pct(len(title_leak_records), len(rows)),
            "records_file": "title_leak_cases.jsonl",
        },
        "counterfactual_smallest_change": {
            "negative_semantic_repair_users_addressed": len(semantic_users),
            "remaining_observed_formatting_only_users_before_deterministic_wrapper": len(
                formatting_only_users
            ),
            "recommendation": (
                "Move ownership of the negative slot out of free-form generation: "
                "append exactly one deterministic supported-genre sentence or one "
                "deterministic abstention from the precomputed evidence object. Keep "
                "V2 positive prose unchanged. Render the Summary prefix in code."
            ),
        },
    }

    write_jsonl(OUTPUT / "per_user_root_causes.jsonl", per_user)
    with (OUTPUT / "per_user_root_causes.csv").open("w", newline="") as handle:
        fieldnames = [
            "user_id",
            "evidence_category",
            "primary_failure_type",
            "why_repair_was_required",
            "semantic_repair",
            "formatting_only",
            "raw_word_count",
            "final_word_count",
            "absolute_word_reduction",
            "materially_compressed",
            "operation_types",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in per_user:
            writer.writerow(
                {
                    key: record[key]
                    for key in fieldnames
                    if key != "operation_types"
                }
                | {
                    "operation_types": ";".join(
                        op["type"] for op in record["all_operations"]
                    )
                }
            )
    write_jsonl(OUTPUT / "contradiction_cases.jsonl", contradiction_records)
    write_jsonl(OUTPUT / "title_leak_cases.jsonl", title_leak_records)
    (OUTPUT / "root_cause_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (OUTPUT / "README.md").write_text(
        "# V2 1,000-user offline root-cause analysis\n\n"
        "Read-only inputs: completed production calibration v002/v008. No model "
        "calls, TMDB data, summary generation, artifact mutation, or training.\n\n"
        "- `root_cause_report.json`: aggregate counts and diagnostics\n"
        "- `per_user_root_causes.jsonl`: exact reason, evidence, operation, and spans "
        "for each of the 867 changed users\n"
        "- `per_user_root_causes.csv`: compact one-row-per-user failure index\n"
        "- `contradiction_cases.jsonl`: all 11 manual-sample contradictions\n"
        "- `title_leak_cases.jsonl`: all four confirmed title leaks\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
