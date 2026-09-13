"""Live before/after evaluation for the online Task 1a summary prompt.

This script performs generation only when invoked explicitly. It never deploys,
changes recommender artifacts, or repairs generated preference semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace
from typing import Any

from openai import OpenAI

import pilot_api
from tears_training.evidence_gated_summary_harness_v11 import (
    build_separated_evidence,
    validate_summary_contract,
)
from tears_training.summaries import (
    SUMMARY_SCHEMA,
    SYSTEM_PROMPT,
    render_history_prompt,
    validate_text,
)


CASES: dict[str, list[tuple[str, float, str]]] = {
    "one_positive_item": [("CourtroomEchoZ", 5, "Drama")],
    "one_negative_item": [("CellarFrightZ", 1, "Horror")],
    "one_positive_plus_one_negative": [
        ("CourtroomEchoZ", 5, "Drama"),
        ("CellarFrightZ", 1, "Horror"),
    ],
    "sparse_weak_negative_evidence": [
        ("ExpeditionTrailZ", 2, "Adventure"),
        ("OrdinaryAfternoonZ", 3, "Drama"),
    ],
    "rich_positive_evidence": [
        ("WarmReunionZ", 5, "Drama|Romance"),
        ("SecondChanceZ", 4, "Drama|Romance"),
        ("FamilyLetterZ", 5, "Drama"),
        ("GentleDetourZ", 4, "Comedy|Drama"),
        ("SharedTableZ", 5, "Comedy|Drama"),
        ("BrightNeighborZ", 4, "Comedy")
    ],
    "rich_mixed_evidence": [
        ("WarmReunionZ", 5, "Drama|Romance"),
        ("SecondChanceZ", 4, "Drama|Romance"),
        ("FamilyLetterZ", 5, "Drama"),
        ("CellarFrightZ", 1, "Horror"),
        ("DarkPassageZ", 1, "Horror|Thriller"),
        ("FinalScreamZ", 2, "Horror"),
    ],
    "conflicting_genres": [
        ("CityPortraitZ", 5, "Drama"),
        ("FamilyLetterZ", 4, "Drama"),
        ("WarmReunionZ", 5, "Drama"),
        ("BleakMonologueZ", 1, "Drama"),
        ("EndlessArgumentZ", 2, "Drama"),
        ("ColdFarewellZ", 1, "Drama"),
    ],
}

SALIENT_COVERAGE: dict[str, list[tuple[str, ...]]] = {
    "one_positive_item": [("drama",), ("courtroom", "legal", "moral")],
    "one_negative_item": [("horror",), ("fear", "fright", "scare", "intense")],
    "one_positive_plus_one_negative": [
        ("drama",),
        ("courtroom", "legal", "moral"),
        ("horror",),
        ("fear", "fright", "scare", "suspense"),
    ],
    "sparse_weak_negative_evidence": [("adventure",), ("expedition", "exploration")],
    "rich_positive_evidence": [
        ("drama",),
        ("comedy", "comedies", "humor"),
        ("family", "relationship", "reunion", "second chance"),
        ("warm", "gentle", "humor"),
    ],
    "rich_mixed_evidence": [
        ("drama",),
        ("family", "relationship", "reunion"),
        ("horror",),
        ("fear", "scare", "gore", "scream", "dread"),
    ],
    "conflicting_genres": [
        ("drama",),
        ("mixed", "selective", "varies", "contrasting"),
        ("warm", "family", "reunion"),
        ("bleak", "conflict", "argument", "cold", "farewell"),
    ],
}


def history_lines(items: list[tuple[str, float, str]]) -> list[str]:
    return [
        f"- title={title!r}; private_rating={rating:g}/5; genres={genres}"
        for title, rating, genres in items
    ]


def evidence(items: list[tuple[str, float, str]]) -> dict[str, Any]:
    return build_separated_evidence(
        {
            "user_id": 0,
            "history_hash": "online-evaluation",
            "history_items": len(items),
            "movie_ids": list(range(1, len(items) + 1)),
            "titles": [item[0] for item in items],
            "ratings": [item[1] for item in items],
            "genres": [item[2] for item in items],
        }
    )


def before_prompt(lines: list[str], errors: list[str] | None = None) -> str:
    prompt = render_history_prompt("\n".join(lines), 140, 180)
    prompt += (
        "\n\nBefore returning, perform this contract preflight without adding unsupported "
        "claims: write roughly 35-45 words in each sentence on the first attempt, "
        "preferably 36-40 words per sentence, and aim near 152 words without going "
        "outside 140-180 words total. Sentence one must contain only supported liked "
        "genres, or state that no strong liked genre is supported. Sentence two must "
        "contain only supported liked themes or content, or state that none is supported. "
        "Sentence three must contain only supported disliked genres or styles, or state "
        "that none is supported. Sentence four must contain only supported disliked plot "
        "or content preferences and what other viewers may enjoy, or state that none is "
        "supported. Do not move dislike claims into sentences one or two, and do not move "
        "the viewer's like claims into sentences three or four. Count the words in the "
        "complete four-sentence summary before returning it. For private evidence "
        "direction, values at or below 2 support dislikes, values at or above 4 support "
        "likes, and middle values are neutral or inconclusive; apply one item's direction "
        "consistently rather than splitting its genres into opposing preferences. Use each "
        "private value only to infer supported preference direction and strength, then "
        "express the underlying preference semantically only as a like, dislike, relative "
        "preference, or abstention. Do not explain a conclusion by referring to how the "
        "preference was measured. The summary must not contain rating metadata. For sparse "
        "or negative-only evidence, reach the length target with grounded uncertainty and "
        "abstention language in unsupported slots; never invent preferences merely to add "
        "words. A dislike does not imply liking its opposite."
    )
    if errors:
        prompt += (
            "\n\nThe previous draft was rejected by the unchanged TEARS validator for: "
            + ", ".join(errors)
            + ". Generate a fresh replacement from the original private evidence above. "
            "Preserve preference direction and do not add unsupported genres, themes, plot "
            "elements, or viewing needs. Write exactly four complete sentences totaling "
            "140-180 words and return all four sentences."
        )
    return prompt


def call(client: OpenAI, system: str, user: str) -> tuple[str, float]:
    start = perf_counter()
    response = client.responses.create(
        model=pilot_api.SUMMARY_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        reasoning={"effort": "minimal"},
        text={
            "format": {
                "type": "json_schema",
                "name": "viewer_profile",
                "strict": True,
                "schema": SUMMARY_SCHEMA,
            },
            "verbosity": "high",
        },
        max_output_tokens=600,
        store=False,
    )
    return str(json.loads(response.output_text)["summary"]).strip(), (
        perf_counter() - start
    ) * 1000


def generate_before(
    client: OpenAI, lines: list[str], config: Any, titles: list[str]
) -> tuple[str, float, list[str]]:
    errors: list[str] | None = None
    total_ms = 0.0
    summary = ""
    for _ in range(pilot_api.SUMMARY_MAX_ATTEMPTS):
        summary, elapsed = call(client, SYSTEM_PROMPT, before_prompt(lines, errors))
        total_ms += elapsed
        errors = validate_text(summary, config, titles)
        if not errors:
            break
    return summary, total_ms, errors or []


def sentence_count(summary: str) -> int:
    body = re.sub(r"^Summary:\s*", "", summary.strip())
    return len([part for part in re.split(r"(?<=[.!?])\s+", body) if part.strip()])


def leakage(summary: str, config: Any, titles: list[str]) -> dict[str, bool]:
    errors = validate_text(summary, config, titles)
    return {
        "title": any(error.startswith("title_leakage:") for error in errors),
        "year": "year_leakage" in errors,
        "rating": "rating_leakage" in errors,
    }


def grounding(summary: str, case_evidence: dict[str, Any]) -> dict[str, Any]:
    result = validate_summary_contract(summary, case_evidence)
    return {
        "passed": result.pass_contract,
        "details": {
            key: value
            for key, value in result.__dict__.items()
            if key != "pass_contract" and value
        },
    }


def salient_coverage(case: str, summary: str) -> dict[str, Any]:
    lowered = summary.lower()
    missing = [
        list(group)
        for group in SALIENT_COVERAGE[case]
        if not any(term in lowered for term in group)
    ]
    return {"passed": not missing, "missing_keyword_groups": missing}


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Online Task 1a prompt before/after evaluation",
        "",
        f"Model: `{payload['model']}`. Latency is end-to-end generation latency and includes retries.",
        "",
        "Frozen V10 production reference: 1,000 summaries; 59–249 words (median 123); "
        "3–8 sentences; 196/1,000 within 140–180 words; 846/1,000 with four sentences. "
        "The V10 prompt explicitly did not force an exact word count or four sentences.",
        "",
        f"Overall AFTER gate: grounding `{'PASS' if payload['all_after_grounded'] else 'FAIL'}`, "
        f"salient coverage `{'PASS' if payload['all_after_salient_coverage'] else 'FAIL'}`, "
        f"leakage `{'PASS' if payload['all_after_leakage_free'] else 'FAIL'}`, "
        f"validator `{'PASS' if payload['all_after_validator_passed'] else 'FAIL'}`.",
        "",
    ]
    for row in payload["cases"]:
        lines.extend([f"## {row['case']}", ""])
        lines.append("| Version | Words | Sentences | Grounding | Salient coverage | Leakage (title/year/rating) | Validator | Latency |")
        lines.append("|---|---:|---:|---|---|---|---|---:|")
        for version in ("before", "after"):
            result = row[version]
            leak = result["leakage"]
            leak_text = "/".join("FAIL" if leak[key] else "PASS" for key in ("title", "year", "rating"))
            grounding_text = "PASS" if result["grounding"]["passed"] else "FAIL"
            coverage_text = "PASS" if result["salient_coverage"]["passed"] else "FAIL"
            validator_text = "PASS" if result["validator"]["passed"] else "FAIL: " + ", ".join(result["validator"]["errors"])
            lines.append(
                f"| {version.upper()} | {result['word_count']} | {result['sentence_count']} | "
                f"{grounding_text} | {coverage_text} | {leak_text} | {validator_text} | {result['latency_ms']:.1f} ms |"
            )
        lines.extend(["", "**BEFORE**", "", "> " + row["before"]["summary"], ""])
        if row["before"]["grounding"]["details"]:
            lines.append(
                "Grounding findings: `"
                + json.dumps(row["before"]["grounding"]["details"], sort_keys=True)
                + "`."
            )
            lines.append("")
        lines.extend(["**AFTER**", "", "> " + row["after"]["summary"], ""])
        if row["after"]["grounding"]["details"]:
            lines.append(
                "Grounding findings: `"
                + json.dumps(row["after"]["grounding"]["details"], sort_keys=True)
                + "`."
            )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = OpenAI()
    config = SimpleNamespace(summaries=SimpleNamespace(min_words=140, max_words=180))
    rows = []
    for name, items in CASES.items():
        lines = history_lines(items)
        titles = [item[0] for item in items]
        case_evidence = evidence(items)
        before, before_ms, before_errors = generate_before(
            client, lines, config, titles
        )
        start = perf_counter()
        after, attempts = pilot_api._generate_valid_tears_summary(
            client,
            [pilot_api._render_online_inference_payload(case_evidence)],
            SimpleNamespace(config=config),
            titles,
            [],
            grounding_evidence=case_evidence,
        )
        after_ms = (perf_counter() - start) * 1000
        rows.append(
            {
                "case": name,
                "negative_evidence_status": case_evidence["negative_evidence_status"],
                "before": {
                    "summary": before,
                    "word_count": len(before.split()),
                    "sentence_count": sentence_count(before),
                    "grounding": grounding(before, case_evidence),
                    "salient_coverage": salient_coverage(name, before),
                    "leakage": leakage(before, config, titles),
                    "validator": {"passed": not before_errors, "errors": before_errors},
                    "latency_ms": round(before_ms, 1),
                },
                "after": {
                    "summary": after,
                    "word_count": len(after.split()),
                    "sentence_count": sentence_count(after),
                    "grounding": grounding(after, case_evidence),
                    "salient_coverage": salient_coverage(name, after),
                    "leakage": leakage(after, config, titles),
                    "validator": {
                        "passed": not attempts[-1]["validator_errors"],
                        "errors": attempts[-1]["validator_errors"],
                    },
                    "latency_ms": round(after_ms, 1),
                    "attempts": len(attempts),
                },
            }
        )
    payload = {
        "model": pilot_api.SUMMARY_MODEL,
        "cases": rows,
        "all_after_grounded": all(row["after"]["grounding"]["passed"] for row in rows),
        "all_after_salient_coverage": all(
            row["after"]["salient_coverage"]["passed"] for row in rows
        ),
        "all_after_leakage_free": all(
            not any(row["after"]["leakage"].values()) for row in rows
        ),
        "all_after_validator_passed": all(row["after"]["validator"]["passed"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
