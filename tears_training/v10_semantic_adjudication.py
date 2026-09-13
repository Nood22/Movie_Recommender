"""Offline, high-precision semantic adjudication for frozen V10 summaries.

This module never calls a model and never changes summary text.  It evaluates
the frozen V10 NONE/WEAK/STRONG evidence contract using the already-materialized
evidence objects.  Rules are deliberately high precision: neutral abstentions,
qualified WEAK claims, and selective/mixed language are not failures.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping

from .artifacts import atomic_write_bytes, atomic_write_json, sha256_file, stable_hash
from . import production_summary_protocol as lexical


ADJUDICATOR_VERSION = "v10-contract-semantic-adjudicator-v2-20260817"
SAMPLE_SEED = 20260817
MANUAL_SAMPLE_SIZE = 120

ABSTENTION_PATTERNS = (
    r"\b(?:available|provided|supplied) history (?:does not|doesn't) "
    r"(?:reveal|indicate|show|provide|support)\b.{0,120}\b(?:negative|dislike)",
    r"\bno (?:clear|strong|consistent|reliable|supported|categorical)\b.{0,100}"
    r"\b(?:negative preference|dislike|dislikes)\b",
    r"\bno (?:supplied )?negative evidence\b",
    r"\b(?:cannot|can't|should not) (?:infer|assert|claim)\b.{0,100}\bdislike",
    r"\bnot (?:enough|sufficient) evidence\b.{0,100}\b(?:negative|dislike)",
    r"\bnot (?:strongly )?supported as (?:a )?negative preference\b",
    r"\bdo not justify asserting dislikes\b",
    r"\b(?:there is|there are|the (?:available )?(?:history|record))\b.{0,100}"
    r"\bno\b.{0,70}\b(?:negative preference|dislikes?|aversion)\b",
    r"\bno\b.{0,60}\b(?:negative preference|dislikes?|aversion)\b.{0,80}"
    r"\b(?:evident|supported|clear|reliable|revealed|warranted)\b",
    r"\b(?:history|record)\b.{0,100}\bdoes not\b.{0,80}"
    r"\b(?:negative preference|dislikes?|aversion)\b",
    r"\bno other categorical dislikes? (?:are )?warranted\b",
)
META_NEGATION_PATTERNS = (
    r"\bshould not be treated as\b.{0,100}\b(?:preference|dislike)",
    r"\bnot be (?:read|interpreted|treated|taken) as\b.{0,100}\bdislike",
    r"\bnot (?:a )?(?:clear|strong|categorical|definitive) dislike\b",
    r"\bno basis (?:to|for)\b.{0,80}\bdislike",
)
QUALIFIER_PATTERN = re.compile(
    r"\b(?:weak|limited|tentative|cautious|caution|suggests?|may|might|seems?|"
    r"possible|possibly|occasional(?:ly)?|some|certain|particular|specific|"
    r"a few|few|couple|one|isolated|less|reservations?|not definitive|"
    r"not categorical|narrow|selective|within|when|depending|mixed|varies|"
    r"history is limited|evidence is limited|examples?)\b",
    re.IGNORECASE,
)
RECONCILIATION_PATTERN = re.compile(
    r"\b(?:selective|mixed|within|certain|particular|specific|some|when|depending|"
    r"rather than|not (?:all|categorical|definitive)|subtype|flavo(?:u)?rs?|"
    r"treatments?|examples?|occasionally|can enjoy|while still|even though)\b",
    re.IGNORECASE,
)
STRONG_BROAD_NEGATIVE_PATTERN = re.compile(
    r"\b(?:strongly|clearly|categorically|consistently|broadly)\b.{0,50}"
    r"\b(?:dislikes?|does not enjoy|avoids?|averse|not a fan)|"
    r"\b(?:dislikes?|does not enjoy|avoids?|averse to|not a fan of)\b",
    re.IGNORECASE,
)
POSITIVE_PATTERN = re.compile(
    r"\b(?:likes?|enjoys?|prefers?|favors?|favours?|drawn to|fond of|"
    r"enthusiastic about|appreciates?)\b",
    re.IGNORECASE,
)
ADDITIONAL_NEGATIVE_PATTERN = re.compile(
    r"\b(?:disfavo(?:u)?rs?|rejects?|rejection of|aversion to|unwelcome|"
    r"unlikely to appeal|reluctance toward|negative pattern|negative evidence)\b",
    re.IGNORECASE,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    value = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
    if path.exists() and path.read_bytes() != value:
        raise RuntimeError(f"Refusing to overwrite different adjudication artifact: {path}")
    if not path.exists():
        atomic_write_bytes(path, value)


def _sentences(summary: str) -> list[str]:
    value = re.sub(r"^Summary\s*:\s*", "", summary.strip(), flags=re.I)
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", value) if part.strip()]


def _clauses(sentence: str) -> list[str]:
    # Contrastive boundaries keep "likes X, but dislikes Y" from assigning both
    # polarities to every genre in the full sentence.
    return [
        part.strip(" ,;:")
        for part in re.split(
            r"\s*(?:;|\bbut\b|\bhowever\b|\bwhereas\b|\bwhile\b|\balthough\b)\s*",
            sentence,
            flags=re.I,
        )
        if part.strip(" ,;:")
    ]


def is_abstention_or_meta(text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in ABSTENTION_PATTERNS + META_NEGATION_PATTERNS)


def _claim_clauses(summary: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    negative: list[dict[str, Any]] = []
    positive: list[dict[str, Any]] = []
    abstentions: list[str] = []
    for sentence in _sentences(summary):
        if is_abstention_or_meta(sentence):
            abstentions.append(sentence)
            # A sentence can contain a real negative claim before a semicolon
            # and a meta-abstention after it; classify its clauses separately.
        for clause in _clauses(sentence):
            if is_abstention_or_meta(clause):
                continue
            genres = sorted(lexical._mentioned_genres(clause))
            if lexical._is_user_negative_claim(clause) or ADDITIONAL_NEGATIVE_PATTERN.search(clause):
                negative.append(
                    {
                        "text": clause,
                        "genres": genres,
                        "qualified": bool(QUALIFIER_PATTERN.search(clause)),
                        "reconciled": bool(RECONCILIATION_PATTERN.search(clause)),
                        "broad_categorical": bool(STRONG_BROAD_NEGATIVE_PATTERN.search(clause)),
                    }
                )
            elif POSITIVE_PATTERN.search(clause):
                positive.append(
                    {
                        "text": clause,
                        "genres": genres,
                        "qualified": bool(QUALIFIER_PATTERN.search(clause)),
                        "reconciled": bool(RECONCILIATION_PATTERN.search(clause)),
                    }
                )
    return negative, positive, abstentions


def adjudicate(summary: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    status = str(evidence["negative_evidence_status"])
    negative, positive, abstentions = _claim_clauses(summary)
    supported = set(evidence["genre_classifications"]["supported_negative"])
    mixed = set(evidence["genre_classifications"]["mixed_conflicting"])
    failures: set[str] = set()
    details: dict[str, Any] = {}

    if status == "NONE" and negative:
        failures.add("fabricated_negative_for_none")

    if status == "WEAK":
        overbroad = [
            claim for claim in negative
            if claim["broad_categorical"] and not claim["qualified"] and not claim["reconciled"]
        ]
        if overbroad:
            failures.add("overbroad_negative_for_weak")
            details["overbroad_weak_claims"] = [claim["text"] for claim in overbroad]

    if status == "STRONG" and supported and not negative:
        failures.add("inappropriate_abstention_for_strong")

    unsupported_claims: list[dict[str, Any]] = []
    mixed_claims: list[dict[str, Any]] = []
    if status == "STRONG":
        for claim in negative:
            if not claim["broad_categorical"] or claim["qualified"] or claim["reconciled"]:
                continue
            genres = set(claim["genres"])
            unsupported = sorted(genres - supported)
            if unsupported:
                unsupported_claims.append({"text": claim["text"], "genres": unsupported})
            conflicted = sorted(genres & mixed)
            if conflicted:
                mixed_claims.append({"text": claim["text"], "genres": conflicted})
    if unsupported_claims:
        failures.add("unsupported_categorical_negative_for_strong")
        details["unsupported_claims"] = unsupported_claims
    if mixed_claims:
        failures.add("categorical_negative_on_mixed_genre")
        details["mixed_claims"] = mixed_claims

    contradiction_genres: set[str] = set()
    contradiction_pairs: list[dict[str, Any]] = []
    for liked in positive:
        if liked["qualified"] or liked["reconciled"]:
            continue
        for disliked in negative:
            if (
                not disliked["broad_categorical"]
                or disliked["qualified"]
                or disliked["reconciled"]
            ):
                continue
            overlap = set(liked["genres"]) & set(disliked["genres"])
            if overlap:
                contradiction_genres |= overlap
                contradiction_pairs.append(
                    {
                        "genres": sorted(overlap),
                        "positive": liked["text"],
                        "negative": disliked["text"],
                    }
                )
    if contradiction_genres:
        failures.add("substantive_positive_negative_contradiction")
        details["contradiction_pairs"] = contradiction_pairs

    return {
        "outcome": "confirmed_failure" if failures else "semantic_pass",
        "failure_types": sorted(failures),
        "negative_claims": negative,
        "positive_claims": positive,
        "abstentions": abstentions,
        "supported_negative_genres": sorted(supported),
        "mixed_genres": sorted(mixed),
        "details": details,
    }


def _manual_sample(rows: list[dict[str, Any]], size: int = MANUAL_SAMPLE_SIZE) -> list[dict[str, Any]]:
    rng = random.Random(SAMPLE_SEED)
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['negative_evidence_status']}/{row['new_outcome']}/old_{'flagged' if row['old_screen_flagged'] else 'pass'}"
        strata[key].append(row)
    chosen: list[dict[str, Any]] = []
    # Ensure every populated status/outcome/old-screen combination is covered.
    for key in sorted(strata):
        values = sorted(strata[key], key=lambda row: row["user_id"])
        rng.shuffle(values)
        chosen.extend(values[: min(10, len(values))])
    chosen_ids = {row["user_id"] for row in chosen}
    pool = [row for row in rows if row["user_id"] not in chosen_ids]
    rng.shuffle(pool)
    chosen.extend(pool[: max(0, size - len(chosen))])
    if len(chosen) > size:
        # Retain all confirmed-failure strata first, then deterministic coverage.
        failures = [row for row in chosen if row["new_outcome"] == "confirmed_failure"]
        passes = [row for row in chosen if row["new_outcome"] != "confirmed_failure"]
        chosen = failures[: min(len(failures), size // 2)] + passes[: size - min(len(failures), size // 2)]
    return sorted(chosen, key=lambda row: row["user_id"])


def run(root: Path, shard: int = 0) -> dict[str, Any]:
    evidence_rows = _jsonl(root / "evidence" / f"shard-{shard:03d}.jsonl")
    evidence = {int(row["user_id"]): row for row in evidence_rows}
    old_rows = []
    for suffix in ("accepted", "failed"):
        old_rows.extend(
            _jsonl(root / "validated" / "attempts" / f"shard-{shard:03d}-attempt-00-{suffix}.jsonl")
        )
    old = {int(row["user_id"]): row for row in old_rows}
    if set(old) != set(evidence) or len(old) != 15_000:
        raise RuntimeError("Shard evidence and summaries are not aligned at 15,000 users")

    output: list[dict[str, Any]] = []
    for user_id in sorted(old):
        source = old[user_id]
        result = adjudicate(source["final_summary"], evidence[user_id])
        output.append(
            {
                "user_id": user_id,
                "negative_evidence_status": evidence[user_id]["negative_evidence_status"],
                "old_screen_flagged": bool(source["failures"]),
                "old_failure_types": source["failures"],
                "new_outcome": result["outcome"],
                "new_failure_types": result["failure_types"],
                "adjudication": result,
                "summary": source["final_summary"],
                "evidence": {
                    "positive_example_count": evidence[user_id]["positive_example_count"],
                    "negative_example_count": evidence[user_id]["negative_example_count"],
                    "genre_classifications": evidence[user_id]["genre_classifications"],
                    "genre_statistics": evidence[user_id]["genre_statistics"],
                    "negative_examples": evidence[user_id]["inference_payload"]["negative_evidence"],
                },
            }
        )
    directory = root / "reports" / "shards" / f"shard-{shard:03d}-semantic-adjudication-v2"
    adjudications_path = directory / "all_adjudications.jsonl"
    _write_jsonl(adjudications_path, output)
    sample = _manual_sample(output)
    sample_path = directory / "manual_audit_queue.jsonl"
    _write_jsonl(sample_path, sample)

    failures = [row for row in output if row["new_outcome"] == "confirmed_failure"]
    old_flagged = {row["user_id"] for row in output if row["old_screen_flagged"]}
    new_flagged = {row["user_id"] for row in failures}
    report = {
        "adjudicator_version": ADJUDICATOR_VERSION,
        "shard": shard,
        "users": len(output),
        "semantic_passes": len(output) - len(failures),
        "confirmed_semantic_failures": len(failures),
        "failure_type_counts": dict(
            sorted(Counter(kind for row in failures for kind in row["new_failure_types"]).items())
        ),
        "old_keyword_screen_flagged": len(old_flagged),
        "old_keyword_screen_false_positives_before_manual_audit": len(old_flagged - new_flagged),
        "new_failures_missed_by_old_screen": len(new_flagged - old_flagged),
        "status_outcome_counts": {
            f"{status}/{outcome}": count
            for (status, outcome), count in sorted(
                Counter((row["negative_evidence_status"], row["new_outcome"]) for row in output).items()
            )
        },
        "manual_audit": {
            "seed": SAMPLE_SEED,
            "sample_size": len(sample),
            "queue_path": str(sample_path),
            "strata": dict(
                sorted(
                    Counter(
                        f"{row['negative_evidence_status']}/{row['new_outcome']}/old_{'flagged' if row['old_screen_flagged'] else 'pass'}"
                        for row in sample
                    ).items()
                )
            ),
            "completed": False,
        },
        "artifacts": {
            "all_adjudications": str(adjudications_path),
            "all_adjudications_sha256": sha256_file(adjudications_path),
        },
        "no_api_calls": True,
        "no_summary_changes": True,
        "no_retries": True,
        "shard_1_submitted": False,
    }
    report["fingerprint"] = stable_hash(report)
    atomic_write_json(directory / "automated_adjudication_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.shard), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
