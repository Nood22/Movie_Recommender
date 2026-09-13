"""Evidence-gated Emiliano-style summary-generation harness.

This experimental protocol separates positive and negative rating evidence
before prompting. It performs no semantic post-generation repair: only privacy
sanitization and non-semantic formatting normalization are permitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from . import final_summaries as emiliano
from . import final_summaries_v4 as frozen_evidence
from .final_production_summary_protocol import sanitize_privacy


PROTOCOL_VERSION = "tears-emiliano-evidence-gated-experimental-20260816"
EVIDENCE_SCHEMA_VERSION = "tears-separated-rating-evidence-v1"

ORIGINAL_EMILIANO_PROMPT = emiliano.EMILIANO_SYSTEM_PROMPT
EVIDENCE_GATED_PROMPT = """Task: You will now help me generate a highly detailed summary based on the broad common elements of movies.
Do not comment on the year of production. Do not mention any specific movie titles or actors.
Do not comment on the ratings but use qualitative speech such as the user likes, or the user does not enjoy.
Remember you are an expert crafter of these summaries so any other expert should be able to craft a similar summary to yours given this task.

Write a natural-language preference profile that preserves this intended semantic organization:
1. Specific details about genres the user enjoys.
2. Specific details of plot points, characteristics, or narrative elements the user seems to enjoy.
3. Specific details about genres or styles the user does not enjoy, but only when supported by NEGATIVE EVIDENCE.
4. Specific details of plot points or content the user does not enjoy, but only when supported by NEGATIVE EVIDENCE.

Use the separated evidence blocks under the following strict contract:
- Infer positive preferences only from POSITIVE EVIDENCE and SUPPORTED POSITIVE GENRES.
- Infer negative preferences only from NEGATIVE EVIDENCE and the supplied NEGATIVE EVIDENCE STATUS.
- Never infer a dislike from the absence of positive ratings.
- Never use a positively rated movie as evidence for a dislike.
- Never convert MIXED/CONFLICTING GENRES into a categorical positive or negative preference.
- Never invent negative content simply to fill the four-part organization.
- When NEGATIVE EVIDENCE STATUS is NONE, do not fabricate a dislike. State naturally that the available history does not reveal a clear negative preference, or omit unsupported negative detail.
- When NEGATIVE EVIDENCE STATUS is WEAK, keep any negative inference narrow, cautious, and tentative. One or two examples must not become a broad genre dislike, and genres not listed as SUPPORTED NEGATIVE GENRES must not be called categorical dislikes.
- When NEGATIVE EVIDENCE STATUS is STRONG, express only repeated or coherent negative patterns supported by the supplied negative evidence and SUPPORTED NEGATIVE GENRES. Do not add unrelated dislikes.
- Movie titles in the evidence are private evidence only. Never copy them into the summary.
- Never include years or numeric ratings in the summary.

Preserve a rich, natural, editable preference-profile style where evidence supports it. Do not pad to an exact word count and do not fabricate content for length. Do not force exactly four sentences.
Begin with `Summary:`."""

PROMPT_SHA256 = hashlib.sha256(EVIDENCE_GATED_PROMPT.encode()).hexdigest()


def _genres(value: str) -> list[str]:
    return sorted(
        {
            genre.strip()
            for genre in str(value).split("|")
            if genre.strip() and genre.strip() != "(no genres listed)"
        }
    )


def _private_title(title: str) -> str:
    """Remove the catalog year while retaining title evidence for the model."""

    return re.sub(r"\s*\((?:18|19|20)\d{2}\)\s*$", "", str(title)).strip()


def _example(
    movie_id: int,
    title: str,
    genres: str,
    *,
    polarity: str,
) -> dict[str, Any]:
    return {
        "source_movie_id": int(movie_id),
        "private_movie_title": _private_title(title),
        "movielens_genres": _genres(genres),
        "preference_signal": polarity,
    }


def build_separated_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build the complete auditable and inference evidence objects for one user."""

    required = ("movie_ids", "titles", "ratings", "genres")
    if any(key not in record for key in required):
        raise RuntimeError(f"Record lacks required history fields: {required}")
    lengths = {len(record[key]) for key in required}
    if len(lengths) != 1:
        raise RuntimeError("History title/rating/genre arrays are not aligned")

    frozen = frozen_evidence.extract_user_evidence(dict(record))
    positive_examples: list[dict[str, Any]] = []
    negative_examples: list[dict[str, Any]] = []
    neutral_count = 0
    for movie_id, title, rating, genres in zip(
        record["movie_ids"], record["titles"], record["ratings"], record["genres"]
    ):
        value = float(rating)
        if value >= 4.0:
            positive_examples.append(
                _example(movie_id, title, genres, polarity="POSITIVE")
            )
        elif value <= 2.5:
            negative_examples.append(
                _example(movie_id, title, genres, polarity="NEGATIVE")
            )
        else:
            neutral_count += 1

    supported_negative = list(frozen["supported_negative_genres"])
    if not negative_examples:
        status = "NONE"
    elif supported_negative:
        status = "STRONG"
    else:
        status = "WEAK"
    if status == "NONE" and negative_examples:
        raise RuntimeError("NONE status cannot expose negative examples")
    if status == "STRONG" and not supported_negative:
        raise RuntimeError("STRONG status requires supported negative genres")

    stats = frozen["genre_statistics"]
    classifications: dict[str, list[str]] = {
        "supported_positive": [],
        "supported_negative": [],
        "mixed_conflicting": [],
        "insufficient": [],
    }
    for genre, values in sorted(stats.items()):
        if values["conflicting"]:
            classifications["mixed_conflicting"].append(genre)
        elif values["supported_positive"]:
            classifications["supported_positive"].append(genre)
        elif values["supported_negative"]:
            classifications["supported_negative"].append(genre)
        else:
            classifications["insufficient"].append(genre)
    if set(classifications["supported_positive"]) & set(
        classifications["supported_negative"]
    ):
        raise RuntimeError("A genre cannot be both supported positive and negative")
    if set(classifications["mixed_conflicting"]) & (
        set(classifications["supported_positive"])
        | set(classifications["supported_negative"])
    ):
        raise RuntimeError("Mixed genres must not be presented as supported")

    inference = {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "positive_evidence": positive_examples,
        "negative_evidence": negative_examples if status != "NONE" else [],
        "negative_evidence_status": status,
        "supported_positive_genres": classifications["supported_positive"],
        "supported_negative_genres": classifications["supported_negative"],
        "mixed_conflicting_genres": classifications["mixed_conflicting"],
        "insufficient_genres": classifications["insufficient"],
    }
    return {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "user_id": record.get("user_id"),
        "history_hash": record.get("history_hash"),
        "positive_example_count": len(positive_examples),
        "negative_example_count": len(negative_examples),
        "neutral_excluded_count": neutral_count,
        "negative_evidence_status": status,
        "genre_classifications": classifications,
        "genre_statistics": stats,
        "frozen_evidence_hash": frozen["evidence_hash"],
        "thresholds": frozen_evidence.EVIDENCE_RULES,
        "inference_payload": inference,
    }


def render_evidence_prompt(evidence: Mapping[str, Any]) -> str:
    payload = evidence["inference_payload"]
    positive = payload["positive_evidence"]
    negative = payload["negative_evidence"]
    status = str(payload["negative_evidence_status"])
    if status not in {"NONE", "WEAK", "STRONG"}:
        raise RuntimeError(f"Unknown negative evidence status: {status}")
    if status == "NONE" and negative:
        raise RuntimeError("NONE prompt must not contain negative examples")

    def block(examples: Sequence[Mapping[str, Any]]) -> str:
        if not examples:
            return "(none supplied)"
        rows = []
        for index, example in enumerate(examples, 1):
            genres = " | ".join(example["movielens_genres"]) or "(no genres listed)"
            rows.append(
                f"- E{index:02d}: {example['private_movie_title']}\n"
                f"  MovieLens genres: {genres}\n"
                f"  Preference signal: {example['preference_signal']}"
            )
        return "\n".join(rows)

    return (
        "POSITIVE EVIDENCE:\n"
        + block(positive)
        + "\n\nNEGATIVE EVIDENCE:\n"
        + block(negative)
        + f"\n\nNEGATIVE EVIDENCE STATUS: {status}\n\n"
        + "SUPPORTED POSITIVE GENRES: "
        + json.dumps(payload["supported_positive_genres"])
        + "\nSUPPORTED NEGATIVE GENRES: "
        + json.dumps(payload["supported_negative_genres"])
        + "\nMIXED/CONFLICTING GENRES: "
        + json.dumps(payload["mixed_conflicting_genres"])
        + "\nINSUFFICIENT GENRES: "
        + json.dumps(payload["insufficient_genres"])
    )


def build_response_request(
    record: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    model: str = "gpt-5-mini-2025-08-07",
    max_output_tokens: int = 450,
) -> dict[str, Any]:
    request_identity = {
        "protocol": PROTOCOL_VERSION,
        "history_hash": record["history_hash"],
        "evidence_hash": evidence["frozen_evidence_hash"],
    }
    digest = hashlib.sha256(
        json.dumps(request_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "custom_id": f"eg-user-{record['user_id']}-{digest[:16]}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [
                {"role": "system", "content": EVIDENCE_GATED_PROMPT},
                {"role": "user", "content": render_evidence_prompt(evidence)},
            ],
            "reasoning": {"effort": "minimal"},
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "tears_user_preference_summary",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                        "required": ["summary"],
                        "additionalProperties": False,
                    },
                },
            },
            "max_output_tokens": max_output_tokens,
            "store": False,
        },
    }


@dataclass(frozen=True)
class CleanupResult:
    raw_summary: str
    final_summary: str
    operations: tuple[dict[str, Any], ...]
    title_matches_after: tuple[str, ...]
    year_matches_after: tuple[str, ...]
    numeric_rating_matches_after: tuple[str, ...]
    formatting_issues: tuple[str, ...]


YEAR_PATTERN = re.compile(r"\b(?:18|19|20)\d{2}\b")
RATING_PATTERN = re.compile(
    r"\b(?:rated?|ratings?)(?:\s+(?:as|of))?[^.!?\d]{0,60}"
    r"[0-5](?:\.\d+)?(?:\s*/\s*5|\s*stars?)?\b",
    re.IGNORECASE,
)


def privacy_and_format_cleanup(
    summary: str,
    history_titles: Sequence[str],
) -> CleanupResult:
    """Apply only allowed privacy sanitization and formatting normalization."""

    raw = str(summary).strip()
    value = re.sub(r"^\s*Summary\s*:\s*", "", raw, flags=re.I).strip()
    sanitized, privacy_operations, _ = sanitize_privacy(value, history_titles)
    operations = list(privacy_operations)
    final_summary = "Summary: " + sanitized.strip()
    final_summary = re.sub(r"\s+", " ", final_summary).strip()
    if final_summary != raw and not any(
        operation["type"].startswith("privacy_") for operation in operations
    ):
        operations.append(
            {
                "type": "format_normalize_summary_prefix_or_whitespace",
                "original_span": raw,
                "replacement_span": final_summary,
                "semantic_change": False,
            }
        )
    from .final_production_summary_protocol import title_matches

    title_after = tuple(
        match["matched_text"] for match in title_matches(final_summary, history_titles)
    )
    years = tuple(match.group(0) for match in YEAR_PATTERN.finditer(final_summary))
    ratings = tuple(match.group(0) for match in RATING_PATTERN.finditer(final_summary))
    issues: list[str] = []
    if not final_summary.startswith("Summary: ") or final_summary.count("Summary:") != 1:
        issues.append("invalid_summary_prefix")
    if not sanitized:
        issues.append("empty_summary")
    if re.search(r"\{[^{}]+\}|\b(?:TBD|TODO)\b", final_summary, re.I):
        issues.append("literal_placeholder")
    return CleanupResult(
        raw_summary=raw,
        final_summary=final_summary,
        operations=tuple(operations),
        title_matches_after=title_after,
        year_matches_after=years,
        numeric_rating_matches_after=ratings,
        formatting_issues=tuple(issues),
    )


def prompt_delta() -> dict[str, Any]:
    return {
        "source_protocol": "Emiliano original prompt extracted from prompt_gpt.ipynb",
        "source_prompt": ORIGINAL_EMILIANO_PROMPT,
        "source_prompt_sha256": hashlib.sha256(
            ORIGINAL_EMILIANO_PROMPT.encode()
        ).hexdigest(),
        "adapted_protocol": PROTOCOL_VERSION,
        "adapted_prompt": EVIDENCE_GATED_PROMPT,
        "adapted_prompt_sha256": PROMPT_SHA256,
        "preserved": [
            "opening task wording and broad-common-elements framing",
            "privacy prohibitions for years, movie titles, and actors",
            "qualitative preference language instead of rating narration",
            "expert-crafted natural-language profile tone",
            "four-part liked genre / liked characteristics / disliked genre / disliked characteristics organization",
        ],
        "changed": [
            "positive and negative histories are separated before prompting",
            "NONE/WEAK/STRONG negative status is supplied by code",
            "supported positive, supported negative, mixed/conflicting, and insufficient genres are explicit",
            "negative inference is contractually limited to negative evidence",
            "unsupported negative slots may abstain instead of being fabricated",
            "exact 200-word and exactly-four-sentence pressure is removed",
            "Summary prefix remains requested but may be normalized deterministically",
        ],
        "exact_reproduction_claimed": False,
    }
