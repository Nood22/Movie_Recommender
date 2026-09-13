"""Dominance-aware evidence-gated Emiliano summary harness.

This is a minimal successor to the v10 experimental harness.  It keeps title-
based natural-language inference but gives every MovieLens genre one exclusive
deterministic status and exposes explicit coverage requirements.  Generated
preference semantics are never rewritten after generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from . import evidence_gated_summary_harness as v10
from . import final_summaries_v4 as frozen_evidence


PROTOCOL_VERSION = "tears-emiliano-evidence-gated-genre-contract-20260816"
EVIDENCE_SCHEMA_VERSION = "tears-separated-rating-evidence-v2-dominance"
ORIGINAL_EMILIANO_PROMPT = v10.ORIGINAL_EMILIANO_PROMPT

# The first paragraph and four-part organization remain Emiliano-derived.  The
# delta from v10 is limited to exclusive genre statuses and coverage rules.
EVIDENCE_GATED_PROMPT = """Task: You will now help me generate a highly detailed summary based on the broad common elements of movies.
Do not comment on the year of production. Do not mention any specific movie titles or actors.
Do not comment on the ratings but use qualitative speech such as the user likes, or the user does not enjoy.
Remember you are an expert crafter of these summaries so any other expert should be able to craft a similar summary to yours given this task.

Write a natural-language preference profile that preserves this intended semantic organization:
1. Specific details about genres the user enjoys.
2. Specific details of plot points, characteristics, or narrative elements the user seems to enjoy.
3. Specific details about genres or styles the user does not enjoy, but only when supported by NEGATIVE EVIDENCE.
4. Specific details of plot points or content the user does not enjoy, but only when supported by NEGATIVE EVIDENCE.

Use the separated evidence blocks and deterministic genre classifications under this strict contract:
- Infer positive themes, plots, and characteristics only from POSITIVE EVIDENCE.
- Infer negative themes, plots, and characteristics only from NEGATIVE EVIDENCE and the supplied NEGATIVE EVIDENCE STATUS.
- A categorical positive genre claim may use only SUPPORTED POSITIVE GENRES.
- A categorical negative genre claim may use only SUPPORTED NEGATIVE GENRES.
- Never describe a genre in MIXED GENRES categorically as liked or disliked. If it is useful to mention one, explicitly preserve uncertainty with wording such as mixed, selective, context-dependent, or varying by execution.
- Never turn INSUFFICIENT GENRES into positive or negative preferences.
- Do not infer a broad genre preference from an isolated liked or disliked film when the aggregate genre classification disagrees.
- Never infer a dislike from the absence of positive evidence or use a positively rated movie as evidence for a dislike.
- Every genre in REQUIRED NEGATIVE GENRES must be represented naturally in the negative portion of the summary. Do not suppress or silently omit one.
- Every genre in CORE POSITIVE GENRES must be represented naturally in the positive portion. SECONDARY POSITIVE GENRES may be summarized selectively when enumerating all of them would make the prose unnatural.
- Do not merely copy internal labels, counts, evidence statuses, or schema terminology into the profile. Write polished user-facing prose.
- When NEGATIVE EVIDENCE STATUS is NONE, do not fabricate a dislike. State naturally that the available history does not reveal a clear negative preference, or omit unsupported negative detail.
- When NEGATIVE EVIDENCE STATUS is WEAK, keep any negative inference narrow, cautious, and tentative. Weak examples must not become a categorical genre dislike.
- When NEGATIVE EVIDENCE STATUS is STRONG, express every required supported negative genre and only repeated or coherent additional negative patterns supported by NEGATIVE EVIDENCE.
- Movie titles in the evidence are private evidence only. Never copy them into the summary.
- Never include years or numeric ratings in the summary.

Preserve a rich, natural, editable preference-profile style where evidence supports it. Do not pad to an exact word count, fabricate content for length, or force exactly four sentences.
Begin with `Summary:`."""

PROMPT_SHA256 = hashlib.sha256(EVIDENCE_GATED_PROMPT.encode()).hexdigest()


GENRE_STATUS_RULES: dict[str, Any] = {
    "rating_bands": frozen_evidence.EVIDENCE_RULES["rating_bands"],
    "repeated_positive": frozen_evidence.EVIDENCE_RULES["repeated_positive"],
    "repeated_negative": frozen_evidence.EVIDENCE_RULES["repeated_negative"],
    "dominance": (
        "Reuse the audited two-to-one dominance and half-of-all-occurrences rules. "
        "POSITIVE requires repeated positive evidence, positive share >= 0.5, and "
        "positive occurrences >= 2 * negative occurrences. NEGATIVE is symmetric."
    ),
    "mixed": (
        "At least 3 positive and at least 3 negative occurrences, with neither side "
        "meeting the audited two-to-one dominance rule. Unlike v10, a clearly "
        "dominant side is not forced to MIXED merely because both counts reach 3."
    ),
    "insufficient": (
        "Every other observed genre: isolated evidence, inadequate concentration, "
        "or too little two-sided evidence to establish MIXED."
    ),
    "core_positive": (
        "Among POSITIVE genres, require coverage for genres whose positive occurrence "
        "count is at least half of the strongest POSITIVE genre count. This reuses the "
        "audited two-to-one materiality principle and leaves weaker supported genres "
        "secondary so prose need not enumerate every minor genre."
    ),
    "required_negative": "Every genre classified NEGATIVE is required in negative prose.",
}


def _classify_genre(counts: Mapping[str, Any]) -> dict[str, Any]:
    total = int(counts["occurrences"])
    positive = int(counts["positive_occurrences"])
    negative = int(counts["negative_occurrences"])
    very_negative = int(counts["very_negative_occurrences"])
    repeated_positive = positive >= 3
    repeated_negative = negative >= 3 or very_negative >= 2
    positive_share = positive / total
    negative_share = negative / total
    positive_supported = (
        repeated_positive and positive_share >= 0.5 and positive >= 2 * negative
    )
    negative_supported = (
        repeated_negative and negative_share >= 0.5 and negative >= 2 * positive
    )
    if positive_supported:
        status = "POSITIVE"
        reason = "repeated_positive_with_two_to_one_dominance_and_half_share"
    elif negative_supported:
        status = "NEGATIVE"
        reason = "repeated_negative_with_two_to_one_dominance_and_half_share"
    elif positive >= 3 and negative >= 3:
        status = "MIXED"
        reason = "substantial_two_sided_evidence_without_two_to_one_dominance"
    else:
        status = "INSUFFICIENT"
        if positive and negative:
            reason = "two_sided_evidence_but_too_little_or_not_concentrated"
        elif positive:
            reason = "positive_evidence_below_support_or_concentration_rule"
        elif negative:
            reason = "negative_evidence_below_support_or_concentration_rule"
        else:
            reason = "no_polarized_rating_evidence"
    return {
        **counts,
        "positive_strength": positive,
        "negative_strength": negative + very_negative,
        "positive_share": positive_share,
        "negative_share": negative_share,
        "positive_to_negative_ratio": None if negative == 0 else positive / negative,
        "negative_to_positive_ratio": None if positive == 0 else negative / positive,
        "repeated_positive": repeated_positive,
        "repeated_negative": repeated_negative,
        "status": status,
        "classification_reason": reason,
    }


def build_separated_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    required = ("movie_ids", "titles", "ratings", "genres")
    if any(key not in record for key in required):
        raise RuntimeError(f"Record lacks required history fields: {required}")
    if len({len(record[key]) for key in required}) != 1:
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
                v10._example(movie_id, title, genres, polarity="POSITIVE")
            )
        elif value <= 2.5:
            negative_examples.append(
                v10._example(movie_id, title, genres, polarity="NEGATIVE")
            )
        else:
            neutral_count += 1

    per_genre = {
        genre: _classify_genre(values)
        for genre, values in sorted(frozen["genre_statistics"].items())
    }
    by_status = {
        status: sorted(
            genre for genre, values in per_genre.items() if values["status"] == status
        )
        for status in ("POSITIVE", "NEGATIVE", "MIXED", "INSUFFICIENT")
    }
    positive_genres = by_status["POSITIVE"]
    negative_genres = by_status["NEGATIVE"]
    if not negative_examples:
        negative_status = "NONE"
    elif negative_genres:
        negative_status = "STRONG"
    else:
        negative_status = "WEAK"

    if positive_genres:
        strongest_positive = max(
            per_genre[genre]["positive_occurrences"] for genre in positive_genres
        )
        core_cutoff = math.ceil(strongest_positive / 2)
        core_positive = sorted(
            genre
            for genre in positive_genres
            if per_genre[genre]["positive_occurrences"] >= core_cutoff
        )
    else:
        strongest_positive = 0
        core_cutoff = 0
        core_positive = []
    secondary_positive = sorted(set(positive_genres) - set(core_positive))

    inference = {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "positive_evidence": positive_examples,
        "negative_evidence": negative_examples if negative_status != "NONE" else [],
        "negative_evidence_status": negative_status,
        "supported_positive_genres": positive_genres,
        "core_positive_genres": core_positive,
        "secondary_positive_genres": secondary_positive,
        "supported_negative_genres": negative_genres,
        "required_negative_genres": negative_genres,
        "mixed_genres": by_status["MIXED"],
        "insufficient_genres": by_status["INSUFFICIENT"],
    }
    result = {
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "user_id": record.get("user_id"),
        "history_hash": record.get("history_hash"),
        "positive_example_count": len(positive_examples),
        "negative_example_count": len(negative_examples),
        "neutral_excluded_count": neutral_count,
        "negative_evidence_status": negative_status,
        "genre_status_counts": {key: len(value) for key, value in by_status.items()},
        "genre_classifications": by_status,
        "genre_statistics": per_genre,
        "core_positive_rule": {
            "strongest_positive_occurrences": strongest_positive,
            "minimum_core_positive_occurrences": core_cutoff,
            "core_positive_genres": core_positive,
            "secondary_positive_genres": secondary_positive,
        },
        "required_negative_genres": negative_genres,
        "frozen_v4_evidence_hash": frozen["evidence_hash"],
        "rules": GENRE_STATUS_RULES,
        "inference_payload": inference,
    }
    result["evidence_hash"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def render_evidence_prompt(evidence: Mapping[str, Any]) -> str:
    payload = evidence["inference_payload"]
    status = payload["negative_evidence_status"]
    if status not in {"NONE", "WEAK", "STRONG"}:
        raise RuntimeError(f"Unknown negative evidence status: {status}")
    if status == "NONE" and payload["negative_evidence"]:
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
        + block(payload["positive_evidence"])
        + "\n\nNEGATIVE EVIDENCE:\n"
        + block(payload["negative_evidence"])
        + f"\n\nNEGATIVE EVIDENCE STATUS: {status}\n\n"
        + "SUPPORTED POSITIVE GENRES: "
        + json.dumps(payload["supported_positive_genres"])
        + "\nCORE POSITIVE GENRES (required coverage): "
        + json.dumps(payload["core_positive_genres"])
        + "\nSECONDARY POSITIVE GENRES (optional coverage): "
        + json.dumps(payload["secondary_positive_genres"])
        + "\nSUPPORTED NEGATIVE GENRES: "
        + json.dumps(payload["supported_negative_genres"])
        + "\nREQUIRED NEGATIVE GENRES (complete coverage required): "
        + json.dumps(payload["required_negative_genres"])
        + "\nMIXED GENRES: "
        + json.dumps(payload["mixed_genres"])
        + "\nINSUFFICIENT GENRES: "
        + json.dumps(payload["insufficient_genres"])
    )


def build_response_request(
    record: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    model: str = "gpt-5-mini-2025-08-07",
    max_output_tokens: int = 450,
    retry_attempt: int = 0,
) -> dict[str, Any]:
    identity = {
        "protocol": PROTOCOL_VERSION,
        "history_hash": record["history_hash"],
        "evidence_hash": evidence["evidence_hash"],
        "retry_attempt": retry_attempt,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "custom_id": f"eg11-user-{record['user_id']}-r{retry_attempt}-{digest[:16]}",
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


GENRE_ALIASES: dict[str, tuple[str, ...]] = {
    "Action": ("action",),
    "Adventure": ("adventure", "adventurous"),
    "Animation": ("animation", "animated"),
    "Children": ("children", "children's", "child-oriented", "kid-focused", "family-oriented", "family-friendly"),
    "Comedy": ("comedy", "comedies", "comedic", "humor", "humour", "slapstick", "rom-com"),
    "Crime": ("crime", "criminal",),
    "Documentary": ("documentary", "documentaries"),
    "Drama": ("drama", "dramas", "dramatic", "melodrama"),
    "Fantasy": ("fantasy", "fantastical"),
    "Film-Noir": ("film-noir", "film noir", "noir"),
    "Horror": ("horror",),
    "IMAX": ("imax", "large-format"),
    "Musical": ("musical", "musicals"),
    "Mystery": ("mystery", "mysteries"),
    "Romance": ("romance", "romantic", "rom-com"),
    "Sci-Fi": ("sci-fi", "science fiction", "science-fiction"),
    "Thriller": ("thriller", "thrillers"),
    "War": ("war", "wartime", "military"),
    "Western": ("western", "westerns"),
}

# These deliberately exclude adjective-like theme words such as ``wartime``,
# ``dramatic``, ``romantic``, and ``adventurous``.  The prompt permits grounded
# plot/theme prose even when the corresponding broad MovieLens genre is not a
# supported categorical preference.  Using the broader coverage aliases for
# polarity validation caused those harmless descriptions to be misclassified.
GENRE_CLAIM_ALIASES: dict[str, tuple[str, ...]] = {
    "Action": ("action", "action films", "action movies", "action fare"),
    "Adventure": ("adventure", "adventure films", "adventure movies", "adventure fare"),
    "Animation": ("animation", "animated films", "animated movies"),
    "Children": ("children's films", "children's fare", "children's movies", "child-oriented films"),
    "Comedy": ("comedy", "comedies", "comedy films", "comedic films", "rom-coms"),
    "Crime": ("crime", "crime films", "crime movies", "crime fare"),
    "Documentary": ("documentary", "documentaries"),
    "Drama": ("drama", "dramas", "drama films", "dramatic films"),
    "Fantasy": ("fantasy", "fantasy films", "fantasy movies"),
    "Film-Noir": ("film-noir", "film noir", "noir films"),
    "Horror": ("horror", "horror films", "horror movies"),
    "IMAX": ("imax", "imax films"),
    "Musical": ("musical", "musicals", "musical films"),
    "Mystery": ("mystery", "mysteries", "mystery films"),
    "Romance": ("romance", "romance films", "romantic films", "romantic comedies", "romantic dramas"),
    "Sci-Fi": ("sci-fi", "science fiction", "science-fiction"),
    "Thriller": ("thriller", "thrillers", "thriller films"),
    "War": ("war films", "war movies", "war genre"),
    "Western": ("western", "westerns", "western films"),
}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
CLAUSE_SPLIT = re.compile(
    r"\s*(?:;|\bbut\b|\bhowever\b|\bwhereas\b|\byet\b|\brather than\b)\s*",
    re.I,
)
POSITIVE_MARKER = re.compile(
    r"\b(?:enjoys?|likes?|favors?|favours?|prefers?|appreciates?|drawn to|"
    r"gravitates? toward|responds? well to|positive (?:affinity|preference|interest)|"
    r"shows? (?:a )?(?:clear )?(?:preference|receptivity|taste)|has (?:a )?(?:clear )?taste|"
    r"taste for|fondness for|interest in|openness to|appeals? to|is welcome|are welcome|"
    r"is liked|are liked|is appealing|are appealing|consistent positive)\b",
    re.I,
)
NEGATIVE_MARKER = re.compile(
    r"\b(?:dislikes?|does not enjoy|do not enjoy|doesn't enjoy|avoids?|rejects?|aversion|averse|less (?:fond|interested|enthusiastic|receptive)|not (?:enjoy|favor|favour|prefer)|negative (?:evidence|pattern|preference)|tends? not to)\b",
    re.I,
)
ABSTENTION_MARKER = re.compile(
    r"\b(?:does not reveal|do not reveal|does not assert|do not assert|"
    r"no (?:clear|supported|reliable|strong|categorical|consistent|specific|firm|broad) "
    r"(?:negative|dislike|evidence .{0,80} dislikes?)|no negative evidence|no dislikes?|"
    r"nothing .{0,40} dislikes?|without .{0,50} dislikes?|"
    r"does not .{0,60}(?:indicate|establish).{0,40}(?:dislikes?|negative preference)|"
    r"no .{0,60}(?:negative preferences?|dislikes?)|"
    r"insufficient .{0,50}(?:claim|infer).{0,40}dislikes?|"
    r"cannot be inferred|avoid inferring dislikes?|"
    r"remain(?:s)? (?:uncertain|inconclusive))\b",
    re.I,
)
MIXED_QUALIFIER = re.compile(
    r"\b(?:mixed|selective|context-dependent|context dependent|varies|varying|depends?|not categorical|rather than (?:a )?broad|when paired|when combined|some .{0,50} (?:but|while))\b",
    re.I,
)
CAUTIOUS_QUALIFIER = re.compile(
    r"\b(?:may|might|seems?|suggests?|tentative|cautious|occasionally|specific|certain|isolated|limited|weak|possible|perhaps|selective|context-dependent|context dependent|when paired|when combined)\b",
    re.I,
)


def _mentioned_genres(text: str) -> set[str]:
    lowered = text.lower().replace("‑", "-").replace("–", "-").replace("—", "-")
    found: set[str] = set()
    for genre, aliases in GENRE_ALIASES.items():
        if any(re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lowered) for alias in aliases):
            found.add(genre)
    return found


def _categorical_genre_claim(sentence: str, genre: str, *, positive: bool) -> bool:
    """Detect an explicit genre-polarity construction, not a distant mention."""

    normalized = sentence.lower().replace("‑", "-").replace("–", "-").replace("—", "-")
    aliases = GENRE_CLAIM_ALIASES.get(genre, (genre.lower(),))
    if positive:
        before = (
            r"(?:enjoys?|likes?|favors?|favours?|prefers?|appreciates?|drawn to|"
            r"positive (?:affinity|preference|interest)(?: (?:for|in))?|"
            r"shows? (?:a )?(?:clear )?(?:preference|receptivity)(?: (?:for|to))?)"
        )
        after = r"(?:is|are)?\s*(?:favored|favoured|liked|preferred|appealing)"
    else:
        before = (
            r"(?:dislikes?|does not enjoy|do not enjoy|doesn't enjoy|avoids?|rejects?|"
            r"aversion (?:to|for)|averse to|not (?:enjoy|favor|favour|prefer)|"
            r"negative (?:evidence|pattern|preference)(?: (?:for|against))?)"
        )
        after = r"(?:is|are)?\s*(?:disliked|avoided|rejected|unappealing)"
    return any(
        re.search(rf"\b{before}\b[^.;!?]{{0,120}}(?<![a-z]){re.escape(alias)}(?![a-z])", normalized)
        or re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])[^.;!?]{{0,50}}\b{after}\b", normalized)
        for alias in aliases
    )


@dataclass(frozen=True)
class ContractValidation:
    pass_contract: bool
    represented_positive_genres: tuple[str, ...]
    represented_negative_genres: tuple[str, ...]
    missing_core_positive_genres: tuple[str, ...]
    missing_required_negative_genres: tuple[str, ...]
    categorical_positive_on_negative_genres: tuple[str, ...]
    categorical_positive_on_mixed_genres: tuple[str, ...]
    categorical_positive_on_insufficient_genres: tuple[str, ...]
    categorical_negative_on_positive_genres: tuple[str, ...]
    categorical_negative_on_mixed_genres: tuple[str, ...]
    categorical_negative_on_insufficient_genres: tuple[str, ...]
    none_fabricated_dislike: bool
    weak_broad_unsupported_genres: tuple[str, ...]
    internal_contract_label_leak: bool


def validate_summary_contract(summary: str, evidence: Mapping[str, Any]) -> ContractValidation:
    payload = evidence["inference_payload"]
    positive_genres = set(payload["supported_positive_genres"])
    negative_genres = set(payload["supported_negative_genres"])
    mixed_genres = set(payload["mixed_genres"])
    insufficient_genres = set(payload["insufficient_genres"])
    core_positive = set(payload["core_positive_genres"])
    required_negative = set(payload["required_negative_genres"])
    represented_positive: set[str] = set()
    represented_negative: set[str] = set()
    positive_on_negative: set[str] = set()
    positive_on_mixed: set[str] = set()
    positive_on_insufficient: set[str] = set()
    negative_on_positive: set[str] = set()
    negative_on_mixed: set[str] = set()
    negative_on_insufficient: set[str] = set()
    weak_broad: set[str] = set()
    fabricated_none = False
    # Natural phrases such as "negative evidence is weak" are acceptable
    # evidence-aware prose, not leaked schema labels.  Fail only literal
    # machine-facing headings/status syntax or counts.
    internal_label_leak = bool(
        re.search(
            r"(?:SUPPORTED POSITIVE GENRES:|SUPPORTED NEGATIVE GENRES:|"
            r"REQUIRED NEGATIVE GENRES|CORE POSITIVE GENRES|MIXED GENRES:|"
            r"INSUFFICIENT GENRES:|NEGATIVE EVIDENCE STATUS:|"
            r"\bstatus is (?:NONE|WEAK|STRONG)\b|\bpositive occurrences?\s*[:=]\s*\d+|"
            r"\bnegative occurrences?\s*[:=]\s*\d+)",
            summary,
        )
    )

    for sentence in SENTENCE_SPLIT.split(summary.strip()):
        for clause in filter(None, CLAUSE_SPLIT.split(sentence)):
            mentioned = _mentioned_genres(clause)
            abstention = bool(ABSTENTION_MARKER.search(clause))
            has_negative = bool(NEGATIVE_MARKER.search(clause)) and not abstention
            has_positive = bool(POSITIVE_MARKER.search(clause)) and not has_negative
            qualified_mixed = bool(MIXED_QUALIFIER.search(clause))
            cautious = bool(CAUTIOUS_QUALIFIER.search(clause))
            if has_positive:
                represented_positive.update(mentioned & positive_genres)
                if not cautious:
                    positive_on_negative.update(
                        genre for genre in mentioned & negative_genres
                        if _categorical_genre_claim(clause, genre, positive=True)
                    )
                if not (qualified_mixed or cautious):
                    positive_on_mixed.update(
                        genre for genre in mentioned & mixed_genres
                        if _categorical_genre_claim(clause, genre, positive=True)
                    )
                if not cautious:
                    positive_on_insufficient.update(
                        genre for genre in mentioned & insufficient_genres
                        if _categorical_genre_claim(clause, genre, positive=True)
                    )
            if has_negative:
                represented_negative.update(mentioned & negative_genres)
                if not cautious:
                    negative_on_positive.update(
                        genre for genre in mentioned & positive_genres
                        if _categorical_genre_claim(clause, genre, positive=False)
                    )
                if not (qualified_mixed or cautious):
                    negative_on_mixed.update(
                        genre for genre in mentioned & mixed_genres
                        if _categorical_genre_claim(clause, genre, positive=False)
                    )
                if not cautious:
                    negative_on_insufficient.update(
                        genre for genre in mentioned & insufficient_genres
                        if _categorical_genre_claim(clause, genre, positive=False)
                    )
                if payload["negative_evidence_status"] == "NONE":
                    fabricated_none = True
                if payload["negative_evidence_status"] == "WEAK" and not cautious:
                    weak_broad.update(
                        genre
                        for genre in mentioned - negative_genres
                        if _categorical_genre_claim(clause, genre, positive=False)
                    )

    missing_positive = core_positive - represented_positive
    missing_negative = required_negative - represented_negative
    pass_contract = not any(
        (
            missing_positive,
            missing_negative,
            positive_on_negative,
            positive_on_mixed,
            positive_on_insufficient,
            negative_on_positive,
            negative_on_mixed,
            negative_on_insufficient,
            weak_broad,
        )
    ) and not fabricated_none and not internal_label_leak
    return ContractValidation(
        pass_contract=pass_contract,
        represented_positive_genres=tuple(sorted(represented_positive)),
        represented_negative_genres=tuple(sorted(represented_negative)),
        missing_core_positive_genres=tuple(sorted(missing_positive)),
        missing_required_negative_genres=tuple(sorted(missing_negative)),
        categorical_positive_on_negative_genres=tuple(sorted(positive_on_negative)),
        categorical_positive_on_mixed_genres=tuple(sorted(positive_on_mixed)),
        categorical_positive_on_insufficient_genres=tuple(sorted(positive_on_insufficient)),
        categorical_negative_on_positive_genres=tuple(sorted(negative_on_positive)),
        categorical_negative_on_mixed_genres=tuple(sorted(negative_on_mixed)),
        categorical_negative_on_insufficient_genres=tuple(sorted(negative_on_insufficient)),
        none_fabricated_dislike=fabricated_none,
        weak_broad_unsupported_genres=tuple(sorted(weak_broad)),
        internal_contract_label_leak=internal_label_leak,
    )


privacy_and_format_cleanup = v10.privacy_and_format_cleanup


def prompt_delta() -> dict[str, Any]:
    return {
        "classification": "evidence-gated adaptation of Emiliano; not an exact reproduction",
        "exact_reproduction_claimed": False,
        "source_prompt": ORIGINAL_EMILIANO_PROMPT,
        "source_prompt_sha256": hashlib.sha256(ORIGINAL_EMILIANO_PROMPT.encode()).hexdigest(),
        "v10_prompt_sha256": v10.PROMPT_SHA256,
        "adapted_prompt": EVIDENCE_GATED_PROMPT,
        "adapted_prompt_sha256": PROMPT_SHA256,
        "minimal_delta_from_v10": [
            "exclusive POSITIVE/NEGATIVE/MIXED/INSUFFICIENT genre statuses",
            "categorical claims licensed only by matching deterministic status",
            "required coverage for every supported negative genre",
            "required coverage for deterministic core positive genres",
            "internal evidence labels must not leak into user-facing prose",
        ],
        "unchanged": [
            "Emiliano-derived opening, tone, four-part semantic organization, and rich profile style",
            "separated positive and negative title/genre evidence",
            "NONE/WEAK/STRONG negative gate",
            "privacy restrictions and no TMDB semantic metadata",
            "no semantic post-generation repair",
        ],
    }
