"""Generic deterministic post-processing for the promoted V2 protocol.

The functions in this module are deliberately user-agnostic. Repair decisions
receive only summary text and deterministic rating/genre evidence. They do not
accept user IDs, calibration labels, manual adjudications, TMDB metadata, or any
other per-user exception source.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


PROTOCOL_VERSION = "tears-v2-generic-production-repair-v5"
NEUTRAL_ABSTENTION = (
    "The available history does not indicate a strong negative preference in this area."
)
SUPPORTED_NEGATIVE_TEMPLATE = (
    "The available history supports a negative preference for {genres}."
)

GENRE_PHRASES: dict[str, str] = {
    "Action": "action films",
    "Adventure": "adventure films",
    "Animation": "animated films",
    "Children": "children's films",
    "Comedy": "comedies",
    "Crime": "crime films",
    "Documentary": "documentaries",
    "Drama": "dramas",
    "Fantasy": "fantasy films",
    "Film-Noir": "film noir",
    "Horror": "horror films",
    "IMAX": "IMAX-format films",
    "Musical": "musicals",
    "Mystery": "mysteries",
    "Romance": "romances",
    "Sci-Fi": "science-fiction films",
    "Thriller": "thrillers",
    "War": "war films",
    "Western": "westerns",
}

GENRE_ALIASES: dict[str, tuple[str, ...]] = {
    "Action": ("action",),
    "Adventure": ("adventure", "adventures"),
    "Animation": ("animation", "animated"),
    "Children": ("children", "children's", "family films", "family-oriented films"),
    "Comedy": ("comedy", "comedies", "comic"),
    "Crime": ("crime",),
    "Documentary": ("documentary", "documentaries"),
    "Drama": ("drama", "dramas"),
    "Fantasy": ("fantasy",),
    "Film-Noir": ("film noir", "film-noir", "noir"),
    "Horror": ("horror",),
    "IMAX": ("imax", "imax-format"),
    "Musical": ("musical", "musicals"),
    "Mystery": ("mystery", "mysteries"),
    "Romance": ("romance", "romances", "romantic films"),
    "Sci-Fi": ("sci-fi", "science fiction", "science-fiction"),
    "Thriller": ("thriller", "thrillers"),
    "War": ("war films", "war movies", "war genre"),
    "Western": ("western", "westerns"),
}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
ABSTENTION_PATTERN = re.compile(
    r"(?:\bno\b|\bdoes not (?:show|indicate|provide)\b).{0,65}\bstrong\b"
    r".{0,45}\bnegative\b|\bnot strongly supported as a negative preference\b|"
    r"\bnegative preference\b.{0,100}\bnot strongly supported\b",
    re.IGNORECASE,
)
NEGATIVE_CLAIM_PATTERN = re.compile(
    r"\b(?:does not enjoy|do not enjoy|doesn't enjoy|disliked?|dislikes?|less fond of|"
    r"less enthusiastic about|less interested in|less engaged by|lukewarm toward|"
    r"less enjoyment|less enjoy|enjoys? .{0,60} less|less favorably|less appealing|"
    r"rates?\b.{0,180}\b(?:lower|low|poorly|less favorably|less highly)|"
    r"rated\b.{0,180}\b(?:lower|low|poorly|less favorably|less highly)|lower ratings?|"
    r"negative reaction|responds? poorly|averse to|avoids?|not drawn to|"
    r"does not favou?r|do not favou?r|not a fan of|little interest in|"
    r"least interested in|tends? not to enjoy|tends? not to favou?r|"
    r"tends? to favou?r .{0,60} less|tends? to rate down|preference against)\b",
    re.IGNORECASE,
)
USER_NEGATIVE_FALLBACK_PATTERN = re.compile(
    r"\b(?:the )?user(?:'s)?\b.{0,240}\b(?:disliked?|dislikes?|lower|"
    r"less (?:highly|favorably|enthusiastic|interested|engaged|enjoyed)|"
    r"not (?:enjoy|favor|favour|prefer)|poorly|aversion)\b",
    re.IGNORECASE,
)
OTHER_VIEWER_PREFIX = re.compile(
    r"^(?:Summary:\s*)?(?:other|some) (?:viewers|users|audiences)\b",
    re.IGNORECASE,
)
MALFORMED_COLON = re.compile(
    r"The user does not enjoy:\s*(no strong negative(?: genre)? preference is "
    r"supported by the available history\.)",
    re.IGNORECASE,
)
GRAMMAR_ABSTENTION = re.compile(
    r"The user does not enjoy (?P<subject>.+?) is not strongly supported as a "
    r"negative preference by the available history\.",
    re.IGNORECASE,
)
TOKEN_CORRUPTIONS: tuple[tuple[str, str], ...] = (
    ("broadly thatmost ", ""),
)
NON_SEMANTIC_SENTENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:^|(?<=[.!?])\s+)If the user's rating history does not provide "
        r"sufficient evidence for a negative preference, do not infer or invent "
        r"one\. Instead, state that no strong negative preference is supported "
        r"by the available history\.(?:\s+|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|(?<=[.!?])\s+)\{Specific details of [^{}]+\}\.(?:\s+|$)",
        re.IGNORECASE,
    ),
)
PLACEHOLDER_PATTERN = re.compile(
    r"\{[^{}]+\}|supported_(?:positive|negative)_genres|negative_status|"
    r"weak_or_ambiguous_negative_status|\b(?:N/?A|TBD|TODO)\b",
    re.IGNORECASE,
)
NON_GENRE_NEGATIVE_ATTRIBUTE_PATTERN = re.compile(
    r"\b(?:plot|plots|plotline|plotlines|story|stories|narrative|narratives|"
    r"theme|themes|thematic|style|styles|stylistic|tone|tones|tonal|pacing|"
    r"pace|paced|character|characters|content|ending|endings|arc|arcs|"
    r"sentimental|formulaic|conventional|derivative|gimmick|gimmicks|"
    r"franchise|franchises|installment|installments|sequel|sequels|"
    r"spectacle|spectacles|set-piece|set-pieces|twist|twists|depth|"
    r"originality|worldbuilding|dialogue|violence|violent|gore|gory|"
    r"humor|humour|emotional|emotionally|slow-burn|high-octane|"
    r"low-tension|shallow|pulpy|quirky|subversive)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProtocolResult:
    raw_summary: str
    final_summary: str
    supported_negative_genres: tuple[str, ...]
    operations: tuple[dict[str, Any], ...]
    positive_sentences_preserved: bool
    supported_negative_sentences_preserved: tuple[str, ...]
    inappropriate_abstention: bool
    supported_negative_evidence_omitted: bool
    formatting_issues: tuple[str, ...]
    literal_placeholder: bool

    @property
    def changed(self) -> bool:
        return self.raw_summary != self.final_summary


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _is_abstention(sentence: str) -> bool:
    return bool(ABSTENTION_PATTERN.search(_normal(sentence)))


def _is_user_negative_claim(sentence: str) -> bool:
    stripped = sentence.strip()
    if OTHER_VIEWER_PREFIX.match(stripped) and not re.search(
        r"\b(?:the )?user\b", stripped[20:], re.IGNORECASE
    ):
        return False
    normalized = _normal(stripped)
    has_claim_marker = bool(
        NEGATIVE_CLAIM_PATTERN.search(normalized)
        or USER_NEGATIVE_FALLBACK_PATTERN.search(normalized)
    )
    return has_claim_marker


def _mentioned_genres(sentence: str) -> set[str]:
    normalized = f" {_normal(sentence)} "
    result: set[str] = set()
    for genre, aliases in GENRE_ALIASES.items():
        if any(f" {_normal(alias)} " in normalized for alias in aliases):
            result.add(genre)
    return result


def _is_narrow_supported_negative_claim(
    sentence: str,
    supported_genres: tuple[str, ...],
) -> bool:
    """Accept only genre-only negative prose supported by deterministic evidence."""

    if not supported_genres or _is_abstention(sentence):
        return False
    mentioned = _mentioned_genres(sentence)
    if not mentioned or not mentioned.issubset(set(supported_genres)):
        return False
    if NON_GENRE_NEGATIVE_ATTRIBUTE_PATTERN.search(sentence):
        return False
    return True


def _format_genre_list(genres: tuple[str, ...]) -> str:
    phrases = tuple(GENRE_PHRASES[genre] for genre in genres)
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"


def _negative_slot(genres: tuple[str, ...]) -> str:
    if not genres:
        return NEUTRAL_ABSTENTION
    return SUPPORTED_NEGATIVE_TEMPLATE.format(genres=_format_genre_list(genres))


def _normalize_formatting(text: str) -> tuple[str, list[dict[str, Any]]]:
    value = text.strip()
    operations: list[dict[str, Any]] = []
    for pattern in NON_SEMANTIC_SENTENCE_PATTERNS:
        matches = list(pattern.finditer(value))
        for match in reversed(matches):
            original = match.group(0)
            value = (value[: match.start()] + " " + value[match.end() :]).strip()
            value = re.sub(r"\s{2,}", " ", value)
            operations.append(
                {
                    "type": "format_remove_literal_instruction_or_placeholder",
                    "original_span": original.strip(),
                    "replacement_span": "",
                    "semantic_change": False,
                }
            )
    for original, replacement in TOKEN_CORRUPTIONS:
        count = value.count(original)
        if count:
            value = value.replace(original, replacement)
            operations.append(
                {
                    "type": "format_token_corruption",
                    "original_span": original,
                    "replacement_span": replacement,
                    "occurrences": count,
                    "semantic_change": False,
                }
            )
    grammar_matches = list(GRAMMAR_ABSTENTION.finditer(value))
    for match in reversed(grammar_matches):
        original = match.group(0)
        replacement = (
            f"A negative preference for {match.group('subject')} is not strongly "
            "supported by the available history."
        )
        value = value[: match.start()] + replacement + value[match.end() :]
        operations.append(
            {
                "type": "format_ungrammatical_abstention",
                "original_span": original,
                "replacement_span": replacement,
                "semantic_change": False,
            }
        )
    colon_matches = list(MALFORMED_COLON.finditer(value))
    for match in reversed(colon_matches):
        original = match.group(0)
        captured = match.group(1)
        replacement = captured[0].upper() + captured[1:]
        value = value[: match.start()] + replacement + value[match.end() :]
        operations.append(
            {
                "type": "format_malformed_abstention_leadin",
                "original_span": original,
                "replacement_span": replacement,
                "semantic_change": False,
            }
        )
    if not value.startswith("Summary:"):
        value = "Summary: " + value
        operations.append(
            {
                "type": "format_add_summary_prefix",
                "original_span": "",
                "replacement_span": "Summary: ",
                "semantic_change": False,
            }
        )
    return value, operations


def formatting_issues(text: str) -> tuple[str, ...]:
    issues: list[str] = []
    if not text.startswith("Summary:"):
        issues.append("missing_summary_prefix")
    if MALFORMED_COLON.search(text):
        issues.append("malformed_abstention_leadin")
    if GRAMMAR_ABSTENTION.search(text):
        issues.append("ungrammatical_abstention")
    if any(original in text for original, _ in TOKEN_CORRUPTIONS):
        issues.append("known_token_corruption")
    if any(pattern.search(text) for pattern in NON_SEMANTIC_SENTENCE_PATTERNS):
        issues.append("literal_instruction_or_placeholder_sentence")
    return tuple(issues)


def repair_summary(
    summary: str,
    evidence: Mapping[str, Any],
) -> ProtocolResult:
    """Apply generic grounding, abstention, and formatting repairs.

    The function intentionally has no user identifier argument. The only
    semantic decision input is ``supported_negative_genres`` from deterministic
    rating/genre evidence.
    """

    raw_summary = summary
    formatted, format_operations = _normalize_formatting(summary)
    genres = tuple(sorted(str(value) for value in evidence["supported_negative_genres"]))
    unknown = sorted(set(genres) - set(GENRE_PHRASES))
    if unknown:
        raise RuntimeError(f"Unsupported MovieLens genre labels: {unknown}")

    sentences = SENTENCE_SPLIT.split(formatted)
    all_negative_indexes = {
        index for index, sentence in enumerate(sentences) if _is_user_negative_claim(sentence)
    }
    supported_negative_indexes = {
        index
        for index in all_negative_indexes
        if _is_narrow_supported_negative_claim(sentences[index], genres)
    }
    unsupported_negative_indexes = all_negative_indexes - supported_negative_indexes
    all_abstention_indexes = {
        index for index, sentence in enumerate(sentences) if _is_abstention(sentence)
    }
    eligible_abstention_indexes = all_abstention_indexes - unsupported_negative_indexes
    retained_abstention_indexes = (
        {min(eligible_abstention_indexes)}
        if not genres and eligible_abstention_indexes
        else set()
    )
    repair_abstention_indexes = all_abstention_indexes - retained_abstention_indexes
    target_indexes = sorted(unsupported_negative_indexes | repair_abstention_indexes)
    positive_sentences = [
        sentence
        for index, sentence in enumerate(sentences)
        if index not in target_indexes and index not in supported_negative_indexes
    ]
    semantic_operations: list[dict[str, Any]] = []
    slot = _negative_slot(genres)

    has_retained_negative_slot = bool(
        supported_negative_indexes or retained_abstention_indexes
    )
    if not target_indexes and has_retained_negative_slot:
        final_summary = formatted
    elif target_indexes:
        anchor = target_indexes[0]
        rebuilt: list[str] = []
        for index, sentence in enumerate(sentences):
            if index == anchor and not has_retained_negative_slot:
                rebuilt.append(slot)
                semantic_operations.append(
                    {
                        "type": (
                            "abstention_to_supported_negative"
                            if genres
                            and index in repair_abstention_indexes
                            and index not in unsupported_negative_indexes
                            else "grounding_narrow_or_neutralize"
                        ),
                        "original_span": sentence,
                        "replacement_span": slot,
                        "supported_negative_genres": list(genres),
                    }
                )
            elif index in target_indexes:
                semantic_operations.append(
                    {
                        "type": "remove_additional_negative_or_abstention_span",
                        "original_span": sentence,
                        "replacement_span": "",
                        "supported_negative_genres": list(genres),
                    }
                )
            else:
                rebuilt.append(sentence)
        final_summary = " ".join(rebuilt).strip()
    else:
        final_summary = (formatted.rstrip() + " " + slot).strip()
        semantic_operations.append(
            {
                "type": "insert_missing_negative_slot",
                "original_span": "",
                "replacement_span": slot,
                "supported_negative_genres": list(genres),
            }
        )

    # The Summary: prefix can be attached to a first sentence that is itself a
    # repair target. Re-run only the non-semantic normalizer after span repair.
    final_summary, trailing_format_operations = _normalize_formatting(final_summary)
    format_operations.extend(trailing_format_operations)
    final_issues = formatting_issues(final_summary)
    if final_issues:
        raise RuntimeError(f"Formatting normalization failed: {final_issues}")
    positive_preserved = all(sentence in final_summary for sentence in positive_sentences)
    if not positive_preserved:
        raise RuntimeError("A non-negative sentence changed during deterministic repair")
    preserved_negative_sentences = tuple(
        sentences[index] for index in sorted(supported_negative_indexes)
    )
    if not all(sentence in final_summary for sentence in preserved_negative_sentences):
        raise RuntimeError("A supported narrow negative sentence changed during repair")
    final_has_abstention = bool(genres and any(
        _is_abstention(sentence) for sentence in SENTENCE_SPLIT.split(final_summary)
    ))
    inappropriate = bool(genres and final_has_abstention)
    omitted = bool(genres and slot != _negative_slot(genres))
    operations = tuple(format_operations + semantic_operations)
    return ProtocolResult(
        raw_summary=raw_summary,
        final_summary=final_summary,
        supported_negative_genres=genres,
        operations=operations,
        positive_sentences_preserved=positive_preserved,
        supported_negative_sentences_preserved=preserved_negative_sentences,
        inappropriate_abstention=inappropriate,
        supported_negative_evidence_omitted=omitted,
        formatting_issues=final_issues,
        literal_placeholder=bool(PLACEHOLDER_PATTERN.search(final_summary)),
    )
