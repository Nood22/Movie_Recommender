"""Frozen final summary composition architecture.

The LLM owns positive preference prose only. Negative preference language,
privacy cleanup, consistency checks, and the ``Summary:`` prefix are owned by
generic deterministic code. The implementation accepts no user identifier,
manual label, TMDB metadata, or model/API client.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence

from . import production_summary_protocol as legacy


PROTOCOL_VERSION = "tears-final-positive-llm-deterministic-negative-20260816"

# This is a final architectural prompt, not a new calibration variant. It keeps
# the positive half of the Emiliano structure and removes every free-form
# negative slot. The prefix is deliberately absent because code renders it.
FINAL_POSITIVE_SYSTEM_PROMPT = """Task: You will now help me generate a highly detailed positive-preference summary based on the broad common elements of movies.
Do not comment on the year of production. Do not mention any specific movie titles or actors.
Do not quote or mention numeric ratings; use qualitative preference language.
Use the supplied history only as private evidence. Never copy a title from it into the output.
Describe only positive preferences: specific genres the user enjoys and specific plot or narrative elements the user seems to enjoy.
Do not generate negative preferences, dislikes, abstentions, negative plots, negative themes, negative styles, negative tones, or content about what other users may enjoy.
Keep the positive prose concise and natural. Do not add a Summary: prefix; it is rendered deterministically after validation."""

POSITIVE_ABSTENTION = (
    "The available history does not indicate a strong positive preference in this area."
)

YEAR_PATTERN = re.compile(r"\b(?:18|19|20)\d{2}\b")
NUMERIC_RATING_PATTERN = re.compile(
    r"\b(?:rated?|ratings?)(?:\s+(?:as|of))?[^.!?\d]{0,60}"
    r"[0-5](?:\.\d+)?(?:\s*/\s*5|\s*stars?)?\b",
    re.IGNORECASE,
)
POSITIVE_CUE_PATTERN = re.compile(
    r"\b(?:enjoys?|likes?|favo(?:u)?rs?|prefers?|appreciates?|"
    r"tends? to (?:enjoy|like|favo(?:u)?r|prefer)|seems? to enjoy|"
    r"gravitates? (?:toward|towards|to)|responds? well to|drawn to|"
    r"strong interest in|interest in|soft spot for|enthusiastic about)\b",
    re.IGNORECASE,
)
NON_POSITIVE_CONTEXT_PATTERN = re.compile(
    r"\b(?:rather than|instead of|as opposed to|not|without|avoids?|"
    r"dislikes?|less|negative(?:ly)?|averse to|preference against)\b",
    re.IGNORECASE,
)
TITLE_CONTEXT_PATTERN = re.compile(
    r"(?:\be\.g\.?\s*,?\s*|\bsuch as\s+|\bincluding\s+|\blike\s+)$",
    re.IGNORECASE,
)
PARENTHETICAL_PATTERN = re.compile(r"\([^()]*\)")
ARTICLE_SUFFIX_PATTERN = re.compile(
    r"^(?P<body>.+),\s*(?P<article>The|A|An|Les|Le|La)$", re.IGNORECASE
)


@dataclass(frozen=True)
class FinalCompositionResult:
    raw_summary: str
    extracted_positive_prose: str
    sanitized_positive_prose: str
    final_summary: str
    supported_negative_genres: tuple[str, ...]
    operations: tuple[dict[str, Any], ...]
    title_matches_before: tuple[str, ...]
    title_matches_after: tuple[str, ...]
    positive_negative_contradictions_before: tuple[str, ...]
    positive_negative_contradictions_after: tuple[str, ...]
    year_leaks_after: tuple[str, ...]
    numeric_rating_leaks_after: tuple[str, ...]
    formatting_issues: tuple[str, ...]


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _strip_summary_prefix(text: str) -> str:
    return re.sub(r"^\s*Summary\s*:\s*", "", text, flags=re.IGNORECASE).strip()


def _sentences(text: str) -> list[str]:
    value = text.strip()
    if not value:
        return []
    return [part.strip() for part in legacy.SENTENCE_SPLIT.split(value) if part.strip()]


def _format_genres(genres: Sequence[str]) -> str:
    unknown = sorted(set(genres) - set(legacy.GENRE_PHRASES))
    if unknown:
        raise RuntimeError(f"Unsupported MovieLens genre labels: {unknown}")
    return legacy._format_genre_list(tuple(genres))


def deterministic_negative_sentence(genres: Sequence[str]) -> str:
    ordered = tuple(sorted(str(value) for value in genres))
    if not ordered:
        return legacy.NEUTRAL_ABSTENTION
    return legacy.SUPPORTED_NEGATIVE_TEMPLATE.format(genres=_format_genres(ordered))


def deterministic_positive_fallback(genres: Sequence[str]) -> str:
    ordered = tuple(sorted(str(value) for value in genres))
    if not ordered:
        return POSITIVE_ABSTENTION
    return (
        "The available history supports a positive preference for "
        f"{_format_genres(ordered)}."
    )


def _extract_positive_prose(summary: str, evidence: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Discard every generated negative/abstention slot using the frozen repair parser."""

    repaired = legacy.repair_summary(summary, evidence)
    negative_slot = deterministic_negative_sentence(
        evidence.get("supported_negative_genres", ())
    )
    kept: list[str] = []
    removed: list[str] = []
    for sentence in _sentences(repaired.final_summary):
        clean = _strip_summary_prefix(sentence)
        if _normal(clean) == _normal(negative_slot):
            removed.append(sentence)
            continue
        if legacy._is_user_negative_claim(clean) or legacy._is_abstention(clean):
            removed.append(sentence)
            continue
        if legacy.OTHER_VIEWER_PREFIX.match(clean):
            removed.append(sentence)
            continue
        kept.append(clean)
    operations = [
        {
            "type": "discard_llm_negative_or_abstention_content",
            "original_span": span,
            "replacement_span": "",
        }
        for span in removed
    ]
    return " ".join(kept).strip(), operations


def _title_aliases(title: str) -> tuple[str, ...]:
    """Derive exact display aliases from one supplied MovieLens title."""

    value = re.sub(r"\s*\((?:18|19|20)\d{2}\)\s*$", "", title).strip()
    candidates: set[str] = {value}
    without_parentheses = re.sub(r"\s*\([^()]*\)", "", value).strip()
    if without_parentheses:
        candidates.add(without_parentheses)
    for group in re.findall(r"\(([^()]*)\)", value):
        group = re.sub(r"^(?:a\.k\.a\.|aka)\s+", "", group, flags=re.I).strip()
        if group:
            candidates.add(group)
    expanded: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip(" ,;:-")
        if not candidate:
            continue
        expanded.add(candidate)
        article = ARTICLE_SUFFIX_PATTERN.match(candidate)
        if article:
            body = article.group("body").strip()
            expanded.add(body)
            expanded.add(f"{article.group('article')} {body}")
        if ":" in candidate:
            expanded.add(candidate.split(":", 1)[0].strip())
        sequel_base = re.sub(
            r"\s+(?:Part\s+)?(?:\d+|[IVX]+)$", "", candidate, flags=re.I
        ).strip()
        if sequel_base != candidate:
            expanded.add(sequel_base)
    cleaned = {
        item.strip(" ,;:-")
        for item in expanded
        if len(_normal(item)) >= 4
    }
    return tuple(sorted(cleaned, key=lambda item: (-len(item), item.casefold())))


def _alias_pattern(alias: str) -> re.Pattern[str]:
    pieces = re.split(r"([’'])", alias)
    escaped = "".join("[’']" if part in {"’", "'"} else re.escape(part) for part in pieces)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\-", r"[-‐‑‒–— ]")
    return re.compile(rf"(?<![\w]){escaped}(?![\w])", re.IGNORECASE)


def title_matches(text: str, history_titles: Iterable[str]) -> tuple[dict[str, Any], ...]:
    matches: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    aliases: list[tuple[str, str]] = []
    for supplied in history_titles:
        aliases.extend((alias, supplied) for alias in _title_aliases(str(supplied)))
    aliases.sort(key=lambda pair: -len(pair[0]))
    for alias, supplied in aliases:
        token_count = len(_normal(alias).split())
        for match in _alias_pattern(alias).finditer(text):
            span = (match.start(), match.end())
            if any(start <= span[0] and span[1] <= end for start, end in seen):
                continue
            if token_count == 1:
                prefix = text[max(0, match.start() - 24) : match.start()]
                in_parentheses = (
                    text.rfind("(", 0, match.start())
                    > text.rfind(")", 0, match.start())
                )
                exact_case = match.group(0) == alias
                direct_cue = bool(TITLE_CONTEXT_PATTERN.search(prefix))
                next_text = text[match.end() : match.end() + 8]
                parenthetical_example = in_parentheses and (
                    "e.g." in prefix.lower() or next_text.lstrip().startswith(",")
                )
                style_suffix = bool(re.match(r"[-‐‑‒–—]style\b", next_text, re.I))
                if (
                    not exact_case
                    or style_suffix
                    or not (direct_cue or parenthetical_example)
                ):
                    continue
            seen.add(span)
            matches.append(
                {
                    "matched_text": match.group(0),
                    "alias": alias,
                    "supplied_title": supplied,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return tuple(sorted(matches, key=lambda item: (item["start"], -item["end"])))


def _cleanup_after_span_removal(text: str) -> str:
    value = re.sub(r"\s+", " ", text)
    value = re.sub(r"\s+([,.;:])", r"\1", value)
    value = re.sub(r",\s*,+", ",", value)
    value = re.sub(r"\(\s*\)", "", value)
    value = re.sub(r"\s+([—–-])\s*([,.;])", r"\2", value)
    value = re.sub(r"\s+([—–-])\s*$", "", value)
    value = re.sub(r"\s+,", ",", value)
    return value.strip()


def sanitize_privacy(
    positive_prose: str,
    history_titles: Iterable[str],
) -> tuple[str, list[dict[str, Any]], tuple[str, ...]]:
    """Remove only copied title/year/numeric-rating spans from positive prose."""

    titles = tuple(str(value) for value in history_titles)
    value = positive_prose
    operations: list[dict[str, Any]] = []
    before = title_matches(value, titles)

    # Parenthetical title examples are removed as one local unit, preserving the
    # surrounding preference claim.
    for match in reversed(list(PARENTHETICAL_PATTERN.finditer(value))):
        group = match.group(0)
        if title_matches(group, titles):
            value = value[: match.start()] + value[match.end() :]
            operations.append(
                {
                    "type": "privacy_remove_title_example_parenthetical",
                    "original_span": group,
                    "replacement_span": "",
                }
            )

    # Remove any remaining direct title reference. Expand over a local example
    # cue such as "like" so the sentence stays grammatical.
    remaining = list(title_matches(value, titles))
    for match in reversed(remaining):
        start, end = int(match["start"]), int(match["end"])
        prefix = value[max(0, start - 24) : start]
        cue = TITLE_CONTEXT_PATTERN.search(prefix)
        if cue:
            start = max(0, start - 24) + cue.start()
        original = value[start:end]
        value = value[:start] + value[end:]
        operations.append(
            {
                "type": "privacy_remove_direct_title_copy",
                "original_span": original,
                "replacement_span": "",
                "supplied_title": match["supplied_title"],
            }
        )

    for match in reversed(list(YEAR_PATTERN.finditer(value))):
        original = match.group(0)
        value = value[: match.start()] + value[match.end() :]
        operations.append(
            {
                "type": "privacy_remove_year",
                "original_span": original,
                "replacement_span": "",
            }
        )
    for match in reversed(list(NUMERIC_RATING_PATTERN.finditer(value))):
        original = match.group(0)
        value = value[: match.start()] + value[match.end() :]
        operations.append(
            {
                "type": "privacy_remove_numeric_rating",
                "original_span": original,
                "replacement_span": "",
            }
        )
    value = _cleanup_after_span_removal(value)
    after = tuple(match["matched_text"] for match in title_matches(value, titles))
    return value, operations, after


def _genre_alias_occurrences(text: str, genre: str) -> list[re.Match[str]]:
    matches: list[re.Match[str]] = []
    for alias in legacy.GENRE_ALIASES[genre]:
        matches.extend(_alias_pattern(alias).finditer(text))
    return sorted(matches, key=lambda item: item.start())


def _looks_like_broad_genre_use(sentence: str, genre: str, match: re.Match[str]) -> bool:
    if genre != "Action":
        return True
    local = sentence[max(0, match.start() - 28) : min(len(sentence), match.end() + 32)]
    if re.search(
        r"\baction(?:[- ](?:leaning|oriented|heavy))?\s+"
        r"(?:films?|movies?|genre|dramas?|thrillers?|comed(?:y|ies)|adventures?)\b",
        local,
        re.IGNORECASE,
    ):
        return True
    return bool(
        re.search(
            r"\b(?:crime|drama|sci[- ]?fi|thriller|adventure)\b.{0,18}\baction\b|"
            r"\baction\b.{0,18}\b(?:crime|drama|sci[- ]?fi|thriller|adventure)\b",
            local,
            re.IGNORECASE,
        )
    )


def positive_genre_claims(sentence: str) -> tuple[str, ...]:
    """Return broad genre mentions governed by genuine positive-preference cues."""

    cues = list(POSITIVE_CUE_PATTERN.finditer(sentence))
    if not cues:
        return ()
    claims: set[str] = set()
    for genre in legacy.GENRE_ALIASES:
        for match in _genre_alias_occurrences(sentence, genre):
            prior_cues = [cue for cue in cues if cue.start() < match.start()]
            if not prior_cues:
                continue
            cue = prior_cues[-1]
            between = sentence[cue.end() : match.start()]
            contrast = NON_POSITIVE_CONTEXT_PATTERN.search(between[-55:])
            if contrast:
                continue
            if not _looks_like_broad_genre_use(sentence, genre, match):
                continue
            claims.add(genre)
    return tuple(sorted(claims))


def resolve_positive_negative_contradictions(
    positive_prose: str,
    supported_negative_genres: Sequence[str],
) -> tuple[str, list[dict[str, Any]], tuple[str, ...], tuple[str, ...]]:
    """Remove only sentences with genuine broad positive/negative genre collisions."""

    negative = set(map(str, supported_negative_genres))
    kept: list[str] = []
    operations: list[dict[str, Any]] = []
    before: set[str] = set()
    for sentence in _sentences(positive_prose):
        collision = set(positive_genre_claims(sentence)) & negative
        if collision:
            before.update(collision)
            operations.append(
                {
                    "type": "consistency_remove_contradictory_positive_sentence",
                    "original_span": sentence,
                    "replacement_span": "",
                    "contradictory_genres": sorted(collision),
                }
            )
        else:
            kept.append(sentence)
    value = " ".join(kept).strip()
    after: set[str] = set()
    for sentence in _sentences(value):
        after.update(set(positive_genre_claims(sentence)) & negative)
    return value, operations, tuple(sorted(before)), tuple(sorted(after))


def final_formatting_issues(text: str) -> tuple[str, ...]:
    issues = list(legacy.formatting_issues(text))
    if text.count("Summary:") != 1 or not text.startswith("Summary: "):
        issues.append("summary_prefix_not_exactly_once")
    if re.search(r"\s{2,}", text):
        issues.append("repeated_whitespace")
    if not text.rstrip().endswith((".", "!", "?")):
        issues.append("missing_terminal_punctuation")
    return tuple(sorted(set(issues)))


def compose_final_summary(
    positive_candidate: str,
    evidence: Mapping[str, Any],
    history_titles: Sequence[str],
) -> FinalCompositionResult:
    """Compose one final summary without any free-form negative generation."""

    negative_genres = tuple(
        sorted(str(value) for value in evidence.get("supported_negative_genres", ()))
    )
    positive_genres = tuple(
        sorted(str(value) for value in evidence.get("supported_positive_genres", ()))
    )
    extracted, extraction_ops = _extract_positive_prose(positive_candidate, evidence)
    title_before_records = title_matches(extracted, history_titles)
    sanitized, privacy_ops, title_after = sanitize_privacy(extracted, history_titles)
    consistent, consistency_ops, contradictions_before, contradictions_after = (
        resolve_positive_negative_contradictions(sanitized, negative_genres)
    )
    fallback_ops: list[dict[str, Any]] = []
    if not consistent.strip():
        fallback = deterministic_positive_fallback(positive_genres)
        consistent = fallback
        fallback_ops.append(
            {
                "type": "insert_grounded_positive_fallback",
                "original_span": "",
                "replacement_span": fallback,
                "supported_positive_genres": list(positive_genres),
            }
        )
    consistent = _cleanup_after_span_removal(_strip_summary_prefix(consistent))
    negative_sentence = deterministic_negative_sentence(negative_genres)
    final_summary = f"Summary: {consistent} {negative_sentence}".strip()
    final_summary = _cleanup_after_span_removal(final_summary)
    title_after_final = tuple(
        match["matched_text"] for match in title_matches(final_summary, history_titles)
    )
    year_after = tuple(match.group(0) for match in YEAR_PATTERN.finditer(final_summary))
    rating_after = tuple(
        match.group(0) for match in NUMERIC_RATING_PATTERN.finditer(final_summary)
    )
    issues = final_formatting_issues(final_summary)
    if title_after_final or year_after or rating_after or contradictions_after or issues:
        raise RuntimeError(
            "Final composition validation failed: "
            f"titles={title_after_final}, years={year_after}, ratings={rating_after}, "
            f"contradictions={contradictions_after}, formatting={issues}"
        )
    return FinalCompositionResult(
        raw_summary=positive_candidate,
        extracted_positive_prose=extracted,
        sanitized_positive_prose=consistent,
        final_summary=final_summary,
        supported_negative_genres=negative_genres,
        operations=tuple(extraction_ops + privacy_ops + consistency_ops + fallback_ops),
        title_matches_before=tuple(
            match["matched_text"] for match in title_before_records
        ),
        title_matches_after=title_after_final,
        positive_negative_contradictions_before=contradictions_before,
        positive_negative_contradictions_after=contradictions_after,
        year_leaks_after=year_after,
        numeric_rating_leaks_after=rating_after,
        formatting_issues=issues,
    )
