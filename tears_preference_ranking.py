"""Conservative explicit-genre alignment for online TEARS candidate ranking."""
from __future__ import annotations

import re
import torch
from tears_training.evidence_gated_summary_harness_v11 import _mentioned_genres

POLICY_ID = "tears-compound-genre-alignment-v3"
# Do not interpret the mandatory unknown-preference sentences as preferences.
ABSTENTION = re.compile(
    r"\b(?:no (?:strong |clear |supported |firm )?(?:preference|dislike|evidence)|"
    r"not (?:strongly )?supported|does not support|unspecified|unknown|"
    r"insufficient|cannot (?:infer|establish)|mixed|selective|context-dependent)\b", re.I)
CLAUSES = re.compile(r"[.!?;]|\b(?:but|however|whereas|yet|rather than)\b", re.I)
DIRECTION = re.compile(
    r"(?P<negative>\b(?:does not|do not|doesn[’']t|don[’']t)\s+(?:like|enjoy|prefer)|"
    r"\b(?:dislikes?|hates?|avoids?|rejects?|aversion to|averse to|less interested in|responds? negatively to|and not))"
    r"|(?P<positive>\b(?:likes?|loves?|enjoys?|prefers?|favors?|favours?|appreciates?|"
    r"preference for|interest in|interested in|responds? (?:positively )?to|open to|receptive to|fond of|drawn to))", re.I)

# Preserve explicit combinations rather than treating each modifier as an
# interchangeable taste. These are language patterns, never movie-ID rules.
COMPOUNDS = (
    (re.compile(r"\b(?:animation|animated)\b[^.;!?]{0,70}\b(?:family[ -](?:friendly|oriented)|children|child[ -](?:friendly|oriented))\b|"
                r"\b(?:family[ -](?:friendly|oriented)|children|child[ -](?:friendly|oriented))\b[^.;!?]{0,70}\b(?:animation|animated)\b", re.I),
     frozenset(("Animation", "Children"))),
    (re.compile(r"\bromantic comed(?:y|ies)\b", re.I), frozenset(("Romance", "Comedy"))),
)


def mentioned_genres(text: str) -> set[str]:
    # Online-only inflection handling; leave the frozen training validator alone.
    normalized = re.sub(r"\badventures\b", "adventure", text, flags=re.I)
    return _mentioned_genres(normalized)


def explicit_genre_preferences(summary: str) -> dict:
    positive, negative = set(), set()
    positive_segments = []
    for clause in CLAUSES.split(summary):
        if ABSTENTION.search(clause):
            continue
        anchors = list(DIRECTION.finditer(clause))
        for index, anchor in enumerate(anchors):
            end = anchors[index + 1].start() if index + 1 < len(anchors) else len(clause)
            segment = clause[anchor.end():end]
            # "not interested in horror" is not a positive interest claim.
            if re.search(r"\b(?:not|no longer)\s*$", clause[:anchor.start()], re.I):
                continue
            target = negative if anchor.lastgroup == "negative" else positive
            target.update(mentioned_genres(segment))
            if anchor.lastgroup == "positive":
                positive_segments.append(segment)
    conflicting = positive & negative
    combinations = set()
    for segment in positive_segments:
        scoped_genres = mentioned_genres(segment)
        # An independently stated taste outside this scope is an alternative,
        # not a requirement to combine all of a person's interests in one film.
        if any(mentioned_genres(other) - scoped_genres for other in positive_segments):
            continue
        for pattern, genres in COMPOUNDS:
            match = pattern.search(segment)
            if (genres <= positive - negative and match
                    and not re.search(r"\b(?:or|also|separately)\b", match.group(), re.I)):
                combinations.add(tuple(sorted(genres)))
    return {"preferred_genres": sorted(positive - conflicting),
            "avoided_genres": sorted(negative - conflicting),
            "conflicting_genres": sorted(conflicting),
            "preferred_genre_combinations": [list(group) for group in sorted(combinations)]}


def align_scores(logits: torch.Tensor, genre_membership: dict[str, torch.Tensor], preferences):
    """Lexicographic preference groups, preserving model order within each group."""
    preferred = torch.zeros_like(logits, dtype=torch.bool)
    avoided = torch.zeros_like(logits, dtype=torch.bool)
    for genre in preferences["preferred_genres"]:
        if genre in genre_membership:
            preferred |= genre_membership[genre].to(logits.device)
    for genre in preferences["avoided_genres"]:
        if genre in genre_membership:
            avoided |= genre_membership[genre].to(logits.device)
    compound = torch.zeros_like(logits, dtype=torch.bool)
    for combination in preferences.get("preferred_genre_combinations", []):
        if not combination or any(genre not in genre_membership for genre in combination):
            continue
        matches = torch.ones_like(logits, dtype=torch.bool)
        for genre in combination:
            matches &= genre_membership[genre].to(logits.device)
        compound |= matches
    finite = logits[torch.isfinite(logits)]
    if not finite.numel():
        return logits
    # A whole score range makes explicit alignment take precedence without
    # tuning a coefficient on one participant's requested titles.
    span = (finite.max() - finite.min()).clamp_min(1) + 1
    # Compound matches precede partial matches; every explicit dislike still
    # takes precedence. Outside an explicit compound, retain the previous tiers.
    priority = preferred.to(logits.dtype) + compound.to(logits.dtype) - 3 * avoided.to(logits.dtype)
    return logits + priority * span
