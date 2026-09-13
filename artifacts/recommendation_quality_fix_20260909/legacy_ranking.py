"""Conservative explicit-genre alignment for online TEARS candidate ranking."""
from __future__ import annotations

import re
import torch
from tears_training.evidence_gated_summary_harness_v11 import _mentioned_genres

POLICY_ID = "tears-explicit-genre-alignment-v1"
# Do not interpret the mandatory unknown-preference sentences as preferences.
ABSTENTION = re.compile(
    r"\b(?:no (?:strong |clear |supported |firm )?(?:preference|dislike|evidence)|"
    r"not (?:strongly )?supported|does not support|unspecified|unknown|"
    r"insufficient|cannot (?:infer|establish)|mixed|selective|context-dependent)\b", re.I)
CLAUSES = re.compile(r"[.!?;]|\b(?:but|however|whereas|yet|rather than)\b", re.I)
DIRECTION = re.compile(
    r"(?P<negative>\b(?:does not|do not|doesn[’']t|don[’']t)\s+(?:like|enjoy|prefer)|"
    r"\b(?:dislikes?|hates?|avoids?|rejects?|aversion to|averse to|less interested in|and not))"
    r"|(?P<positive>\b(?:likes?|loves?|enjoys?|prefers?|favors?|favours?|appreciates?|"
    r"preference for|interest in|receptive to|fond of|drawn to))", re.I)


def explicit_genre_preferences(summary: str) -> dict[str, list[str]]:
    positive, negative = set(), set()
    for clause in CLAUSES.split(summary):
        if ABSTENTION.search(clause):
            continue
        anchors = list(DIRECTION.finditer(clause))
        for index, anchor in enumerate(anchors):
            end = anchors[index + 1].start() if index + 1 < len(anchors) else len(clause)
            segment = clause[anchor.end():end]
            # "not interested in horror" is not a positive interest claim.
            if re.search(r"\bnot\s*$", clause[:anchor.start()], re.I):
                continue
            target = negative if anchor.lastgroup == "negative" else positive
            target.update(_mentioned_genres(segment))
    conflicting = positive & negative
    return {"preferred_genres": sorted(positive - conflicting),
            "avoided_genres": sorted(negative - conflicting),
            "conflicting_genres": sorted(conflicting)}


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
    finite = logits[torch.isfinite(logits)]
    if not finite.numel():
        return logits
    # A whole score range makes explicit alignment take precedence without
    # tuning a coefficient on one participant's requested titles.
    span = (finite.max() - finite.min()).clamp_min(1) + 1
    priority = preferred.to(logits.dtype) - 2 * avoided.to(logits.dtype)
    return logits + priority * span
