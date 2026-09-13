"""Offline, manually adjudicated v1-v2 calibration comparison."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any


V1 = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "calibration_100/v001_20260813"
)
V2 = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "calibration_100/v002_20260813_negative_grounding_safeguard"
)
AUDIT_DIR = V2 / "reports" / "paired_comparison_v1_v2"


# Manual, user-level adjudication against every supplied history. A supported
# result includes summaries that make no material negative claim because they
# abstain; inappropriate under-claiming is measured separately below.
UNSUPPORTED: dict[int, str] = {
    1288: "No rating is <=2.5; after abstaining, v2 still invents dislike of diminished-stakes/franchise and low-tension action/thriller plotting.",
    40516: "No rating is <=2.5, yet v2 asserts dislike of mainstream family animation and conventional light musicals.",
    41043: "No rating is <=2.5; v2 correctly abstains on genre but invents dislike of meandering ambiguity and experimental storytelling.",
    91121: "The only low-rated item is a war drama; the asserted dislikes of melodrama, slapstick comedy, and light fluff are unrelated to that evidence.",
    115873: "No rating is <=2.5; v2 abstains and then invents dislike of thin, cheaply executed standalone action.",
    119291: "No rating is <=2.5; v2 abstains and then invents dislikes of sentimental melodrama, meandering plots, and ungrounded fantasy.",
    185526: "No rating is <=2.5, yet v2 asserts dislikes of slapstick, lowbrow comedy, and lightweight romantic capers.",
}

OVERSTATED: dict[int, str] = {
    2015: "Two low ratings support a narrow musical/fairy-tale signal, but v2 adds conventional romance and derivative-installment claims not established by those items.",
    32589: "Some horror/action negatives exist, but jump-scare and formulaic-procedural aversions remain broader than the mixed evidence.",
    53501: "Two failed sci-fi spectacles and one romantic comedy support those narrow categories, but broad slapstick is an unsupported addition.",
    59442: "The action/blockbuster pattern is grounded; the broad slapstick/comedy generalization rests mainly on one low-rated comedy.",
    61038: "Children's animation and several romances are low-rated, but the added slapstick and gimmicky romantic-musical framing exceeds the supplied evidence.",
    61207: "Three low-rated action/war/superhero films support a cautious spectacle inference, not the added claim about inscrutable fantasy detours.",
    76281: "One low-rated dark comedy supports a narrow eccentric-comedy inference; children’s musicals, surrealism, and offbeat thrillers are unsupported additions.",
    108650: "Two low-rated dramas give limited evidence, but v2 generalizes them into a dislike of lightweight or purely character-driven drama without momentum.",
    138469: "Three mixed low-rated films provide no support for the added lightweight children's-musical/family-animation aversion.",
    156456: "One low-rated long-form drama supports only a cautious pacing inference; v2 expands this to metaphysical, experimental, enigmatic, and heavily stylized narratives.",
    165531: "Two failed pulpy sci-fi films support that narrow signal, but v2 adds slow, ambiguous, horror-leaning sci-fi and melodrama.",
    168549: "Exploitation violence is grounded, but the added slow, sentimental coming-of-age/children's-fantasy dislike overgeneralizes four mixed low ratings.",
    182046: "One low-rated psychological horror film supports a narrow grim-horror signal, not the added procedural crime, melodrama, and convoluted-plot dislikes.",
    183589: "Family/blockbuster aversion has support, but conventional Westerns and a broad spectacle generalization are not recurring negative patterns in the history.",
}

# Abstention assessment is independent of unsupported/overstated claims.
# "Inappropriate" means the statement suppresses a clear recurring negative
# pattern. "Mixed" means an evidentially sensible abstention is paired with a
# contradictory unsupported/overstated negative assertion.
INAPPROPRIATE_ABSTENTION = {
    7190,
    22939,
    25944,
    30459,
    44856,
    48604,
    69317,
    80161,
    83613,
    146379,
    169241,
    178348,
    179370,
}
MIXED_ABSTENTION = {1288, 2015, 40516, 41043, 91121, 115873, 119291, 185526}
NO_ABSTENTION = {190890}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p1 = load_json(V1 / "request_plan.json")
    p2 = load_json(V2 / "request_plan.json")
    v1 = {x["user_id"]: x for x in load_jsonl(V1 / "validated/all_summaries.jsonl")}
    v2 = {x["user_id"]: x for x in load_jsonl(V2 / "validated/all_summaries.jsonl")}
    a1 = {
        x["user_id"]: x
        for x in load_jsonl(V1 / "reports/grounding_audit_v1/per_user_grounding_audit.jsonl")
    }
    records = {x["user_id"]: x for x in p2["records"]}
    ids = p2["sample"]["user_ids"]
    assert ids == p1["sample"]["user_ids"]
    assert len(ids) == len(set(ids)) == 100
    assert not (set(UNSUPPORTED) & set(OVERSTATED))
    assert not (INAPPROPRIATE_ABSTENTION & MIXED_ABSTENTION)
    assert (INAPPROPRIATE_ABSTENTION | MIXED_ABSTENTION | NO_ABSTENTION) <= set(ids)

    paired: list[dict[str, Any]] = []
    for uid in ids:
        old = v1[uid]
        new = v2[uid]
        prior = a1[uid]
        record = records[uid]
        if uid in UNSUPPORTED:
            v2_verdict = "unsupported"
            note = UNSUPPORTED[uid]
        elif uid in OVERSTATED:
            v2_verdict = "overstated"
            note = OVERSTATED[uid]
        else:
            v2_verdict = "supported"
            note = "Material negative claims are tied to supplied low-rated evidence, or v2 abstains without inventing a negative claim."
        if uid in NO_ABSTENTION:
            abstention = "none"
        elif uid in INAPPROPRIATE_ABSTENTION:
            abstention = "inappropriate"
        elif uid in MIXED_ABSTENTION:
            abstention = "mixed_or_contradictory"
        else:
            abstention = "appropriate"
        old_failure = prior["negative_claim_verdict"] != "supported"
        new_failure = v2_verdict != "supported"
        if old_failure and not new_failure:
            transition = "resolved_v1_failure"
        elif old_failure and new_failure:
            transition = "persistent_failure"
        elif not old_failure and new_failure:
            transition = "new_v2_failure"
        else:
            transition = "remained_grounded"
        old_norm, new_norm = normalized(old["summary"]), normalized(new["summary"])
        low_items = [
            {"title": title, "rating": rating, "genres": genres}
            for title, rating, genres in zip(record["titles"], record["ratings"], record["genres"])
            if rating <= 2.5
        ]
        high_items = [
            {"title": title, "rating": rating, "genres": genres}
            for title, rating, genres in zip(record["titles"], record["ratings"], record["genres"])
            if rating >= 4.0
        ]
        paired.append(
            {
                "user_id": uid,
                "split": record["split"],
                "activity_band": record["activity_band"],
                "history_hash": record["history_hash"],
                "negative_evidence_tier": prior["negative_evidence_tier"],
                "low_rated_items": low_items,
                "high_rated_items": high_items,
                "v1": {
                    "summary": old["summary"],
                    "word_count": old["word_count"],
                    "negative_verdict": prior["negative_claim_verdict"],
                    "grounding_failure": old_failure,
                },
                "v2": {
                    "summary": new["summary"],
                    "word_count": new["word_count"],
                    "negative_verdict": v2_verdict,
                    "grounding_failure": new_failure,
                    "adjudication_note": note,
                    "abstention_status": abstention,
                    "positive_preference_grounding": "supported",
                    "positive_grounding_note": "At least one stated positive genre occurs in the supplied history, and the material themes/plots align with high-rated supplied titles.",
                    "semantic_structure": {
                        "liked_genres": True,
                        "liked_themes_or_plots": True,
                        "disliked_genres_or_styles_or_explicit_abstention": True,
                        "disliked_plots_or_content_or_explicit_abstention": True,
                    },
                    "privacy_detector": new["privacy"],
                    "confirmed_identifiable_privacy_leakage": False,
                },
                "paired_transition": transition,
                "focus_v1_grounding_failure": old_failure,
                "summary_exactly_equal": old_norm == new_norm,
                "summary_sequence_similarity": SequenceMatcher(
                    None, old_norm, new_norm, autojunk=False
                ).ratio(),
            }
        )

    tier_v1: dict[str, Counter[str]] = defaultdict(Counter)
    tier_v2: dict[str, Counter[str]] = defaultdict(Counter)
    for x in paired:
        tier = x["negative_evidence_tier"]
        tier_v1[tier][x["v1"]["negative_verdict"]] += 1
        tier_v2[tier][x["v2"]["negative_verdict"]] += 1
    v1_counts = Counter(x["v1"]["negative_verdict"] for x in paired)
    v2_counts = Counter(x["v2"]["negative_verdict"] for x in paired)
    transitions = Counter(x["paired_transition"] for x in paired)
    abstentions = Counter(x["v2"]["abstention_status"] for x in paired)
    by_tier_abstention: dict[str, Counter[str]] = defaultdict(Counter)
    for x in paired:
        by_tier_abstention[x["negative_evidence_tier"]][x["v2"]["abstention_status"]] += 1

    r1 = load_json(V1 / "reports/validation_report.json")
    r2 = load_json(V2 / "reports/validation_report.json")
    words1 = [x["v1"]["word_count"] for x in paired]
    words2 = [x["v2"]["word_count"] for x in paired]
    v1_failure = v1_counts["overstated"] + v1_counts["unsupported"]
    v2_failure = v2_counts["overstated"] + v2_counts["unsupported"]
    audit = {
        "audit_version": "v1-v2-paired-grounding-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "users": 100,
            "same_user_ids": True,
            "same_history_hashes": all(
                a["history_hash"] == b["history_hash"]
                for a, b in zip(p1["records"], p2["records"])
            ),
            "new_api_requests_beyond_v2_calibration": 0,
            "full_cohort_started": False,
            "tears_training_started": False,
        },
        "negative_grounding": {
            "v1_counts": dict(v1_counts),
            "v2_counts": dict(v2_counts),
            "v1_failure_count": v1_failure,
            "v2_failure_count": v2_failure,
            "v1_failure_rate": v1_failure / 100,
            "v2_failure_rate": v2_failure / 100,
            "absolute_rate_reduction": (v1_failure - v2_failure) / 100,
            "relative_failure_reduction": (v1_failure - v2_failure) / v1_failure,
            "by_evidence_tier": {
                tier: {
                    "users": sum(tier_v2[tier].values()),
                    "v1": dict(tier_v1[tier]),
                    "v2": dict(tier_v2[tier]),
                    "v1_failure_rate": (
                        tier_v1[tier]["overstated"] + tier_v1[tier]["unsupported"]
                    ) / sum(tier_v1[tier].values()),
                    "v2_failure_rate": (
                        tier_v2[tier]["overstated"] + tier_v2[tier]["unsupported"]
                    ) / sum(tier_v2[tier].values()),
                }
                for tier in ("strong", "weak", "none")
            },
        },
        "paired_transitions": dict(transitions),
        "v1_failure_focus": {
            "total": v1_failure,
            "resolved": transitions["resolved_v1_failure"],
            "persistent": transitions["persistent_failure"],
            "resolved_user_ids": [
                x["user_id"] for x in paired if x["paired_transition"] == "resolved_v1_failure"
            ],
            "persistent_user_ids": [
                x["user_id"] for x in paired if x["paired_transition"] == "persistent_failure"
            ],
            "new_v2_failure_user_ids": [
                x["user_id"] for x in paired if x["paired_transition"] == "new_v2_failure"
            ],
        },
        "abstentions": {
            "summaries_with_abstention": 100 - abstentions["none"],
            "appropriate": abstentions["appropriate"],
            "inappropriate_underclaiming": abstentions["inappropriate"],
            "mixed_or_contradictory": abstentions["mixed_or_contradictory"],
            "none": abstentions["none"],
            "by_evidence_tier": {
                tier: dict(by_tier_abstention[tier]) for tier in ("strong", "weak", "none")
            },
        },
        "positive_preference_grounding": {
            "supported": 100,
            "overstated_or_unsupported": 0,
            "automated_users_with_at_least_one_history_genre": r2["grounding"]["users_with_supported_history_genre"],
            "manual_finding": "All material positive genre and theme/plot claims were traceable to supplied high-rated history items. No material positive grounding regression from v1 was found.",
        },
        "semantic_structure": {
            "liked_genres": 100,
            "liked_themes_or_plots": 100,
            "disliked_genres_or_styles_or_explicit_abstention": 100,
            "disliked_plots_or_content_or_explicit_abstention": 100,
            "exact_four_sentence_rule_applied": False,
            "note": "An explicit evidence-based abstention satisfies a negative slot; distinct negative sentences are not required.",
        },
        "privacy": {
            "v1_automated_title_matches": len(r1["privacy_leakage"]["movie_title_user_ids"]),
            "v2_automated_title_matches": len(r2["privacy_leakage"]["movie_title_user_ids"]),
            "v2_automated_title_match_user_ids": r2["privacy_leakage"]["movie_title_user_ids"],
            "v2_confirmed_identifiable_title_leaks": 0,
            "v2_year_leaks": len(r2["privacy_leakage"]["year_user_ids"]),
            "v2_numeric_rating_leaks": len(r2["privacy_leakage"]["numeric_rating_user_ids"]),
            "manual_note": "The sole v2 detector match was the generic phrase 'alien threats' overlapping the one-word title 'Alien'; it is not an identifiable title disclosure.",
        },
        "duplicates": {
            "v2_exact_cross_user_duplicate_groups": r2["cross_user_duplicate_groups"],
            "v2_near_duplicate_pairs": r2["cross_user_near_duplicate_pairs"],
            "paired_v1_v2_exactly_equal": sum(x["summary_exactly_equal"] for x in paired),
        },
        "length": {
            "v1": {
                "minimum": min(words1),
                "mean": statistics.mean(words1),
                "median": statistics.median(words1),
                "maximum": max(words1),
                "below_120": sum(x < 120 for x in words1),
            },
            "v2": {
                "minimum": min(words2),
                "mean": statistics.mean(words2),
                "median": statistics.median(words2),
                "maximum": max(words2),
                "below_120": sum(x < 120 for x in words2),
            },
            "mean_change_words": statistics.mean(words2) - statistics.mean(words1),
        },
        "api_and_cost": {
            "successful_api_responses": r2["successful_api_responses"],
            "successful_structured_responses": r2["successful_structured_responses"],
            "missing": len(r2["missing_user_ids"]),
            "input_tokens": r2["usage"]["input_tokens"],
            "cached_input_tokens": r2["usage"]["cached_input_tokens"],
            "uncached_input_tokens": r2["usage"]["uncached_input_tokens"],
            "output_tokens": r2["usage"]["output_tokens"],
            "reasoning_tokens": r2["usage"]["reasoning_tokens_in_output"],
            "total_tokens": r2["usage"]["input_tokens"] + r2["usage"]["output_tokens"],
            "pricing_usd_per_million": r2["cost"]["pricing_usd_per_million"],
            "exact_cost_usd": r2["cost"]["exact_calibration_usd"],
            "average_cost_per_user_usd": r2["cost"]["mean_per_user_usd"],
        },
        "quality_findings": [
            "The safeguard appears in scoped or blanket form in 99/100 summaries, creating repetitive boilerplate.",
            "Thirteen abstentions suppress a clear recurring negative pattern; eight more are mixed or contradicted by a problematic negative assertion.",
            "Some outputs use awkward literal constructions such as 'The user does not enjoy: no strong negative preference...' and one contains a grammatical corruption ('broadly thatmost').",
            "The safeguard reduced negative overclaiming but shortened outputs and increased sub-120-word summaries from 52 to 70.",
        ],
        "conclusion": {
            "substantial_reduction": True,
            "structure_materially_preserved": True,
            "ready_for_full_cohort": False,
            "reason": "Negative grounding failures fell materially, but 21% still contain unsupported/overstated dislikes, 75% of no-evidence users still fail, and 13 summaries understate clear negative evidence through blanket abstention.",
        },
    }

    AUDIT_DIR.mkdir(parents=True, exist_ok=False)
    with (AUDIT_DIR / "per_user_paired_comparison.jsonl").open("w", encoding="utf-8") as handle:
        for row in paired:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (AUDIT_DIR / "paired_comparison_report.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# V2 calibration and paired v1-v2 audit",
        "",
        "## Outcome",
        "",
        f"V2 reduced unsupported/overstated negative outcomes from **{v1_failure}/100 ({v1_failure}%)** to **{v2_failure}/100 ({v2_failure}%)**: a **{v1_failure-v2_failure}-point absolute** and **{(v1_failure-v2_failure)/v1_failure:.1%} relative** reduction. Of the 44 v1 failures, {transitions['resolved_v1_failure']} were resolved and {transitions['persistent_failure']} persisted; {transitions['new_v2_failure']} new failures appeared.",
        "",
        "| Evidence | V1 supported | V1 over/unsupported | V1 failure | V2 supported | V2 overstated | V2 unsupported | V2 failure |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in ("strong", "weak", "none"):
        n = sum(tier_v2[tier].values())
        f1 = tier_v1[tier]["overstated"] + tier_v1[tier]["unsupported"]
        f2 = tier_v2[tier]["overstated"] + tier_v2[tier]["unsupported"]
        lines.append(
            f"| {tier.title()} ({n}) | {tier_v1[tier]['supported']} | {f1} | {f1/n:.1%} | {tier_v2[tier]['supported']} | {tier_v2[tier]['overstated']} | {tier_v2[tier]['unsupported']} | {f2/n:.1%} |"
        )
    lines += [
        "",
        "## Abstention and structure",
        "",
        f"An abstention appears in **{100-abstentions['none']}/100** summaries: {abstentions['appropriate']} are appropriate, {abstentions['inappropriate']} inappropriately suppress clear negative evidence, and {abstentions['mixed_or_contradictory']} are mixed or contradicted by a problematic negative assertion. All 100 preserve the four semantic slots when an explicit abstention is accepted for an unsupported negative slot; no exactly-four-sentence rule was used.",
        "",
        "Positive genre grounding and manually reviewed positive theme/plot grounding were supported for all 100 users. The safeguard did not materially harm positive content or four-part semantic coverage, but it did reduce negative specificity in the inappropriate-abstention cases.",
        "",
        "## Other validation",
        "",
        f"- API/structured responses: {r2['successful_api_responses']}/100 and {r2['successful_structured_responses']}/100; missing: {len(r2['missing_user_ids'])}.",
        f"- Privacy: 0 year leaks, 0 numeric-rating leaks, and 0 confirmed identifiable title leaks. One automated match was the generic phrase 'alien threats' overlapping the title 'Alien'.",
        f"- Duplicates: {r2['cross_user_duplicate_groups']} exact cross-user groups and {r2['cross_user_near_duplicate_pairs']} near-duplicate pairs.",
        f"- Length: v2 minimum {min(words2)}, mean {statistics.mean(words2):.2f}, median {statistics.median(words2):.1f}, maximum {max(words2)}; {sum(x<120 for x in words2)} are below 120 words. V1 mean was {statistics.mean(words1):.2f}, with {sum(x<120 for x in words1)} below 120.",
        "",
        "## Exact usage and cost",
        "",
        f"V2 used **{r2['usage']['input_tokens']:,} input tokens** (all uncached), **{r2['usage']['output_tokens']:,} output tokens**, and **{r2['usage']['reasoning_tokens_in_output']:,} reasoning tokens**, for **${r2['cost']['exact_calibration_usd']:.8f}** total and **${r2['cost']['mean_per_user_usd']:.9f}** per user.",
        "",
        "## Recommendation",
        "",
        "The minimal safeguard **substantially reduces** unsupported/overstated dislike claims without removing the intended four semantic content slots. It is not yet safe to promote unchanged: no-evidence failures remain high, blanket abstention sometimes hides real negative patterns, and the safeguard creates repetitive or awkward prose. Do not start full-cohort generation or TEARS training; refine the conditional behavior and re-calibrate first.",
    ]
    (AUDIT_DIR / "paired_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "manifest_version": "v2-paired-audit-v1",
        "source_artifacts": {
            "v1/request_plan.json": sha256(V1 / "request_plan.json"),
            "v1/validated/all_summaries.jsonl": sha256(V1 / "validated/all_summaries.jsonl"),
            "v1/reports/grounding_audit_v1/per_user_grounding_audit.jsonl": sha256(V1 / "reports/grounding_audit_v1/per_user_grounding_audit.jsonl"),
            "v2/request_plan.json": sha256(V2 / "request_plan.json"),
            "v2/responses": {
                p.name: sha256(p) for p in sorted((V2 / "responses").glob("*.jsonl"))
            },
            "v2/validated/all_summaries.jsonl": sha256(V2 / "validated/all_summaries.jsonl"),
            "v2/prompt/protocol_delta.json": sha256(V2 / "prompt/protocol_delta.json"),
        },
        "audit_artifacts": {},
    }
    for name in ("per_user_paired_comparison.jsonl", "paired_comparison_report.json", "paired_comparison_report.md"):
        manifest["audit_artifacts"][name] = sha256(AUDIT_DIR / name)
    (AUDIT_DIR / "paired_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
