from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from tears_training import final_production_summary_protocol as final


CALIBRATION = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "production_calibration_1000"
)
SOURCE = CALIBRATION / "v002_20260816_v2_generic_repair"
VALIDATED = CALIBRATION / "v008_20260816_v2_generic_repair_final"


def _evidence(negative: list[str], positive: list[str] | None = None) -> dict:
    return {
        "supported_negative_genres": negative,
        "supported_positive_genres": positive or [],
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_composition_api_has_no_user_or_manual_label_input() -> None:
    assert tuple(inspect.signature(final.compose_final_summary).parameters) == (
        "positive_candidate",
        "evidence",
        "history_titles",
    )
    source = inspect.getsource(final)
    forbidden = (
        "manual_label",
        "adjudication_note",
        "REPAIR_SPECS",
        "INAPPROPRIATE_ABSTENTION_USERS",
    )
    assert not any(value in source.lower() for value in forbidden)
    tree = ast.parse(source)
    imports = {
        alias.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any("openai" in value or "tmdb" in value for value in imports)


def test_no_integer_keyed_user_exception_map() -> None:
    tree = ast.parse(inspect.getsource(final))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and node.keys:
            assert not all(
                isinstance(key, ast.Constant)
                and isinstance(key.value, int)
                and not isinstance(key.value, bool)
                for key in node.keys
                if key is not None
            )


def test_final_prompt_is_positive_only_and_prefix_is_code_owned() -> None:
    prompt = final.FINAL_POSITIVE_SYSTEM_PROMPT
    assert "Do not generate negative preferences" in prompt
    assert "Do not add a Summary: prefix" in prompt
    assert "other users may enjoy" in prompt
    assert "{Specific details" not in prompt


def test_free_form_negative_content_is_discarded_and_slot_is_deterministic() -> None:
    raw = (
        "Summary: The user enjoys layered dramas and mysteries. "
        "The user dislikes formulaic horror and slow sentimental plots. "
        "Other users may enjoy those conventional stories."
    )
    result = final.compose_final_summary(raw, _evidence(["Horror"]), [])
    assert "layered dramas and mysteries" in result.final_summary
    assert "formulaic" not in result.final_summary
    assert "Other users" not in result.final_summary
    assert result.final_summary.endswith(
        "The available history supports a negative preference for horror films."
    )


def test_no_negative_evidence_gets_exactly_one_neutral_abstention() -> None:
    raw = "The user enjoys clever mysteries."
    result = final.compose_final_summary(raw, _evidence([]), [])
    assert result.final_summary.startswith("Summary: ")
    assert result.final_summary.count(final.legacy.NEUTRAL_ABSTENTION) == 1
    assert result.final_summary.count("Summary:") == 1


def test_title_parenthetical_is_removed_without_rewriting_surrounding_prose() -> None:
    raw = (
        "The user enjoys cerebral science fiction with moral complexity "
        "(e.g., Gattaca, RoboCop). They appreciate witty comedies."
    )
    history = ["Gattaca (1997)", "RoboCop (1987)"]
    result = final.compose_final_summary(raw, _evidence([]), history)
    assert "Gattaca" not in result.final_summary
    assert "RoboCop" not in result.final_summary
    assert "cerebral science fiction with moral complexity" in result.final_summary
    assert "They appreciate witty comedies." in result.final_summary


def test_short_franchise_title_and_like_cue_are_sanitized() -> None:
    raw = "The user enjoys lean thrillers like Memento and Toy Story-style animation."
    history = ["Memento (2000)", "Toy Story 3 (2010)"]
    result = final.compose_final_summary(raw, _evidence([]), history)
    assert "Memento" not in result.final_summary
    assert "Toy Story" not in result.final_summary


def test_ambiguous_single_word_title_is_not_a_leak_without_title_context() -> None:
    raw = "The user enjoys stories about alien worlds and contact with strangers."
    history = ["Alien (1979)", "Contact (1997)"]
    result = final.compose_final_summary(raw, _evidence([]), history)
    assert "alien worlds" in result.final_summary
    assert "contact with strangers" in result.final_summary


def test_year_and_numeric_rating_are_removed() -> None:
    raw = (
        "The user enjoys 1997 science-fiction dramas and rated the strongest examples 4.5/5."
    )
    result = final.compose_final_summary(raw, _evidence([]), [])
    assert "1997" not in result.final_summary
    assert "4.5" not in result.final_summary
    assert result.year_leaks_after == ()
    assert result.numeric_rating_leaks_after == ()


def test_direct_positive_negative_genre_contradiction_is_removed() -> None:
    raw = (
        "The user enjoys action films with tight pacing. "
        "They appreciate morally complex character studies."
    )
    result = final.compose_final_summary(raw, _evidence(["Action"]), [])
    assert "enjoys action" not in result.final_summary
    assert "morally complex character studies" in result.final_summary
    assert result.positive_negative_contradictions_before == ("Action",)
    assert result.positive_negative_contradictions_after == ()


def test_rather_than_action_is_not_a_positive_action_claim() -> None:
    raw = (
        "The user tends to like crime drama when it emphasizes atmosphere "
        "rather than pulpy action."
    )
    result = final.compose_final_summary(raw, _evidence(["Action"]), [])
    assert "rather than pulpy action" in result.final_summary
    assert result.positive_negative_contradictions_before == ()


def test_empty_positive_after_consistency_uses_grounded_positive_fallback() -> None:
    raw = "The user enjoys action films."
    result = final.compose_final_summary(
        raw,
        _evidence(["Action"], ["Drama"]),
        [],
    )
    assert "positive preference for dramas" in result.final_summary
    assert "negative preference for action films" in result.final_summary


def test_known_calibration_contradiction_and_title_leak_mechanisms() -> None:
    rows = {
        int(row["user_id"]): row
        for row in _jsonl(VALIDATED / "validated/all_raw_and_final_summaries.jsonl")
    }
    evidence = {
        int(row["user_id"]): row
        for row in _jsonl(SOURCE / "evidence/all_user_evidence.jsonl")
    }
    plans = {
        int(row["user_id"]): row
        for row in json.loads((SOURCE / "request_plan.json").read_text())["records"]
    }
    substantive = {
        8742,
        18515,
        50674,
        60875,
        65242,
        83310,
        92810,
        159382,
        166368,
        199226,
    }
    for user_id in sorted(substantive):
        result = final.compose_final_summary(
            rows[user_id]["raw_summary"],
            evidence[user_id],
            plans[user_id]["titles"],
        )
        assert result.positive_negative_contradictions_before
        assert result.positive_negative_contradictions_after == ()
    lexical_false_positive = final.compose_final_summary(
        rows[118834]["raw_summary"],
        evidence[118834],
        plans[118834]["titles"],
    )
    assert lexical_false_positive.positive_negative_contradictions_before == ()
    assert "rather than pulpy action" in lexical_false_positive.final_summary

    for user_id in (35148, 39004, 92423, 127899):
        result = final.compose_final_summary(
            rows[user_id]["raw_summary"],
            evidence[user_id],
            plans[user_id]["titles"],
        )
        assert result.title_matches_before
        assert result.title_matches_after == ()
