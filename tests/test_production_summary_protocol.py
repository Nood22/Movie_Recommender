from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from tears_training import production_summary_protocol as protocol


CALIBRATION_ROOT = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "calibration_100"
)


def _evidence(genres: list[str]) -> dict:
    return {"supported_negative_genres": genres}


def test_repair_api_has_no_user_identifier_or_manual_label_input() -> None:
    parameters = inspect.signature(protocol.repair_summary).parameters
    assert tuple(parameters) == ("summary", "evidence")
    source = inspect.getsource(protocol)
    forbidden = (
        "per_user_paired_comparison",
        "per_user_repair_audit",
        "manual_claim",
        "adjudication_note",
        "INAPPROPRIATE_ABSTENTION_USERS",
        "REPAIR_SPECS",
    )
    assert not any(value in source for value in forbidden)


def test_no_hard_coded_calibration_ids_or_integer_keyed_exception_maps() -> None:
    original_ids = set(
        json.loads(
            (
                CALIBRATION_ROOT
                / "v002_20260813_negative_grounding_safeguard"
                / "calibration_user_ids.json"
            ).read_text()
        )["user_ids"]
    )
    tree = ast.parse(inspect.getsource(protocol))
    integer_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    }
    assert not (integer_literals & original_ids)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and node.keys:
            assert not all(
                isinstance(key, ast.Constant)
                and isinstance(key.value, int)
                and not isinstance(key.value, bool)
                for key in node.keys
                if key is not None
            )


def test_repair_is_invariant_to_irrelevant_evidence_fields() -> None:
    summary = (
        "Summary: The user enjoys dramas and likes layered character stories. "
        "No strong negative preference is supported by the available history."
    )
    first = protocol.repair_summary(summary, _evidence(["Horror"]))
    second = protocol.repair_summary(
        summary,
        {
            "supported_negative_genres": ["Horror"],
            "user_id": 123456,
            "manual_label": "must_be_ignored",
        },
    )
    assert first.final_summary == second.final_summary
    assert first.operations == second.operations


def test_no_evidence_neutralizes_only_negative_span_and_preserves_positive() -> None:
    positive = "Summary: The user enjoys dramas and likes layered character stories."
    raw = positive + " The user dislikes slapstick and slow, sentimental plots."
    result = protocol.repair_summary(raw, _evidence([]))
    assert positive in result.final_summary
    assert "slapstick" not in result.final_summary
    assert protocol.NEUTRAL_ABSTENTION in result.final_summary
    assert result.positive_sentences_preserved
    assert not result.inappropriate_abstention


def test_supported_evidence_replaces_abstention_with_narrow_genres() -> None:
    raw = (
        "The user enjoys mysteries and clever stories. "
        "No strong negative preference is supported by the available history."
    )
    result = protocol.repair_summary(raw, _evidence(["Horror", "Sci-Fi"]))
    assert result.final_summary.startswith("Summary:")
    assert "horror films and science-fiction films" in result.final_summary
    assert "No strong negative" not in result.final_summary
    assert not result.inappropriate_abstention
    assert not result.supported_negative_evidence_omitted


def test_overstatement_is_narrowed_to_supported_genres_only() -> None:
    raw = (
        "Summary: The user enjoys dramas and complex stories. "
        "The user does not enjoy horror, slapstick, slow pacing, or sentimental plots."
    )
    result = protocol.repair_summary(raw, _evidence(["Horror"]))
    assert "horror films" in result.final_summary
    assert "slapstick" not in result.final_summary
    assert "slow pacing" not in result.final_summary
    assert "sentimental plots" not in result.final_summary


def test_narrow_supported_negative_claim_is_preserved_byte_for_byte() -> None:
    negative = "The user does not enjoy horror films or thrillers."
    raw = "Summary: The user enjoys dramas. " + negative
    result = protocol.repair_summary(raw, _evidence(["Horror", "Thriller"]))
    assert result.final_summary == raw
    assert result.supported_negative_sentences_preserved == (negative,)
    assert not result.changed


def test_supported_genre_does_not_license_unsupported_style_claims() -> None:
    raw = (
        "Summary: The user enjoys dramas. "
        "The user does not enjoy formulaic action spectacles with shallow plots."
    )
    result = protocol.repair_summary(raw, _evidence(["Action"]))
    assert "formulaic" not in result.final_summary
    assert "shallow plots" not in result.final_summary
    assert "negative preference for action films" in result.final_summary


def test_appropriate_abstention_is_retained_while_extra_claim_is_removed() -> None:
    abstention = protocol.NEUTRAL_ABSTENTION
    raw = (
        "Summary: The user enjoys dramas. "
        + abstention
        + " The user is less enthusiastic about slow pacing and sentimental plots."
    )
    result = protocol.repair_summary(raw, _evidence([]))
    assert abstention in result.final_summary
    assert "slow pacing" not in result.final_summary
    assert result.final_summary.count(abstention) == 1


def test_prefix_is_restored_when_first_sentence_is_a_repair_target() -> None:
    raw = (
        "Summary: The user dislikes formulaic comedy and sentimental plots. "
        "The user enjoys dramas and mysteries."
    )
    result = protocol.repair_summary(raw, _evidence([]))
    assert result.final_summary.startswith("Summary: ")
    assert "The user enjoys dramas and mysteries." in result.final_summary
    assert result.formatting_issues == ()


def test_negative_claim_combined_with_abstention_is_not_retained() -> None:
    raw = (
        "Summary: The user enjoys crime dramas. "
        "The user does not enjoy children's animation; no strong negative "
        "preference is supported by the available history."
    )
    result = protocol.repair_summary(raw, _evidence([]))
    assert "children's animation" not in result.final_summary
    assert result.final_summary.count(protocol.NEUTRAL_ABSTENTION) == 1


def test_indirect_lower_rating_language_is_repaired() -> None:
    raw = (
        "Summary: The user enjoys dramas. The user rates glossy action spectacles "
        "and formulaic plots less favorably."
    )
    result = protocol.repair_summary(raw, _evidence(["Action"]))
    assert "glossy" not in result.final_summary
    assert "formulaic" not in result.final_summary
    assert "negative preference for action films" in result.final_summary


def test_other_viewer_leadin_does_not_hide_later_user_negative_claim() -> None:
    raw = (
        "Summary: The user enjoys dramas. "
        "Other users may enjoy broad animation, which the user rates lower."
    )
    result = protocol.repair_summary(raw, _evidence([]))
    assert "rates lower" not in result.final_summary
    assert protocol.NEUTRAL_ABSTENTION in result.final_summary


def test_literal_prompt_instruction_and_placeholder_are_removed_nonsemantically() -> None:
    raw = (
        "Summary: The user enjoys dramas. "
        + protocol.NEUTRAL_ABSTENTION
        + " If the user's rating history does not provide sufficient evidence for "
        "a negative preference, do not infer or invent one. Instead, state that no "
        "strong negative preference is supported by the available history. "
        "{Specific details of plot points the user seems to enjoy}."
    )
    result = protocol.repair_summary(raw, _evidence([]))
    assert "do not infer" not in result.final_summary
    assert "{Specific details" not in result.final_summary
    assert not result.literal_placeholder


def test_formatting_normalization_is_generic_and_nonsemantic() -> None:
    raw = (
        "The user enjoys drama. The user does not enjoy: no strong negative "
        "preference is supported by the available history."
    )
    result = protocol.repair_summary(raw, _evidence([]))
    assert result.final_summary == (
        "Summary: The user enjoys drama. "
        "No strong negative preference is supported by the available history."
    )
    assert result.formatting_issues == ()
