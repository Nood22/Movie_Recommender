from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from tears_training import evidence_gated_summary_harness as harness


CALIBRATION = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "production_calibration_1000/v002_20260816_v2_generic_repair"
)


def record(ratings: list[float], genres: list[str] | None = None) -> dict:
    genres = genres or ["Drama"] * len(ratings)
    return {
        "user_id": 1,
        "history_hash": "history",
        "history_items": len(ratings),
        "movie_ids": list(range(1, len(ratings) + 1)),
        "titles": [f"Private Film {index} (2000)" for index in range(len(ratings))],
        "ratings": ratings,
        "genres": genres,
    }


def test_none_status_supplies_no_negative_examples() -> None:
    evidence = harness.build_separated_evidence(record([4.0, 4.5, 3.5]))
    assert evidence["negative_evidence_status"] == "NONE"
    assert evidence["inference_payload"]["negative_evidence"] == []
    prompt = harness.render_evidence_prompt(evidence)
    assert "NEGATIVE EVIDENCE STATUS: NONE" in prompt
    assert "Preference signal: NEGATIVE" not in prompt


def test_weak_status_uses_isolated_negative_examples_without_genre_dislike() -> None:
    evidence = harness.build_separated_evidence(
        record([4.5, 4.0, 1.0], ["Drama", "Drama", "Action"])
    )
    assert evidence["negative_evidence_status"] == "WEAK"
    assert evidence["negative_example_count"] == 1
    assert evidence["genre_classifications"]["supported_negative"] == []


def test_strong_status_requires_frozen_supported_negative_genre() -> None:
    evidence = harness.build_separated_evidence(
        record([1.0, 1.5, 2.0], ["Action", "Action", "Action"])
    )
    assert evidence["negative_evidence_status"] == "STRONG"
    assert evidence["genre_classifications"]["supported_negative"] == ["Action"]


def test_many_positive_plus_one_negative_is_not_broad_negative() -> None:
    evidence = harness.build_separated_evidence(
        record(
            [5.0, 4.5, 4.0, 2.0],
            ["Action", "Action", "Action", "Action"],
        )
    )
    assert evidence["genre_classifications"]["supported_positive"] == ["Action"]
    assert evidence["genre_classifications"]["supported_negative"] == []
    assert evidence["negative_evidence_status"] == "WEAK"


def test_repeated_positive_and_negative_is_mixed_not_supported() -> None:
    evidence = harness.build_separated_evidence(
        record(
            [5.0, 4.5, 4.0, 2.5, 2.0, 1.0],
            ["Action"] * 6,
        )
    )
    assert evidence["genre_classifications"]["mixed_conflicting"] == ["Action"]
    assert evidence["genre_classifications"]["supported_positive"] == []
    assert evidence["genre_classifications"]["supported_negative"] == []
    assert evidence["negative_evidence_status"] == "WEAK"


def test_neutral_interactions_are_excluded_from_both_blocks() -> None:
    evidence = harness.build_separated_evidence(record([3.0, 3.5]))
    assert evidence["positive_example_count"] == 0
    assert evidence["negative_example_count"] == 0
    assert evidence["neutral_excluded_count"] == 2


def test_request_preserves_model_and_generation_settings() -> None:
    item = record([5.0])
    evidence = harness.build_separated_evidence(item)
    request = harness.build_response_request(item, evidence)
    body = request["body"]
    assert body["model"] == "gpt-5-mini-2025-08-07"
    assert body["reasoning"] == {"effort": "minimal"}
    assert body["text"]["verbosity"] == "low"
    assert body["max_output_tokens"] == 450
    assert body["store"] is False
    assert body["input"][0]["content"] == harness.EVIDENCE_GATED_PROMPT
    assert "POSITIVE EVIDENCE:" in body["input"][1]["content"]


def test_cleanup_is_privacy_and_format_only() -> None:
    raw = (
        "The user enjoys thoughtful dramas (e.g., Private Film) and dislikes "
        "formulaic action."
    )
    result = harness.privacy_and_format_cleanup(raw, ["Private Film (2000)"])
    assert result.final_summary.startswith("Summary: ")
    assert "Private Film" not in result.final_summary
    assert "dislikes formulaic action" in result.final_summary
    assert all(
        operation["type"].startswith(("privacy_", "format_"))
        for operation in result.operations
    )


def test_harness_has_no_user_specific_or_semantic_repair_dependencies() -> None:
    source = inspect.getsource(harness)
    forbidden = (
        "REPAIR_SPECS",
        "INAPPROPRIATE_ABSTENTION_USERS",
        "manual_label",
        "adjudication_note",
        "repair_summary(",
        "compose_final_summary(",
    )
    assert not any(value.lower() in source.lower() for value in forbidden)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and node.keys:
            assert not all(
                isinstance(key, ast.Constant)
                and isinstance(key.value, int)
                and not isinstance(key.value, bool)
                for key in node.keys
                if key is not None
            )


def test_same_frozen_1000_ids_and_regression_cases_have_separated_blocks() -> None:
    plan = json.loads((CALIBRATION / "request_plan.json").read_text())
    records = {int(row["user_id"]): row for row in plan["records"]}
    assert len(records) == 1000
    # Prior substantive positive/negative contradiction cases are evaluation-only
    # fixtures. They are never referenced by production inference code.
    cases = {
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
    for user_id in cases:
        evidence = harness.build_separated_evidence(records[user_id])
        payload = evidence["inference_payload"]
        assert not (
            set(payload["supported_positive_genres"])
            & set(payload["supported_negative_genres"])
        )
        assert all(
            example["preference_signal"] == "POSITIVE"
            for example in payload["positive_evidence"]
        )
        assert all(
            example["preference_signal"] == "NEGATIVE"
            for example in payload["negative_evidence"]
        )


def test_prompt_delta_is_an_adaptation_not_exact_reproduction() -> None:
    delta = harness.prompt_delta()
    assert delta["exact_reproduction_claimed"] is False
    assert delta["source_prompt"] == harness.ORIGINAL_EMILIANO_PROMPT
    assert delta["adapted_prompt"] == harness.EVIDENCE_GATED_PROMPT
    assert delta["source_prompt_sha256"] != delta["adapted_prompt_sha256"]
