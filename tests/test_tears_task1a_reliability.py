from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import pilot_api
from tears_training.evidence_gated_summary_harness_v11 import build_separated_evidence
from tears_training.summaries import validate_text


SAFE_SUFFIX = (
    ", with this description limited to that broad tendency and not implying any "
    "additional genre theme plot element or viewing need beyond the information "
    "used to form this concise description"
)


def semantic_summary(parts: list[str]) -> str:
    return " ".join(
        f"{'Summary: ' if index == 0 else ''}{part}{SAFE_SUFFIX}."
        for index, part in enumerate(parts)
    )


CASES = {
    "positive": {
        "history": [
            "- title='Private Drama'; private_rating=5/5; genres=Drama|Comedy",
            "- title='Private Comedy'; private_rating=4/5; genres=Comedy",
        ],
        "parts": [
            "The viewer prefers drama and comedy",
            "They favor character focused stories and hopeful humor",
        ],
        "expected": ("prefers drama and comedy", "character focused stories"),
    },
    "negative": {
        "history": [
            "- title='Private Horror'; private_rating=1/5; genres=Horror",
            "- title='Private Thriller'; private_rating=2/5; genres=Thriller",
            "- explicitly_disliked=Private Horror",
        ],
        "parts": [
            "The viewer may be less interested in horror",
            "They may avoid frightening and graphic content",
        ],
        "expected": ("less interested in horror", "may avoid frightening"),
    },
    "mixed": {
        "history": [
            "- title='Private Drama'; private_rating=5/5; genres=Drama",
            "- title='Private Comedy'; private_rating=4/5; genres=Comedy",
            "- title='Private Horror'; private_rating=1/5; genres=Horror",
            "- title='Private Thriller'; private_rating=2/5; genres=Thriller",
            "- explicitly_disliked=Private Horror",
        ],
        "parts": [
            "The viewer prefers drama",
            "They favor character focused and emotionally grounded stories",
            "The viewer dislikes horror",
            "They avoid frightening and graphic content",
        ],
        "expected": ("prefers drama", "dislikes horror"),
    },
    "context": {
        "history": [
            "- title='Private Comedy'; private_rating=5/5; genres=Comedy",
            "- title='Private Drama'; private_rating=4/5; genres=Drama",
            "- viewing_context=I want to unwind",
        ],
        "parts": [
            "The viewer prefers comedy",
            "They favor gentle humor and relaxed stories when unwinding",
        ],
        "expected": ("prefers comedy", "when unwinding"),
    },
    "sparse": {
        "history": [
            "- title='Private Action'; private_rating=2/5; genres=Action|Adventure|Drama",
        ],
        "parts": [
            "The viewer may be less interested in action adventure and drama",
            "They may avoid fast paced spectacle",
        ],
        "expected": ("less interested in action adventure and drama", "fast paced spectacle"),
    },
}


class FakeResponses:
    def __init__(self, summaries: list[str]) -> None:
        self.summaries = iter(summaries)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps({"summary": next(self.summaries)})
        )


def recommender():
    return SimpleNamespace(
        config=SimpleNamespace(
            summaries=SimpleNamespace(min_words=120, max_words=260)
        )
    )


@pytest.mark.parametrize(
    ("case_name", "invalid_attempts"),
    (
        ("positive", 0),
        ("negative", 1),
        ("mixed", 1),
        ("context", 2),
        ("sparse", 1),
    ),
)
def test_representative_task_1a_cases_are_validator_gated_and_evidence_preserving(
    case_name: str, invalid_attempts: int
) -> None:
    case = CASES[case_name]
    valid = semantic_summary(case["parts"])
    assert pilot_api._summary_validation_errors(
        valid, recommender(), ["Private Movie"]
    ) == []
    invalid = "Summary: The viewer rated Private Drama 5/5 and prefers it."
    responses = FakeResponses([invalid] * invalid_attempts + [valid])
    client = SimpleNamespace(responses=responses)
    attempts: list[dict] = []

    summary, recorded = pilot_api._generate_valid_tears_summary(
        client,
        case["history"],
        recommender(),
        ["Private Drama", "Private Comedy", "Private Horror"],
        attempts,
    )

    assert summary == valid
    assert recorded is attempts
    assert len(recorded) == invalid_attempts + 1
    assert recorded[-1]["validator_errors"] == []
    assert recorded[-1]["word_count"] == len(valid.split())
    assert all(call["text"]["verbosity"] == "high" for call in responses.calls)
    assert all(call["max_output_tokens"] == 2000 for call in responses.calls)
    assert all(call["reasoning"]["effort"] == "low" for call in responses.calls)
    assert all(
        "Follow the supplied RICHNESS TARGET or SPARSE STYLE TARGET" in call["input"][1]["content"]
        and "unsupported categories have been omitted silently"
        in call["input"][1]["content"]
        and "without turning length into a validity condition"
        in call["input"][1]["content"]
        and "140-180" not in call["input"][1]["content"]
        and "exactly four sentences" not in call["input"][1]["content"]
        for call in responses.calls
    )
    assert all(
        "every positive claim comes from positive evidence" in call["input"][1]["content"]
        and "every negative claim comes from negative evidence" in call["input"][1]["content"]
        and "no opposite preference has been inferred" in call["input"][1]["content"]
        for call in responses.calls
    )
    assert all(
        call["input"][0]["content"] == pilot_api.ONLINE_TASK1A_SYSTEM_PROMPT
        for call in responses.calls
    )
    for call in responses.calls[1:]:
        repair_prompt = call["input"][1]["content"]
        assert "original private evidence" in repair_prompt
        assert "do not add genres, themes, plot elements" in repair_prompt
        assert "privacy and format validator" in repair_prompt
        assert "every statement about their absence" in repair_prompt
        assert "140-180" not in repair_prompt
        assert "all four sentences" not in repair_prompt
        assert all(line in repair_prompt for line in case["history"])
    lowered = summary.lower()
    assert all(expected in lowered for expected in case["expected"])
    assert "private drama" not in lowered
    assert "private comedy" not in lowered
    assert "private horror" not in lowered


def test_rating_leakage_feedback_preserves_evidence_without_repeating_metadata() -> None:
    case = CASES["mixed"]
    valid = semantic_summary(case["parts"])
    leaked = valid.replace(
        "The viewer prefers drama",
        "The viewer rated drama 5/5 and prefers it",
        1,
    )
    assert pilot_api._summary_validation_errors(
        leaked, recommender(), ["Private Drama"]
    ) == ["rating_leakage"]
    responses = FakeResponses([leaked, valid])

    summary, attempts = pilot_api._generate_valid_tears_summary(
        SimpleNamespace(responses=responses),
        case["history"],
        recommender(),
        ["Private Drama", "Private Comedy", "Private Horror", "Private Thriller"],
        [],
    )

    assert summary == valid
    assert attempts[0]["validator_errors"] == ["rating_leakage"]
    assert attempts[1]["validator_errors"] == []
    repair_prompt = responses.calls[1]["input"][1]["content"]
    assert "for: rating_leakage" in repair_prompt
    assert "expressing the underlying preference semantically" in repair_prompt
    assert "Do not merely delete a grounded preference claim" in repair_prompt
    assert "not mention the evidence measurement" in repair_prompt
    assert all(line in repair_prompt for line in case["history"])
    assert validate_text(summary, recommender().config, ["Private Drama"]) == []


def test_online_validator_accepts_concise_profile_without_four_sentences() -> None:
    concise = "Summary: The viewer enjoys drama and comedy."
    assert validate_text(concise, recommender().config, ["Private Movie"])
    assert pilot_api._summary_validation_errors(
        concise, recommender(), ["Private Movie"]
    ) == []
    responses = FakeResponses([concise])

    summary, attempts = pilot_api._generate_valid_tears_summary(
        SimpleNamespace(responses=responses),
        CASES["positive"]["history"],
        recommender(),
        ["Private Drama", "Private Comedy"],
        [],
    )

    assert summary == concise
    assert attempts[0]["validator_errors"] == []
    assert attempts[0]["generation_errors"] == []
    assert attempts[0]["retry_errors"] == []
    assert len(responses.calls) == 1


def test_online_generation_style_targets_follow_evidence_volume() -> None:
    sparse = build_separated_evidence(
        {
            "user_id": 0,
            "history_hash": "sparse-style",
            "history_items": 1,
            "movie_ids": [1],
            "titles": ["Private Horror"],
            "ratings": [1],
            "genres": ["Horror"],
        }
    )
    rich = build_separated_evidence(
        {
            "user_id": 0,
            "history_hash": "rich-style",
            "history_items": 4,
            "movie_ids": [1, 2, 3, 4],
            "titles": ["Private A", "Private B", "Private C", "Private D"],
            "ratings": [5, 4, 5, 4],
            "genres": ["Drama", "Drama", "Comedy", "Comedy"],
        }
    )

    sparse_prompt = pilot_api._render_online_inference_payload(sparse)
    rich_prompt = pilot_api._render_online_inference_payload(rich)

    assert "SPARSE STYLE TARGET (style only, not a validity gate)" in sparse_prompt
    assert "usually in 2 natural sentences" in sparse_prompt
    assert "RICHNESS TARGET (style only, not a validity gate)" in rich_prompt
    assert "3-4 sentences and roughly 90-150 words" in rich_prompt
    assert "Never pad, repeat, or invent content" in rich_prompt


def test_online_validator_rejects_third_person_agreement_error() -> None:
    assert pilot_api._summary_validation_errors(
        "Summary: The viewer likes drama, and they prefers comedies.",
        recommender(),
        ["Private Movie"],
    ) == ["grammar_subject_verb_agreement"]


def test_online_validator_rejects_audit_process_language() -> None:
    assert pilot_api._summary_validation_errors(
        "Summary: The viewer has a weak negative signal for horror.",
        recommender(),
        ["Private Movie"],
    ) == ["audit_process_language"]
    assert pilot_api._summary_validation_errors(
        "Summary: The viewer has a tentative, isolated dislike for horror.",
        recommender(),
        ["Private Movie"],
    ) == ["audit_process_language"]


def test_online_validator_silences_abstention_when_any_signal_exists() -> None:
    evidence = build_separated_evidence(
        {
            "user_id": 0,
            "history_hash": "negative-only",
            "history_items": 1,
            "movie_ids": [1],
            "titles": ["Private Horror"],
            "ratings": [1],
            "genres": ["Horror"],
        }
    )
    errors = pilot_api._summary_validation_errors(
        "Summary: The viewer may be less interested in horror. There is not enough "
        "information yet to identify clear positive preferences.",
        recommender(),
        ["Private Horror"],
        evidence,
    )
    assert "participant_facing_abstention" in errors
    substituted_meta = pilot_api._summary_validation_errors(
        "Summary: The viewer may be less interested in horror. Other genre preferences "
        "are not clearly indicated; more information would allow better recommendations.",
        recommender(),
        ["Private Horror"],
        evidence,
    )
    assert "participant_facing_abstention" in substituted_meta


def test_online_validator_allows_neutral_fallback_only_without_any_signal() -> None:
    evidence = build_separated_evidence(
        {
            "user_id": 0,
            "history_hash": "neutral-only",
            "history_items": 1,
            "movie_ids": [1],
            "titles": ["Private Movie"],
            "ratings": [3],
            "genres": ["Drama"],
        }
    )
    assert pilot_api._summary_validation_errors(
        "Summary: The viewer can provide more preference information.",
        recommender(),
        ["Private Movie"],
        evidence,
    ) == []


def test_online_grounding_gate_retries_categorical_weak_dislike() -> None:
    evidence = build_separated_evidence(
        {
            "user_id": 0,
            "history_hash": "test",
            "history_items": 1,
            "movie_ids": [1],
            "titles": ["Private Horror"],
            "ratings": [1],
            "genres": ["Horror"],
        }
    )
    invalid = "Summary: The viewer dislikes horror."
    valid = "Summary: The viewer may be less interested in horror films."
    responses = FakeResponses([invalid, valid])

    summary, attempts = pilot_api._generate_valid_tears_summary(
        SimpleNamespace(responses=responses),
        ["separated evidence"],
        recommender(),
        ["Private Horror"],
        [],
        grounding_evidence=evidence,
    )

    assert summary == valid
    assert attempts[0]["validator_errors"] == [
        "grounding_negative_on_unsupported:Horror",
        "grounding_weak_broad_dislike:Horror",
    ]
    assert attempts[1]["validator_errors"] == []
    assert "grounding_negative_on_unsupported:Horror" in responses.calls[1]["input"][1][
        "content"
    ]


def test_online_grounding_retry_restores_missing_required_negative_genre() -> None:
    evidence = build_separated_evidence(
        {
            "user_id": 0,
            "history_hash": "required-negative",
            "history_items": 4,
            "movie_ids": [1, 2, 3, 4],
            "titles": ["Private A", "Private B", "Private C", "Private D"],
            "ratings": [1, 1, 2, 2],
            "genres": ["Horror|Thriller", "Horror|Thriller", "Horror", "Thriller"],
        }
    )
    missing = "Summary: The viewer dislikes horror films and frightening stories."
    valid = (
        "Summary: The viewer dislikes horror and thriller films, especially frightening "
        "stories built around sustained tension."
    )
    responses = FakeResponses([missing, valid])

    summary, attempts = pilot_api._generate_valid_tears_summary(
        SimpleNamespace(responses=responses),
        [pilot_api._render_online_inference_payload(evidence)],
        recommender(),
        ["Private A", "Private B", "Private C", "Private D"],
        [],
        grounding_evidence=evidence,
    )

    assert summary == valid
    assert attempts[0]["validator_errors"] == ["grounding_missing_negative:Thriller"]
    repair = responses.calls[1]["input"][1]["content"]
    assert "Restore natural negative coverage" in repair
    assert "every genre named by grounding_missing_negative" in repair
    assert attempts[1]["validator_errors"] == []


def test_task_1a_stops_after_the_bounded_validator_retry_limit() -> None:
    invalid = "Summary: The viewer rated Private Drama 5/5 and prefers it."
    responses = FakeResponses([invalid] * pilot_api.SUMMARY_MAX_ATTEMPTS)
    attempts: list[dict] = []

    with pytest.raises(HTTPException) as raised:
        pilot_api._generate_valid_tears_summary(
            SimpleNamespace(responses=responses),
            CASES["positive"]["history"],
            recommender(),
            ["Private Drama", "Private Comedy"],
            attempts,
        )

    assert raised.value.status_code == 422
    assert f"after {pilot_api.SUMMARY_MAX_ATTEMPTS} attempts" in raised.value.detail
    assert len(responses.calls) == pilot_api.SUMMARY_MAX_ATTEMPTS
    assert len(attempts) == pilot_api.SUMMARY_MAX_ATTEMPTS
    assert all(item["validator_errors"] for item in attempts)
