from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import pilot_api
from tears_training.summaries import validate_text


SAFE_SUFFIX = (
    ", with the available evidence supporting only this broad statement and not "
    "implying any additional genre theme plot element or viewing need beyond the "
    "directly observed information supplied for this profile and its stated level of support"
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
            "No strong disliked genre or style is supported",
            "No strong disliked plot point or content preference is supported",
        ],
        "expected": ("prefers drama and comedy", "no strong disliked genre"),
    },
    "negative": {
        "history": [
            "- title='Private Horror'; private_rating=1/5; genres=Horror",
            "- title='Private Thriller'; private_rating=2/5; genres=Thriller",
            "- explicitly_disliked=Private Horror",
        ],
        "parts": [
            "No strong liked genre is supported for the viewer",
            "No strong liked theme or content preference is supported",
            "The viewer dislikes horror",
            "They avoid frightening and graphic content",
        ],
        "expected": ("no strong liked genre", "dislikes horror"),
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
            "No strong disliked genre or style is supported",
            "No strong disliked plot point or content preference is supported",
        ],
        "expected": ("prefers comedy", "when unwinding"),
    },
    "sparse": {
        "history": [
            "- title='Private Action'; private_rating=2/5; genres=Action|Adventure|Drama",
        ],
        "parts": [
            "No strong liked genre is supported for the viewer",
            "No strong liked theme or content preference is supported",
            "The viewer dislikes action adventure and drama",
            "No strong disliked plot point or content preference is supported",
        ],
        "expected": ("no strong liked genre", "dislikes action adventure and drama"),
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
    assert validate_text(valid, recommender().config, ["Private Movie"]) == []
    invalid = "Summary: The viewer has a brief supported preference."
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
    assert 120 <= recorded[-1]["word_count"] <= 260
    assert all(call["text"]["verbosity"] == "high" for call in responses.calls)
    assert all(call["max_output_tokens"] == 600 for call in responses.calls)
    assert all(
        "Return a 140-180 word TEARS profile" in call["input"][1]["content"]
        and
        "roughly 35-45 words in each sentence on the first attempt"
        in call["input"][1]["content"]
        and "preferably 36-40 words per sentence" in call["input"][1]["content"]
        and "values at or below 2 support dislikes" in call["input"][1]["content"]
        and "values at or above 4 support likes" in call["input"][1]["content"]
        and "middle values are neutral or inconclusive" in call["input"][1]["content"]
        for call in responses.calls
    )
    assert all(
        "express the underlying preference semantically" in call["input"][1]["content"]
        and "Private-rating semantic conversion is mandatory" in call["input"][1][
            "content"
        ]
        and "must not contain the words rating, ratings, rated, star" in call["input"][1][
            "content"
        ]
        and "A dislike does not imply liking its opposite" in call["input"][1]["content"]
        for call in responses.calls
    )
    for call in responses.calls[1:]:
        repair_prompt = call["input"][1]["content"]
        assert "original private evidence" in repair_prompt
        assert "do not add genres, themes, plot elements" in repair_prompt
        assert "word_count" in repair_prompt
        assert "requires a complete rewrite" in repair_prompt
        assert "totaling 140-180 words" in repair_prompt
        assert "grounded uncertainty or abstention language" in repair_prompt
        assert "aim near 152 words" in repair_prompt
        assert "do not infer that disliking one element means liking its opposite" in repair_prompt
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
    assert "Do not merely delete the preference claim" in repair_prompt
    assert "not mention the evidence measurement" in repair_prompt
    assert all(line in repair_prompt for line in case["history"])
    assert validate_text(summary, recommender().config, ["Private Drama"]) == []


def test_frozen_valid_output_outside_generation_target_gets_full_rewrite() -> None:
    short_sentences = []
    for index in range(4):
        prefix = "Summary: " if index == 0 else ""
        filler = " ".join(["supported"] * 27)
        short_sentences.append(f"{prefix}The viewer has {filler} preferences.")
    frozen_valid_but_short = " ".join(short_sentences)
    assert 120 <= len(frozen_valid_but_short.split()) < 140
    assert validate_text(
        frozen_valid_but_short, recommender().config, ["Private Movie"]
    ) == []
    valid = semantic_summary(CASES["positive"]["parts"])
    responses = FakeResponses([frozen_valid_but_short, valid])

    summary, attempts = pilot_api._generate_valid_tears_summary(
        SimpleNamespace(responses=responses),
        CASES["positive"]["history"],
        recommender(),
        ["Private Drama", "Private Comedy"],
        [],
    )

    assert summary == valid
    assert attempts[0]["validator_errors"] == []
    assert attempts[0]["generation_errors"] == [
        f"generation_word_target:{len(frozen_valid_but_short.split())}"
    ]
    assert attempts[1]["retry_errors"] == []
    retry_prompt = responses.calls[1]["input"][1]["content"]
    assert "requires a complete rewrite" in retry_prompt
    assert "totaling 140-180 words" in retry_prompt


def test_task_1a_stops_after_the_bounded_validator_retry_limit() -> None:
    invalid = "Summary: The viewer has a brief supported preference."
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
