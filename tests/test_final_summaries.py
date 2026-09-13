from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tears_training import final_summaries_v4

from tears_training.final_summaries import (
    CALIBRATION_USERS,
    EMILIANO_PROMPT_SHA256,
    EXPECTED_EMILIANO_PROMPT_SHA256,
    MAX_OUTPUT_TOKENS,
    protocol_manifest,
    render_emiliano_history,
    response_request,
    select_calibration_users,
    validate_summary,
)


def test_exact_emiliano_prompt_and_history_bytes():
    assert EMILIANO_PROMPT_SHA256 == EXPECTED_EMILIANO_PROMPT_SHA256
    history = [(20, 2, "Newest (2000)", 4.0, "Drama|Mystery")]
    assert render_emiliano_history(history) == (
        "\nNewest (2000)\nRating: 4.0\n\\Genres: Drama|Mystery\n"
    )


def test_request_uses_exact_prompt_without_user_wrapper():
    protocol = protocol_manifest()
    history = [(20, 2, "Newest (2000)", 4.0, "Drama|Mystery")]
    request, record = response_request(9, "train", "20-49", history, protocol)
    body = request["body"]
    assert body["input"][0]["content"] == protocol["system_prompt"]
    assert body["input"][1]["content"].startswith("\nNewest (2000)")
    assert body["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["store"] is False
    assert record["user_id"] == 9


def test_representative_calibration_is_exact_and_deterministic():
    records = []
    user_id = 1
    for split, count in (("train", 1800), ("validation", 100), ("test", 100)):
        for index in range(count):
            records.append(
                {
                    "userId": user_id,
                    "interaction_count": 20 + index,
                    "activity_band": ["20-49", "50-99", "100-199", "200-499", "500+"][index % 5],
                    "split": split,
                }
            )
            user_id += 1
    splits = pd.DataFrame(records)
    matrix = pd.DataFrame(
        {
            "userId": splits.userId,
            "modelUserId": np.arange(len(splits)),
            "split": splits.split,
        }
    )
    first = select_calibration_users(splits, matrix)
    second = select_calibration_users(splits, matrix)
    assert first.userId.tolist() == second.userId.tolist()
    assert len(first) == CALIBRATION_USERS
    assert first.split.value_counts().to_dict() == {
        "train": 90,
        "validation": 5,
        "test": 5,
    }
    assert set(first.activity_band) == {
        "20-49",
        "50-99",
        "100-199",
        "200-499",
        "500+",
    }


def test_validation_preserves_categories_without_sentence_count_rule():
    record = {
        "titles": ["Private Film (1999)"],
        "genres": ["Drama|Mystery|Comedy"],
    }
    summary = (
        "Summary: The user enjoys drama and mystery and prefers careful pacing, layered stories, and complex character arcs. "
        "They appreciate thoughtful themes, moral conflict, and narratives with emotional depth and patient development. "
        "The user is less interested in broad comedy when humor displaces coherent plot development or nuanced characterization. "
        "They do not enjoy repetitive content, shallow conflict, or predictable endings, although some viewers may appreciate those accessible patterns. "
        + "Grounded storytelling and reflective tone remain especially appealing, with attention to human choices and meaningful consequences. "
        + "Their preferences favor cohesive atmosphere, sustained tension, and character-driven content over disconnected spectacle. "
        + "The profile supports both positive and negative preferences while leaving room for varied dramatic and mysterious narratives. "
        + "Subtle motivations, believable relationships, and well-earned resolutions add to the kind of immersive experience they value."
    )
    assert 120 <= len(summary.split()) <= 260
    assert validate_summary(summary, record) == []


def test_validation_rejects_private_leakage():
    record = {
        "titles": ["Private Film (1999)"],
        "genres": ["Drama|Mystery"],
    }
    summary = (
        "Summary: The user enjoys Private Film and rated it 5/5 in 1999. "
        + "They prefer drama stories and character themes. "
        + "They dislike shallow content. Other viewers may enjoy it. "
        + "preference " * 120
    )
    errors = validate_summary(summary, record)
    assert "year_leakage" in errors
    assert "rating_leakage" in errors
    assert any(error.startswith("title_leakage") for error in errors)


def _v4_record(ratings: list[float], genres: list[str]) -> dict:
    return {
        "user_id": 9,
        "history_hash": "frozen",
        "history_items": len(ratings),
        "ratings": ratings,
        "genres": genres,
    }


def test_v4_repeated_negative_requires_genre_specific_support():
    evidence = final_summaries_v4.extract_user_evidence(
        _v4_record(
            [1.0, 2.0, 2.5, 5.0, 4.5, 4.0],
            ["Horror", "Horror", "Horror", "Drama", "Drama", "Drama"],
        )
    )
    assert evidence["supported_negative_genres"] == ["Horror"]
    assert evidence["supported_positive_genres"] == ["Drama"]
    assert evidence["conflicting_genres"] == []


def test_v4_conflict_is_excluded_from_both_preference_lists():
    evidence = final_summaries_v4.extract_user_evidence(
        _v4_record(
            [1.0, 2.0, 2.5, 4.0, 4.5, 5.0],
            ["Comedy"] * 6,
        )
    )
    assert evidence["conflicting_genres"] == ["Comedy"]
    assert evidence["supported_negative_genres"] == []
    assert evidence["supported_positive_genres"] == []


def test_v4_model_payload_omits_titles_ratings_counts_and_thresholds():
    evidence = final_summaries_v4.extract_user_evidence(
        _v4_record([5.0, 4.5, 4.0], ["Drama", "Drama", "Drama"])
    )
    payload = json.dumps(evidence["verbalization_payload"])
    assert "5.0" not in payload
    assert "4.5" not in payload
    assert "occurrence" not in payload.lower()
    assert "threshold" not in payload.lower()
    assert "Drama" in payload
