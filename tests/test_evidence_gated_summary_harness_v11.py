from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

from tears_training import evidence_gated_summary_harness_v11 as harness
from tears_training.evidence_gated_calibration_v11 import _retry_reasons


CALIBRATION = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/"
    "production_calibration_1000/v002_20260816_v2_generic_repair"
)
V10 = CALIBRATION.parent / "v010_20260816_evidence_gated_emiliano"


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


def status(ratings: list[float]) -> str:
    evidence = harness.build_separated_evidence(record(ratings, ["Action"] * len(ratings)))
    return evidence["genre_statistics"]["Action"]["status"]


def test_dominance_rule_classifies_conceptual_cases() -> None:
    assert status([5.0] * 8 + [1.0] * 2) == "POSITIVE"
    assert status([5.0] * 2 + [1.0] * 8) == "NEGATIVE"
    assert status([5.0] * 4 + [1.0] * 3) == "MIXED"
    assert status([5.0] * 2 + [1.0] * 2) == "INSUFFICIENT"


def test_every_observed_genre_has_exactly_one_status_and_reason() -> None:
    evidence = harness.build_separated_evidence(
        record([5.0, 4.0, 1.0, 3.0], ["Action|Drama", "Drama", "Action", "Comedy"])
    )
    partitions = evidence["genre_classifications"]
    flattened = [genre for values in partitions.values() for genre in values]
    assert sorted(flattened) == ["Action", "Comedy", "Drama"]
    assert len(flattened) == len(set(flattened))
    for values in evidence["genre_statistics"].values():
        assert values["status"] in {"POSITIVE", "NEGATIVE", "MIXED", "INSUFFICIENT"}
        assert values["classification_reason"]
        assert "positive_strength" in values
        assert "negative_strength" in values


def test_core_positive_reuses_two_to_one_relative_materiality() -> None:
    evidence = harness.build_separated_evidence(
        record(
            [5.0] * 8 + [5.0] * 4 + [5.0] * 3,
            ["Drama"] * 8 + ["Comedy"] * 4 + ["Mystery"] * 3,
        )
    )
    payload = evidence["inference_payload"]
    assert payload["core_positive_genres"] == ["Comedy", "Drama"]
    assert payload["secondary_positive_genres"] == ["Mystery"]


def test_prompt_exposes_complete_exclusive_contract() -> None:
    evidence = harness.build_separated_evidence(
        record([5.0] * 4 + [1.0] * 3, ["Action"] * 7)
    )
    prompt = harness.render_evidence_prompt(evidence)
    for label in (
        "SUPPORTED POSITIVE GENRES:",
        "CORE POSITIVE GENRES (required coverage):",
        "SUPPORTED NEGATIVE GENRES:",
        "REQUIRED NEGATIVE GENRES (complete coverage required):",
        "MIXED GENRES:",
        "INSUFFICIENT GENRES:",
    ):
        assert label in prompt
    assert "categorical positive genre claim may use only" in harness.EVIDENCE_GATED_PROMPT
    assert "Every genre in REQUIRED NEGATIVE GENRES" in harness.EVIDENCE_GATED_PROMPT


def test_coverage_validator_accepts_complete_natural_prose() -> None:
    evidence = harness.build_separated_evidence(
        record(
            [5.0] * 6 + [1.0] * 6,
            ["Drama"] * 6 + ["Action"] * 6,
        )
    )
    result = harness.validate_summary_contract(
        "Summary: The user strongly prefers character-driven drama. "
        "They dislike action films and high-octane action spectacle.",
        evidence,
    )
    assert result.pass_contract
    assert result.missing_core_positive_genres == ()
    assert result.missing_required_negative_genres == ()


def test_coverage_validator_rejects_omission_and_wrong_polarity() -> None:
    evidence = harness.build_separated_evidence(
        record(
            [5.0] * 6 + [1.0] * 6,
            ["Drama"] * 6 + ["Action"] * 6,
        )
    )
    omitted = harness.validate_summary_contract(
        "Summary: The user strongly prefers character-driven drama. "
        "They dislike spectacle-heavy films.",
        evidence,
    )
    assert omitted.missing_required_negative_genres == ("Action",)
    wrong = harness.validate_summary_contract(
        "Summary: The user enjoys drama and action. They dislike action films.",
        evidence,
    )
    assert wrong.categorical_positive_on_negative_genres == ("Action",)


def test_mixed_genre_requires_explicit_uncertainty() -> None:
    evidence = harness.build_separated_evidence(
        record([5.0] * 4 + [1.0] * 3, ["Action"] * 7)
    )
    bad = harness.validate_summary_contract(
        "Summary: The user likes Action films. No clear negative preference is evident.",
        evidence,
    )
    assert bad.categorical_positive_on_mixed_genres == ("Action",)
    good = harness.validate_summary_contract(
        "Summary: The user has selective, mixed preferences within Action. "
        "No clear negative preference is evident.",
        evidence,
    )
    assert good.categorical_positive_on_mixed_genres == ()


def test_validator_distinguishes_theme_words_from_broad_genre_claims() -> None:
    evidence = harness.build_separated_evidence(
        record(
            [5.0] * 6 + [3.0, 3.0],
            ["Drama"] * 6 + ["War", "Romance"],
        )
    )
    result = harness.validate_summary_contract(
        "Summary: The user prefers character-driven drama with occasional wartime "
        "conflict and moments of romantic subtext. No clear dislike is supported.",
        evidence,
    )
    assert result.pass_contract
    assert result.categorical_positive_on_insufficient_genres == ()


def test_validator_accepts_natural_evidence_aware_wording_and_abstention() -> None:
    evidence = harness.build_separated_evidence(
        record([5.0] * 4, ["Drama"] * 4)
    )
    result = harness.validate_summary_contract(
        "Summary: The user has a clear taste for drama. There is no clear evidence "
        "of specific genres or plot elements the user dislikes in the available history.",
        evidence,
    )
    assert result.pass_contract
    assert not result.internal_contract_label_leak
    assert not result.none_fabricated_dislike


def test_validator_splits_contrast_before_assigning_genre_polarity() -> None:
    evidence = harness.build_separated_evidence(
        record([5.0] * 6 + [1.0] * 6, ["Drama"] * 6 + ["Sci-Fi"] * 6)
    )
    result = harness.validate_summary_contract(
        "Summary: The user prefers drama. They dislike science fiction when it "
        "dominates rather than character drama.",
        evidence,
    )
    assert result.pass_contract
    assert result.categorical_negative_on_positive_genres == ()


def test_validator_accepts_welcome_as_positive_coverage() -> None:
    evidence = harness.build_separated_evidence(
        record([5.0] * 6, ["Romance"] * 6)
    )
    result = harness.validate_summary_contract(
        "Summary: Romantic elements are welcome when paired with nuanced stories. "
        "The history does not establish any clear negative preference.",
        evidence,
    )
    assert result.pass_contract


def test_weak_abstention_does_not_assign_all_named_uncertain_genres_negative() -> None:
    evidence = harness.build_separated_evidence(
        record([5.0] * 4 + [2.0], ["Drama"] * 4 + ["Comedy"])
    )
    result = harness.validate_summary_contract(
        "Summary: The user prefers drama. The history does not reliably indicate "
        "dislikes for Animation, Western, or Musical films.",
        evidence,
    )
    assert result.pass_contract
    assert result.weak_broad_unsupported_genres == ()


def test_request_preserves_v10_model_settings_and_changes_retry_identity_only() -> None:
    item = record([5.0, 4.0, 4.5])
    evidence = harness.build_separated_evidence(item)
    first = harness.build_response_request(item, evidence)
    retry = harness.build_response_request(item, evidence, retry_attempt=1)
    assert first["custom_id"] != retry["custom_id"]
    assert first["body"] == retry["body"]
    body = first["body"]
    assert body["model"] == "gpt-5-mini-2025-08-07"
    assert body["reasoning"] == {"effort": "minimal"}
    assert body["text"]["verbosity"] == "low"
    assert body["max_output_tokens"] == 450
    assert body["store"] is False


def test_retry_policy_is_coverage_targeted_not_semantic_repair() -> None:
    audit_only = {
        "categorical_positive_on_mixed_genres": ["Action"],
        "missing_required_negative_genres": [],
        "missing_core_positive_genres": [],
    }
    assert _retry_reasons(["genre_contract_failure"], audit_only) == []
    missing_negative = dict(audit_only, missing_required_negative_genres=["Comedy"])
    assert _retry_reasons(["genre_contract_failure"], missing_negative) == [
        "missing_required_negative_genres"
    ]
    missing_positive = dict(audit_only, missing_core_positive_genres=["Drama"])
    assert _retry_reasons(["genre_contract_failure"], missing_positive) == [
        "missing_core_positive_genres"
    ]
    assert _retry_reasons(["invalid_structured_output"], {}) == [
        "invalid_structured_output"
    ]


def test_previous_positive_and_mixed_failures_are_deterministically_partitioned() -> None:
    plan = json.loads((CALIBRATION / "request_plan.json").read_text())
    records = {int(row["user_id"]): row for row in plan["records"]}
    positive_failures = {
        5282, 8742, 13279, 24796, 26328, 31715, 64863, 83310,
        98331, 111189, 114176, 124152, 157895, 176397, 198747,
    }
    mixed_failures = {5282, 26328, 64863, 83310, 111189, 198747}
    for user_id in positive_failures:
        evidence = harness.build_separated_evidence(records[user_id])
        partitions = evidence["genre_classifications"]
        assert sum(len(values) for values in partitions.values()) == len(
            evidence["genre_statistics"]
        )
    for user_id in mixed_failures:
        evidence = harness.build_separated_evidence(records[user_id])
        old = {
            row["user_id"]: row
            for row in map(
                json.loads,
                (V10 / "evidence/all_user_evidence.jsonl").read_text().splitlines(),
            )
        }[user_id]
        for genre in old["genre_classifications"]["mixed_conflicting"]:
            new = evidence["genre_statistics"][genre]
            assert new["status"] in {"POSITIVE", "NEGATIVE", "MIXED"}
            if new["status"] != "MIXED":
                assert "two_to_one_dominance" in new["classification_reason"]
        prompt = harness.render_evidence_prompt(evidence)
        assert "MIXED GENRES:" in prompt


def test_previous_negative_omissions_are_required_coverage() -> None:
    plan = json.loads((CALIBRATION / "request_plan.json").read_text())
    records = {int(row["user_id"]): row for row in plan["records"]}
    expected = {
        13279: {"Fantasy", "Mystery"},
        83310: {"Comedy"},
        124194: {"Romance"},
        189979: {"Mystery"},
    }
    for user_id, genres in expected.items():
        evidence = harness.build_separated_evidence(records[user_id])
        assert genres <= set(evidence["required_negative_genres"])
        assert genres <= set(evidence["inference_payload"]["required_negative_genres"])


def test_previous_title_leaks_are_still_sanitized_without_semantic_repair() -> None:
    plan = json.loads((CALIBRATION / "request_plan.json").read_text())
    records = {int(row["user_id"]): row for row in plan["records"]}
    v10_rows = {
        int(row["user_id"]): row
        for row in map(
            json.loads,
            (V10 / "validated/all_summaries.jsonl").read_text().splitlines(),
        )
    }
    for user_id in {6941, 28849, 121300, 129133}:
        result = harness.privacy_and_format_cleanup(
            v10_rows[user_id]["raw_summary"],
            records[user_id]["titles"],
        )
        assert result.title_matches_after == ()
        assert result.final_summary == v10_rows[user_id]["final_summary"]
        assert all(
            operation["type"].startswith(("privacy_", "format_"))
            for operation in result.operations
        )


def test_previous_contradiction_cases_have_disjoint_deterministic_statuses() -> None:
    plan = json.loads((CALIBRATION / "request_plan.json").read_text())
    records = {int(row["user_id"]): row for row in plan["records"]}
    for user_id in {8742, 18515, 50674, 60875, 65242, 83310, 92810, 159382, 166368, 199226}:
        evidence = harness.build_separated_evidence(records[user_id])
        partitions = evidence["genre_classifications"]
        assert not (set(partitions["POSITIVE"]) & set(partitions["NEGATIVE"]))
        assert not (set(partitions["MIXED"]) & (set(partitions["POSITIVE"]) | set(partitions["NEGATIVE"])))


def test_harness_is_generic_and_has_no_semantic_repair_or_user_ids() -> None:
    source = inspect.getsource(harness)
    for forbidden in (
        "repair_summary(",
        "compose_final_summary(",
        "manual_label",
        "adjudication_note",
        "REPAIR_SPECS",
    ):
        assert forbidden.lower() not in source.lower()
    tree = ast.parse(source)
    integer_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    }
    plan = json.loads((CALIBRATION / "request_plan.json").read_text())
    assert not (integer_literals & set(plan["sample"]["user_ids"]))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and node.keys:
            assert not all(
                isinstance(key, ast.Constant)
                and isinstance(key.value, int)
                and not isinstance(key.value, bool)
                for key in node.keys
                if key is not None
            )
