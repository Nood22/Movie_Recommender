from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd

from tears_training import production_calibration as production


def _synthetic_users() -> pd.DataFrame:
    rows = []
    user_id = 1
    for split, split_size in (("train", 1_500), ("validation", 450), ("test", 450)):
        for offset in range(split_size):
            rows.append(
                {
                    "userId": user_id,
                    "interaction_count": 20 + offset % 500,
                    "activity_band": ("20-49", "50-99", "100-199", "200+")[
                        offset % 4
                    ],
                    "split": split,
                }
            )
            user_id += 1
    return pd.DataFrame(rows)


def test_sample_is_exact_deterministic_stratified_and_disjoint() -> None:
    users = _synthetic_users()
    matrix_users = users[["userId"]].copy()
    excluded = {3, 19, 500, 1700, 2200}
    first = production.select_production_calibration_users(
        users, matrix_users, excluded
    )
    second = production.select_production_calibration_users(
        users, matrix_users, excluded
    )
    assert len(first) == 1_000
    assert first.userId.nunique() == 1_000
    assert not (set(first.userId) & excluded)
    pd.testing.assert_frame_equal(first, second)
    assert set(first.split) == {"train", "validation", "test"}
    assert set(first.activity_band) == {"20-49", "50-99", "100-199", "200+"}


def test_genericity_preflight_passes_and_records_source_hash() -> None:
    report = production._genericity_checks()
    assert report["pass"]
    assert report["repair_source_sha256"]
    assert all(report["checks"].values())


def test_generation_protocol_is_byte_identical_to_frozen_v2() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    protocol = production._production_protocol(repository_root)
    generation = protocol["generation"]
    assert generation["generation"]["model"] == "gpt-5-mini-2025-08-07"
    assert generation["system_prompt_sha256"] == production.v2.V2_PROMPT_SHA256
    assert generation["system_prompt"] == production.v2.V2_PROMPT
    assert not protocol["post_processing"]["tmdb_semantic_metadata"]
    assert not protocol["post_processing"]["manual_labels_at_inference"]
    assert not protocol["post_processing"]["user_specific_exceptions"]


def test_old_100_is_diagnostic_only_and_does_not_load_manual_labels() -> None:
    report = production._offline_baseline_diagnostic()
    assert report["users"] == 100
    assert not report["manual_labels_loaded"]
    assert report["positive_sentences_preserved"] == 100
    assert len(report["per_user"]) == 100


def test_orchestrator_exposes_no_full_cohort_or_training_action() -> None:
    parser = production.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if getattr(action, "choices", None)
    )
    assert set(subparsers.choices) == {"plan", "submit", "poll", "validate", "finalize"}
    source = inspect.getsource(production.submit_calibration)
    assert "CALIBRATION_USERS" in source
    assert "submitted_requests" in source


def test_repair_inference_has_no_user_id_parameter() -> None:
    assert tuple(inspect.signature(production.repair_summary).parameters) == (
        "summary",
        "evidence",
    )
