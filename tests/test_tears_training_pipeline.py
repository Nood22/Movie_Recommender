from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import torch

from tears_training.config import ExperimentConfig
from tears_training.data import build_sparse_profile
from tears_training.evaluate import aggregate_metric_rows, ranking_metrics
from tears_training.models import RecVAE
from tears_training.orchestrate import select_survivors, tuning_grid
from tears_training.catalog_selection import select_catalog
from tears_training.report import aggregate_seed_metrics
from tears_training.summaries import (
    SummaryRequest,
    _write_plan,
    batch_body,
    enforce_cost_gate,
    projected_cost_usd,
    validate_responses,
    validate_text,
    write_jsonl,
)
from tears_training.summary_wandb import poll_metrics, quality_metrics
from tears_training.train import (
    _restore_rng_state,
    _rng_state,
    resolve_resume_path,
    wandb_tracking_identity,
)
from tears_training.wandb_replay import latest_schedule_metrics


def make_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        scratch_root=tmp_path,
        raw_data=tmp_path / "raw",
        output_root=tmp_path / "output",
        paid_backup_root=tmp_path / "paid",
    )


def test_resolve_resume_path_uses_same_run_checkpoint_automatically(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    automatic = run_dir / "resume.pt"
    automatic.write_bytes(b"checkpoint")

    assert resolve_resume_path(None, run_dir) == automatic


def test_resolve_resume_path_prefers_explicit_and_allows_clean_start(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    explicit = tmp_path / "explicit.pt"

    assert resolve_resume_path(explicit, run_dir) == explicit
    assert resolve_resume_path(None, run_dir) is None


def test_restore_rng_state_normalizes_serialized_byte_state_to_cpu():
    state = _rng_state()
    state["torch"] = state["torch"].numpy().astype(np.int64)

    _restore_rng_state(state)

    assert torch.get_rng_state().device.type == "cpu"
    assert torch.get_rng_state().dtype == torch.uint8


def strict_summary_text(*, leaked: str = "") -> str:
    parts = [
        "The viewer likes grounded drama and thoughtful mystery genres with careful pacing "
        + "preference " * 19,
        "They enjoy character growth, moral choices, human connection, and coherent plots "
        + "content " * 19,
        "The viewer dislikes unsupported spectacle and broad comedy when the evidence supports that conclusion "
        + "dislike " * 18,
        "They avoid shallow twists and repetitive conflict that some other viewers may enjoy "
        + "theme " * 18,
    ]
    if leaked:
        parts[0] += leaked
    return "Summary: " + ". ".join(part.strip() for part in parts) + "."


def test_summary_cache_key_covers_order_model_and_generation(tmp_path):
    first = SummaryRequest(
        7,
        (("A Movie", 5.0, "Drama"), ("B Movie", 2.0, "Comedy")),
        "v1",
        "model-a",
        450,
    )
    reordered = SummaryRequest(
        7,
        tuple(reversed(first.history)),
        "v1",
        "model-a",
        450,
    )
    changed_model = SummaryRequest(7, first.history, "v1", "model-b", 450)
    assert first.cache_key != reordered.cache_key
    assert first.cache_key != changed_model.cache_key
    body = batch_body(first, make_config(tmp_path))["body"]
    assert body["reasoning"] == {"effort": "minimal"}
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["store"] is False


def test_summary_validation_rejects_private_evidence_leakage(tmp_path):
    config = make_config(tmp_path)
    text = strict_summary_text()
    assert validate_text(text, config) == []
    leaked = strict_summary_text(leaked=" The Matrix was rated 5/5 in 1999")
    errors = validate_text(leaked, config, ["The Matrix (1999)"])
    assert "year_leakage" in errors
    assert "rating_leakage" in errors
    assert any(error.startswith("title_leakage") for error in errors)


def test_summary_validation_uses_whole_words_for_privacy_markers(tmp_path):
    config = make_config(tmp_path)
    text = strict_summary_text(
        leaked=" The viewer tolerated psychological character driven stories with strong direction"
    )
    assert validate_text(
        text, config, ["Psycho (1960)", "Tron (1982)", "Drive (2011)"]
    ) == []


def test_summary_validation_requires_emiliano_format(tmp_path):
    config = make_config(tmp_path)
    freeform = "The viewer likes drama. " + "preference " * 120
    errors = validate_text(freeform, config)
    assert "format_prefix" in errors


def test_cost_gate_includes_actual_spend_and_reserve(tmp_path):
    config = make_config(tmp_path)
    plan = {"estimated_input_tokens": 1_000_000, "reserved_output_tokens": 500_000}
    assert projected_cost_usd(plan, 0.10, 1.00) == pytest.approx(0.60)
    enforce_cost_gate(config, 100.0, 100.0)
    with pytest.raises(RuntimeError, match="above"):
        enforce_cost_gate(config, 120.0, 100.0)


def test_summary_response_is_validated_cached_and_mirrored(tmp_path):
    config = make_config(tmp_path)
    request = SummaryRequest(
        9,
        (("Private Film", 4.5, "Drama"),),
        "v1",
        config.summaries.model,
        config.summaries.max_output_tokens,
    )
    root = tmp_path / "summary-run"
    plan = _write_plan(config, root, [request])
    summary = strict_summary_text()
    response = {
        "custom_id": request.custom_id,
        "response": {
            "status_code": 200,
            "body": {"output_text": json.dumps({"summary": summary})},
        },
    }
    write_jsonl(root / "responses" / "batch.jsonl", [response])
    result = validate_responses(config, root / "request_plan.json")
    assert result["plan"] == plan["fingerprint"]
    assert result["expected"] == 1
    assert result["valid"] == 1
    assert result["invalid"] == 0
    assert result["validation_version"] == "tears-summary-privacy-format-v3-emiliano"
    assert len(result["validation_fingerprint"]) == 64
    assert (root / "cache" / f"{request.cache_key}.json").is_file()
    assert (
        config.paid_backup_root
        / plan["fingerprint"]
        / "validated"
        / result["validation_fingerprint"]
        / "summaries.jsonl"
    ).is_file()


def test_ranking_masks_observed_items_and_reports_denominators():
    logits = torch.tensor([[100.0, 9.0, 8.0], [4.0, 3.0, 2.0]])
    observed = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    metrics = ranking_metrics(logits, observed, target, ks=(1,))
    assert metrics["users"] == 2
    assert metrics["eligible_users"] == 1
    assert metrics["recall@1"] == 1.0
    combined = aggregate_metric_rows([metrics, metrics])
    assert combined["users"] == 4
    assert combined["eligible_users"] == 2


def test_recvae_composite_prior_forward_backward_and_prior_update():
    model = RecVAE(item_count=8, latent_dim=4, dropout=0.1, gamma=0.004)
    ratings = torch.tensor(
        [[5.0, 0, 4.0, 0, 0, 3.0, 0, 0], [0, 4.0, 0, 5.0, 0, 0, 2.0, 0]]
    )
    logits, latent = model(ratings)
    loss = model.loss(logits, ratings, latent)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.layers[0].weight.grad is not None
    model.update_prior()
    assert torch.equal(
        model.prior.old_encoder.layers[0].weight, model.encoder.layers[0].weight
    )


def test_sparse_profile_is_chronological_and_leak_free(tmp_path):
    config = make_config(tmp_path)
    config.raw_data.mkdir()
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 2, 2, 2],
            "movieId": [10, 11, 10, 11, 12, 13, 14],
            "rating": [5, 3, 5, 2, 4, 5, 1],
            "timestamp": [1, 2, 1, 2, 3, 4, 5],
        }
    )
    ratings.to_csv(config.raw_data / "ratings.csv", index=False)
    catalog = pd.DataFrame(
        {
            "movieId": [10, 11, 12, 13, 14],
            "title": ["A", "B", "C", "D", "E"],
            "genres": ["Drama"] * 5,
            "modelItemId": range(5),
        }
    )
    catalog_path = tmp_path / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)
    splits = pd.DataFrame(
        {
            "userId": [1, 2],
            "interaction_count": [2, 5],
            "activity_band": ["20-49", "20-49"],
            "split": ["train", "validation"],
        }
    )
    output = tmp_path / "matrix"
    manifest = build_sparse_profile(config, splits, catalog_path, output)
    observed = sparse.load_npz(output / "validation_observed.npz")
    target = sparse.load_npz(output / "validation_target.npz")
    # floor(5 * .7) = 3: movie 13 is the only positive held-out target.
    assert observed[1].indices.tolist() == [0, 1, 2]
    assert target[1].indices.tolist() == [3]
    assert observed.multiply(target).nnz == 0
    assert manifest["eligible_evaluation_users"]["validation"] == 1
    assert (output / "validation_observed_csr" / "data.npy").is_file()


def test_tuning_grid_and_successive_halving(tmp_path):
    assert len(tuning_grid("recvae")) == 54
    assert len(tuning_grid("tears_base")) == 6
    assert len(tuning_grid("gers_recvae")) == 18
    path = tmp_path / "results.jsonl"
    rows = [
        {
            "model": "recvae",
            "validation_selection_ndcg@50": score,
            "gpu_hours": hours,
            "hyperparameters": {"dropout": dropout},
        }
        for score, hours, dropout in ((0.10, 4, 0.1), (0.20, 8, 0.2), (0.199, 2, 0.4))
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    survivors = select_survivors(path, ("recvae",), final=False)
    assert survivors["recvae"] == [{"dropout": 0.2}]
    finalist = select_survivors(path, ("recvae",), final=True)
    assert finalist["recvae"] == [{"dropout": 0.4}]


def test_catalog_selection_prefers_largest_near_best_under_memory_gate():
    rows = [
        {"support": 20, "validation_ndcg@20": 0.195, "peak_gpu_memory_bytes": 70 * 1024**3, "run_dir": "r20", "checkpoint": "c20", "run_fingerprint": "f20"},
        {"support": 50, "validation_ndcg@20": 0.200, "peak_gpu_memory_bytes": 60 * 1024**3, "run_dir": "r50", "checkpoint": "c50", "run_fingerprint": "f50"},
        {"support": 200, "validation_ndcg@20": 0.205, "peak_gpu_memory_bytes": 40 * 1024**3, "run_dir": "r200", "checkpoint": "c200", "run_fingerprint": "f200"},
    ]
    assert select_catalog(rows)["selected_support"] == 20


def test_catalog_selection_rejects_candidate_at_memory_limit():
    rows = [
        {"support": 20, "validation_ndcg@20": 0.3, "peak_gpu_memory_bytes": 72 * 1024**3, "run_dir": "r20", "checkpoint": "c20", "run_fingerprint": "f20"},
        {"support": 50, "validation_ndcg@20": 0.2, "peak_gpu_memory_bytes": 60 * 1024**3, "run_dir": "r50", "checkpoint": "c50", "run_fingerprint": "f50"},
    ]
    assert select_catalog(rows)["selected_support"] == 50


def test_wandb_tracking_identity_separates_training_schedules():
    base = {
        "profile": "smoke",
        "minimum_epochs": 5,
        "patience": 20,
    }
    five_payload, five_id = wandb_tracking_identity(
        "training-fingerprint", argparse.Namespace(**base, epochs=5)
    )
    sixty_payload, sixty_id = wandb_tracking_identity(
        "training-fingerprint",
        argparse.Namespace(**(base | {"minimum_epochs": 30}), epochs=60),
    )
    assert five_payload["training_fingerprint"] == sixty_payload["training_fingerprint"]
    assert five_id != sixty_id


def test_wandb_tracking_identity_separates_retry_attempts():
    common = {
        "profile": "smoke",
        "epochs": 300,
        "minimum_epochs": 30,
        "patience": 20,
    }
    first_payload, first_id = wandb_tracking_identity(
        "training-fingerprint", argparse.Namespace(**common, tracking_attempt=0)
    )
    retry_payload, retry_id = wandb_tracking_identity(
        "training-fingerprint", argparse.Namespace(**common, tracking_attempt=1)
    )
    assert first_payload["attempt"] == 0
    assert retry_payload["attempt"] == 1
    assert first_id != retry_id


def test_summary_wandb_metrics_cover_progress_cost_and_quality():
    submission = {"submitted_at": 100, "projected_cost_usd": 12.0}
    poll = {
        "polled_at": 160,
        "actual_cost_usd": 4.0,
        "cumulative_cost_usd": 4.08,
        "usage": {"input_tokens": 1000, "output_tokens": 200},
        "batches": [
            {
                "status": "in_progress",
                "created_at": 100,
                "request_counts": {"total": 10000, "completed": 2500, "failed": 2},
            }
        ],
    }
    metrics = poll_metrics(poll, submission)
    assert metrics["batch/completion_percent"] == 25.0
    assert metrics["batch/failed_requests"] == 2
    assert metrics["time/elapsed_seconds"] == 60
    assert metrics["usage/input_tokens"] == 1000
    assert metrics["cost/actual_usd"] == 4.0
    quality = quality_metrics(
        {
            "validation": {"missing_user_ids": [3], "duplicate_summary_groups": 2},
            "summary_lengths_words": {"min": 120, "max": 200, "mean": 155},
            "cost_usd": {"per_planned_user": 0.001},
        }
    )
    assert quality["quality/missing_user_ids"] == 1
    assert quality["quality/duplicate_summary_groups"] == 2
    assert quality["quality/summary_length_mean"] == 155


def test_wandb_replay_uses_only_latest_contiguous_schedule():
    rows = [
        {"epoch": epoch, "value": f"old-{epoch}"} for epoch in range(5)
    ] + [
        {"epoch": epoch, "value": f"new-{epoch}"} for epoch in range(3)
    ]
    assert latest_schedule_metrics(rows) == [
        {"epoch": epoch, "value": f"new-{epoch}"} for epoch in range(3)
    ]


def test_report_aggregates_seeds_and_runs_paired_tests():
    rows = []
    for seed, base, gain in (
        (2020, 0.10, 0.019),
        (2021, 0.11, 0.020),
        (2022, 0.12, 0.021),
    ):
        rows.append({"model": "recvae", "seed": seed, "ndcg@50": base})
        rows.append(
            {"model": "tears_recvae", "seed": seed, "ndcg@50": base + gain}
        )
    report = aggregate_seed_metrics(rows)
    assert report["summary"]["tears_recvae"]["ndcg@50"]["n"] == 3
    paired = report["paired_tests_vs_baseline"]["tears_recvae"]["ndcg@50"]
    assert paired["mean_paired_difference"] == pytest.approx(0.02)
