"""Compare the current and legacy TEARS checkpoints on Emiliano's ML-1M split.

The evaluation uses the intersection of both checkpoint catalogs, supplies the
same observed ratings and GPT-4 summaries to both models, selects alpha on the
provided validation users, writes that selection, and only then evaluates the
provided test users.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pilot_recommender import (
    MATRIX_DIR,
    RECVAE_CHECKPOINT,
    TEARS_RUN_DIR,
    _checkpoint_payload,
)
from tears_inference_adapter import TEARSInferenceAdapter
from tears_training.artifacts import atomic_write_json, sha256_file
from tears_training.config import load_config
from tears_training.models import HybridVAE
from tears_training.train import make_model


EMILIANO_ROOT = (
    PROJECT_ROOT / "TEARS_Project" / "Code4Neda" / "data_preprocessed" / "ml-1m"
)
SUMMARY_PATH = (
    PROJECT_ROOT
    / "TEARS_Project"
    / "Code4Neda"
    / "saved_user_summary"
    / "ml-1m"
    / "user_summary_gpt-4-1106-preview_.json"
)
CURRENT_CHECKPOINT = TEARS_RUN_DIR / "best.pt"
EMILIANO_CHECKPOINT = (
    PROJECT_ROOT
    / "TEARS_Project"
    / "saved_model"
    / "ml-1m"
    / "ot_train_vae_ml-1m_embedding_module_OTRecVAE_2024-09-24_13-37-29_2022.csv.pt"
)
ALPHAS = tuple(round(index * 0.025, 3) for index in range(41))


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_contract() -> dict[str, object]:
    old_movie_to_item = {
        int(movie): int(item)
        for movie, item in _load_pickle(EMILIANO_ROOT / "show2id.pkl").items()
    }
    profile_to_user = {
        int(profile): int(user)
        for profile, user in _load_pickle(EMILIANO_ROOT / "profile2id.pkl").items()
    }
    user_to_profile = {user: profile for profile, user in profile_to_user.items()}
    summaries_by_profile = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    catalog = pd.read_csv(MATRIX_DIR / "catalog.csv").sort_values("modelItemId")
    current_movie_to_item = dict(
        zip(catalog.movieId.astype(int), catalog.modelItemId.astype(int))
    )
    shared_movies = sorted(set(old_movie_to_item) & set(current_movie_to_item))
    shared_old_items = np.asarray(
        [old_movie_to_item[movie] for movie in shared_movies], dtype=np.int64
    )
    shared_current_items = np.asarray(
        [current_movie_to_item[movie] for movie in shared_movies], dtype=np.int64
    )
    old_item_to_shared = {
        int(old_item): shared for shared, old_item in enumerate(shared_old_items)
    }
    summaries_by_user = {
        user: summaries_by_profile[str(float(profile))]
        for user, profile in user_to_profile.items()
    }
    return {
        "catalog": catalog,
        "old_movie_to_item": old_movie_to_item,
        "current_movie_to_item": current_movie_to_item,
        "shared_movies": shared_movies,
        "shared_old_items": shared_old_items,
        "shared_current_items": shared_current_items,
        "old_item_to_shared": old_item_to_shared,
        "summaries_by_user": summaries_by_user,
    }


def load_split(split: str, contract: dict[str, object]) -> dict[str, object]:
    observed_frame = pd.read_csv(EMILIANO_ROOT / f"{split}_tr.csv")
    target_frame = pd.read_csv(EMILIANO_ROOT / f"{split}_te.csv")
    users = np.asarray(sorted(observed_frame.uid.astype(int).unique()), dtype=np.int64)
    user_to_row = {int(user): row for row, user in enumerate(users)}
    item_to_shared = contract["old_item_to_shared"]
    assert isinstance(item_to_shared, dict)
    width = len(contract["shared_movies"])
    observed = np.zeros((len(users), width), dtype=np.float32)
    target = np.zeros((len(users), width), dtype=np.float32)
    dropped_observed = 0
    dropped_targets = 0
    for row in observed_frame.itertuples(index=False):
        shared = item_to_shared.get(int(row.sid))
        if shared is None:
            dropped_observed += 1
        else:
            observed[user_to_row[int(row.uid)], shared] = float(row.rating)
    for row in target_frame.itertuples(index=False):
        shared = item_to_shared.get(int(row.sid))
        if shared is None:
            dropped_targets += 1
        elif float(row.rating) >= 4.0:
            target[user_to_row[int(row.uid)], shared] = 1.0
    summaries_by_user = contract["summaries_by_user"]
    assert isinstance(summaries_by_user, dict)
    summaries = [str(summaries_by_user[int(user)]) for user in users]
    return {
        "users": users,
        "observed": observed,
        "target": target,
        "summaries": summaries,
        "shared_observed_interactions": int(np.count_nonzero(observed)),
        "shared_positive_targets": int(np.count_nonzero(target)),
        "dropped_observed_rows": dropped_observed,
        "dropped_target_rows": dropped_targets,
    }


def metric_batch(
    candidate_logits: torch.Tensor,
    observed: torch.Tensor,
    target: torch.Tensor,
    ks: tuple[int, ...] = (20, 50),
    collect_details: bool = False,
) -> dict[str, object]:
    scores = candidate_logits.clone()
    scores[observed > 0] = -torch.inf
    maximum = min(max(ks), scores.shape[1])
    ranked = torch.topk(scores, maximum, dim=1).indices
    truth = target > 0
    relevant = truth.sum(1)
    eligible = relevant > 0
    result: dict[str, object] = {
        "users": float(len(scores)),
        "eligible_users": float(eligible.sum()),
    }
    per_user = [
        {
            "eligible": bool(eligible[index]),
            "relevant_items": int(relevant[index]),
            "recommendations": ranked[index].detach().cpu().tolist(),
        }
        for index in range(len(scores))
    ] if collect_details else []
    for k in ks:
        width = min(k, maximum)
        hits = truth.gather(1, ranked[:, :width]).float()[eligible]
        counts = relevant[eligible].float()
        # Match the recall convention in Emiliano's supplied eval_metrics.py.
        recall = hits.sum(1) / counts.clamp_max(width).clamp_min(1)
        discounts = 1.0 / torch.log2(
            torch.arange(width, dtype=torch.float32, device=scores.device) + 2
        )
        dcg = (hits * discounts).sum(1)
        ideal_lengths = counts.clamp_max(width).long()
        idcg = torch.stack(
            [discounts[: int(length)].sum() for length in ideal_lengths]
        )
        ndcg = dcg / idcg.clamp_min(1e-12)
        precision = hits.sum(1) / float(width)
        hit_rate = (hits.sum(1) > 0).float()
        reciprocal_rank = torch.where(
            hits.bool().any(1),
            1.0 / (hits.bool().float().argmax(1).float() + 1.0),
            torch.zeros(len(hits), device=hits.device),
        )
        cumulative_precision = hits.cumsum(1) / torch.arange(
            1, width + 1, dtype=torch.float32, device=hits.device
        )
        average_precision = (cumulative_precision * hits).sum(1) / counts.clamp_max(
            width
        ).clamp_min(1)
        result[f"recall@{k}_sum"] = float(recall.sum())
        result[f"ndcg@{k}_sum"] = float(ndcg.sum())
        result[f"precision@{k}_sum"] = float(precision.sum())
        result[f"hit_rate@{k}_sum"] = float(hit_rate.sum())
        result[f"mrr@{k}_sum"] = float(reciprocal_rank.sum())
        result[f"map@{k}_sum"] = float(average_precision.sum())
        if collect_details:
            eligible_indices = eligible.nonzero(as_tuple=False).flatten().tolist()
            arrays = {
                f"recall@{k}": recall,
                f"ndcg@{k}": ndcg,
                f"precision@{k}": precision,
                f"hit_rate@{k}": hit_rate,
                f"mrr@{k}": reciprocal_rank,
                f"map@{k}": average_precision,
            }
            for metric, values in arrays.items():
                for local_index, user_index in enumerate(eligible_indices):
                    per_user[user_index][metric] = float(values[local_index])
    if collect_details:
        result["per_user"] = per_user
    return result


def aggregate(parts: list[dict[str, object]]) -> dict[str, float]:
    users = sum(float(part["users"]) for part in parts)
    eligible = sum(float(part["eligible_users"]) for part in parts)
    result = {"users": users, "eligible_users": eligible}
    for k in (20, 50):
        for metric in ("recall", "ndcg", "precision", "hit_rate", "mrr", "map"):
            result[f"{metric}@{k}"] = sum(
                float(part[f"{metric}@{k}_sum"]) for part in parts
            ) / max(eligible, 1)
    return result


def load_current(device: torch.device):
    config = load_config(PROJECT_ROOT / "configs" / "ml32m.toml")
    catalog = pd.read_csv(MATRIX_DIR / "catalog.csv")
    genre_count = len(
        {
            genre
            for genres in catalog.genres.fillna("Unknown")
            for genre in str(genres).split("|")
        }
    )
    model = make_model(
        "tears_recvae",
        len(catalog),
        genre_count,
        config,
        RECVAE_CHECKPOINT,
        dropout=0.1,
        gamma=0.0035,
    )
    if not isinstance(model, HybridVAE):
        raise TypeError("Current TEARS checkpoint did not construct HybridVAE")
    payload = _checkpoint_payload(CURRENT_CHECKPOINT)
    model.load_state_dict(payload["model"], strict=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model.backbone)
    return model.to(device).eval(), tokenizer, config.model.max_text_tokens


def load_emiliano(device: torch.device):
    adapter = TEARSInferenceAdapter(device=device)
    return adapter.model.eval(), adapter.tokenizer, 512


def current_latents(
    model: HybridVAE,
    ratings: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
):
    rec = model.recvae.encode(ratings).mean
    editable = model.editable.encode(input_ids, attention_mask).mean
    return rec, editable, model.recvae.decoder


def emiliano_latents(model, ratings, input_ids, attention_mask):
    base = model.get_base_model()
    sentence = base.llm_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    distribution = base.mlp(sentence)
    editable = distribution[:, :400]
    try:
        vae = base.vae.modules_to_save["default"]
    except (AttributeError, KeyError, TypeError):
        vae = base.vae
    rec, _ = vae.encode(ratings)
    return rec, editable, vae.decoder


@torch.inference_mode()
def evaluate_model(
    name: str,
    split: dict[str, object],
    contract: dict[str, object],
    device: torch.device,
    batch_size: int,
    alphas: tuple[float, ...],
    collect_details: bool = False,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, list[dict[str, object]]],
    dict[str, float | int | str],
]:
    started = time.perf_counter()
    if name == "current":
        model, tokenizer, max_tokens = load_current(device)
        full_width = len(contract["catalog"])
        candidate_indices = torch.as_tensor(
            contract["shared_current_items"], dtype=torch.long, device=device
        )
        latent_function = current_latents
    elif name == "emiliano":
        model, tokenizer, max_tokens = load_emiliano(device)
        full_width = len(contract["old_movie_to_item"])
        candidate_indices = torch.as_tensor(
            contract["shared_old_items"], dtype=torch.long, device=device
        )
        latent_function = emiliano_latents
    else:
        raise ValueError(name)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    observed = split["observed"]
    target = split["target"]
    summaries = split["summaries"]
    assert isinstance(observed, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert isinstance(summaries, list)
    users = split["users"]
    assert isinstance(users, np.ndarray)
    parts: dict[float, list[dict[str, object]]] = {alpha: [] for alpha in alphas}
    details: dict[float, list[dict[str, object]]] = {alpha: [] for alpha in alphas}
    unpadded_tokens = 0
    padded_token_positions = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    alpha_tensor = torch.as_tensor(alphas, dtype=torch.float32, device=device).view(
        -1, 1, 1
    )
    candidate_weight = None
    candidate_bias = None
    batches = (len(summaries) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(summaries), batch_size), start=1):
        stop = min(start + batch_size, len(summaries))
        shared_observed = torch.from_numpy(observed[start:stop]).to(device)
        shared_target = torch.from_numpy(target[start:stop]).to(device)
        ratings = torch.zeros((stop - start, full_width), dtype=torch.float32, device=device)
        ratings[:, candidate_indices] = shared_observed
        encoded = tokenizer(
            summaries[start:stop],
            padding="max_length",
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )
        unpadded_tokens += int(encoded.attention_mask.sum())
        padded_token_positions += int(encoded.input_ids.numel())
        rec, editable, decoder = latent_function(
            model,
            ratings,
            encoded.input_ids.to(device),
            encoded.attention_mask.to(device),
        )
        if candidate_weight is None:
            candidate_weight = decoder.weight.index_select(0, candidate_indices)
            candidate_bias = (
                None
                if decoder.bias is None
                else decoder.bias.index_select(0, candidate_indices)
            )
        # Both paths use the paper convention: alpha=1 is text-only. Stack all
        # alpha latents so one matrix multiplication scores the full sweep.
        latents = alpha_tensor * editable.unsqueeze(0) + (
            1.0 - alpha_tensor
        ) * rec.unsqueeze(0)
        all_candidate_logits = F.linear(latents, candidate_weight, candidate_bias)
        for alpha_index, alpha in enumerate(alphas):
            batch_metrics = metric_batch(
                all_candidate_logits[alpha_index],
                shared_observed,
                shared_target,
                collect_details=collect_details,
            )
            if collect_details:
                raw_details = batch_metrics.pop("per_user")
                assert isinstance(raw_details, list)
                for offset, row in enumerate(raw_details):
                    row["user_id"] = int(users[start + offset])
                details[alpha].extend(raw_details)
            parts[alpha].append(batch_metrics)
        if batch_index == 1 or batch_index % 5 == 0 or batch_index == batches:
            print(f"{name}: batch {batch_index}/{batches}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started
    result = {f"{alpha:.3f}": aggregate(alpha_parts) for alpha, alpha_parts in parts.items()}
    detail_result = {f"{alpha:.3f}": rows for alpha, rows in details.items() if rows}
    resources: dict[str, float | int | str] = {
        "device": str(device),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "load_plus_inference_seconds": time.perf_counter() - started,
        "inference_seconds": inference_seconds,
        "users_per_second": len(summaries) / max(inference_seconds, 1e-9),
        "unpadded_t5_tokens": unpadded_tokens,
        "padded_t5_token_positions": padded_token_positions,
        "unpadded_tokens_per_user": unpadded_tokens / len(summaries),
        "padded_token_positions_per_user": padded_token_positions / len(summaries),
        "max_text_tokens": max_tokens,
        "batch_size": batch_size,
        "batches": batches,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, detail_result, resources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(2024)
    torch.manual_seed(2024)
    device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":
        raise RuntimeError(
            "This matched comparison is GPU-only; pass --device cuda:0 from a GPU job."
        )
    contract = load_contract()
    validation = load_split("validation", contract)

    validation_results: dict[str, object] = {}
    selections: dict[str, object] = {}
    for name in ("current", "emiliano"):
        metrics, _, resources = evaluate_model(
            name, validation, contract, device, args.batch_size, ALPHAS
        )
        selected = max(metrics, key=lambda value: metrics[value]["ndcg@50"])
        validation_results[name] = {
            "metrics_by_alpha": metrics,
            "resources": resources,
        }
        selections[name] = {
            "alpha": float(selected),
            "validation_metrics": metrics[selected],
        }

    selection_artifact = {
        "protocol": "emiliano-ml1m-shared-catalog-v1",
        "selection_metric": "validation/ndcg@50",
        "alpha_grid": list(ALPHAS),
        "shared_items": len(contract["shared_movies"]),
        "emiliano_items": len(contract["old_movie_to_item"]),
        "current_items": len(contract["catalog"]),
        "validation": {
            "users": len(validation["users"]),
            "shared_observed_interactions": validation[
                "shared_observed_interactions"
            ],
            "shared_positive_targets": validation["shared_positive_targets"],
            "dropped_observed_rows": validation["dropped_observed_rows"],
            "dropped_target_rows": validation["dropped_target_rows"],
        },
        "selections": selections,
        "checkpoints": {
            "current": {
                "path": str(CURRENT_CHECKPOINT),
                "sha256": sha256_file(CURRENT_CHECKPOINT),
                "bytes": CURRENT_CHECKPOINT.stat().st_size,
            },
            "emiliano": {
                "path": str(EMILIANO_CHECKPOINT),
                "sha256": sha256_file(EMILIANO_CHECKPOINT),
                "bytes": EMILIANO_CHECKPOINT.stat().st_size,
            },
        },
    }
    selection_path = args.output_dir / "emiliano_ml1m_frozen_selection.json"
    atomic_write_json(selection_path, selection_artifact)

    # Load test only after persisting the validation decision, so neither the
    # alpha sweep nor model selection can inspect the held-out examples.
    test = load_split("test", contract)
    test_results: dict[str, object] = {}
    for name in ("current", "emiliano"):
        alpha = float(selections[name]["alpha"])
        metrics, per_user, resources = evaluate_model(
            name,
            test,
            contract,
            device,
            args.batch_size,
            (alpha,),
            collect_details=True,
        )
        test_results[name] = {
            "selected_alpha": alpha,
            "metrics": metrics[f"{alpha:.3f}"],
            "per_user": per_user[f"{alpha:.3f}"],
            "resources": resources,
        }

    report = {
        "protocol": selection_artifact["protocol"],
        "device": str(device),
        "reproducibility": {
            "seed": 2024,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "candidate_catalog": {
            "shared_items": len(contract["shared_movies"]),
            "emiliano_items": len(contract["old_movie_to_item"]),
            "current_items": len(contract["catalog"]),
            "shared_movie_ids": contract["shared_movies"],
        },
        "inputs": {
            "same_summaries": str(SUMMARY_PATH),
            "same_shared_observed_histories": True,
            "positive_target_rating": 4.0,
            "seen_items_masked": True,
        },
        "frozen_selection": str(selection_path),
        "frozen_selection_sha256": sha256_file(selection_path),
        "validation": validation_results,
        "test": {
            "users": len(test["users"]),
            "shared_observed_interactions": test["shared_observed_interactions"],
            "shared_positive_targets": test["shared_positive_targets"],
            "dropped_observed_rows": test["dropped_observed_rows"],
            "dropped_target_rows": test["dropped_target_rows"],
            "models": test_results,
        },
        "limitations": [
            "The current ML-32M checkpoint is evaluated zero-shot; it is not retrained on ML-1M.",
            "The comparison uses the 2,729-item intersection because 16 Emiliano items are absent from the current support-20 catalog.",
            "The supplied Emiliano checkpoint is one checkpoint, whereas paper tables average five seeds.",
        ],
    }
    report_path = args.output_dir / "emiliano_ml1m_matched_test.json"
    atomic_write_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "frozen_selection": str(selection_path),
                "test_metrics": {
                    name: result["metrics"] for name, result in test_results.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
