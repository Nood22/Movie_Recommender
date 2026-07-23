#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import T5EncoderModel, T5Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_recommenders import make_split, metrics, top_k_indices
from tears_recommender import EASERecommender
from tears_recommender.quality_recommender import row_minmax


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def history_descriptions(catalog, profiles) -> list[str]:
    genre_counts = np.asarray(profiles @ catalog.genre_matrix)
    descriptions = []
    for row in range(profiles.shape[0]):
        genre_order = np.argsort(-genre_counts[row])
        genres = [
            catalog.genre_names[index]
            for index in genre_order[:5]
            if genre_counts[row, index] > 0
        ]
        history = profiles.indices[profiles.indptr[row] : profiles.indptr[row + 1]]
        # Evenly sample a bounded set so prolific users do not dominate token length.
        if len(history) > 12:
            positions = np.linspace(0, len(history) - 1, 12, dtype=int)
            history = history[positions]
        titles = [catalog.titles[index] for index in history]
        descriptions.append(
            "The user enjoys "
            + ", ".join(genres)
            + " movies. Movies they liked include: "
            + "; ".join(titles)
            + "."
        )
    return descriptions


def encode_descriptions(
    descriptions: list[str],
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)
    model = T5EncoderModel.from_pretrained(model_name, local_files_only=True).to(device)
    model.eval()
    encoded = []
    with torch.no_grad():
        for start in range(0, len(descriptions), batch_size):
            tokens = tokenizer(
                descriptions[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            hidden = model(**tokens).last_hidden_state
            mask = tokens.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            encoded.append(F.normalize(pooled, dim=-1).cpu())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(encoded)


def train_head(
    embeddings: torch.Tensor,
    profiles,
    catalog_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> nn.Linear:
    head = nn.Linear(embeddings.shape[1], catalog_size).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    order = np.arange(len(embeddings))
    for epoch in range(epochs):
        np.random.shuffle(order)
        total_loss = 0.0
        head.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            features = embeddings[indices].to(device)
            labels = torch.from_numpy(profiles[indices].toarray()).to(device)
            labels = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)
            logits = head(features)
            loss = -(F.log_softmax(logits, dim=1) * labels).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        print(
            f"epoch={epoch + 1} loss={total_loss / len(order):.6f}",
            flush=True,
        )
    return head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--model-name", default="t5-small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "model" / "saved_models" / "text_alignment.pt",
    )
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    started = time.perf_counter()
    catalog, profiles, targets, validation = make_split(args.max_users, args.seed)
    test = ~validation
    descriptions = history_descriptions(catalog, profiles)
    embeddings = encode_descriptions(
        descriptions, args.model_name, args.encode_batch_size, device
    )
    head = train_head(
        embeddings,
        profiles,
        catalog.size,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        device,
    )

    head.eval()
    with torch.no_grad():
        text_scores = []
        for start in range(0, len(embeddings), args.batch_size):
            text_scores.append(head(embeddings[start : start + args.batch_size].to(device)).cpu())
    text_scores = row_minmax(torch.cat(text_scores).numpy())
    popularity = row_minmax(
        np.log1p(np.asarray(profiles.astype(bool).sum(axis=0)).ravel())
    )

    ease = EASERecommender(500.0).fit(profiles)
    validation_profiles = profiles[validation]
    validation_ease = row_minmax(ease.score_batch(validation_profiles))
    blend_candidates = []
    for text_weight in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
        scores = (
            (1.0 - text_weight) * validation_ease
            + text_weight * text_scores[validation]
        )
        ranked = top_k_indices(scores, validation_profiles, args.top_k)
        result = metrics(
            ranked,
            targets[validation],
            catalog,
            popularity,
        )
        blend_candidates.append((result[f"ndcg@{args.top_k}"], -text_weight, text_weight))
    best_validation_ndcg, _, best_text_weight = max(blend_candidates)

    test_profiles = profiles[test]
    test_ease = row_minmax(ease.score_batch(test_profiles))
    test_models = {
        "text_alignment": text_scores[test],
        "ease_text_blend": (
            (1.0 - best_text_weight) * test_ease
            + best_text_weight * text_scores[test]
        ),
    }
    model_results = []
    for name, scores in test_models.items():
        ranked = top_k_indices(scores, test_profiles, args.top_k)
        model_results.append(
            {
                "name": name,
                "metrics": metrics(
                    ranked,
                    targets[test],
                    catalog,
                    popularity,
                ),
            }
        )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.cpu().state_dict(),
            "model_name": args.model_name,
            "catalog_size": catalog.size,
            "input_dimension": embeddings.shape[1],
            "seed": args.seed,
            "description_template": "top-5 genres plus up to 12 liked titles",
        },
        args.checkpoint,
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "users": profiles.shape[0],
        "test_users": int(test.sum()),
        "selected_text_weight": best_text_weight,
        "validation_ndcg": best_validation_ndcg,
        "models": model_results,
        "checkpoint": str(args.checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"text_alignment_seed{args.seed}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
