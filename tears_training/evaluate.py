from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse
import torch

from .artifacts import atomic_write_json, sha256_file


def ranking_metrics(
    logits: torch.Tensor,
    observed: torch.Tensor,
    target: torch.Tensor,
    ks: Iterable[int] = (20, 50),
) -> dict[str, float]:
    logits = logits.clone()
    logits[observed > 0] = -torch.inf
    maximum = min(max(ks), logits.shape[1])
    top = torch.topk(logits, maximum, dim=1).indices
    truth = target > 0
    eligible = truth.sum(1) > 0
    result: dict[str, float] = {
        "users": float(len(logits)),
        "eligible_users": float(eligible.sum()),
    }
    if not eligible.any():
        return result | {f"recall@{k}": 0.0 for k in ks} | {
            f"ndcg@{k}": 0.0 for k in ks
        }
    for k in ks:
        width = min(k, maximum)
        hits = truth.gather(1, top[:, :width]).float()[eligible]
        counts = truth.sum(1)[eligible].float()
        result[f"recall@{k}"] = float(
            (hits.sum(1) / counts.clamp_min(1)).mean()
        )
        discount = 1.0 / torch.log2(
            torch.arange(width, device=logits.device, dtype=torch.float32) + 2
        )
        dcg = (hits * discount).sum(1)
        ideal_length = counts.clamp_max(width).long()
        idcg = torch.stack([discount[: int(length)].sum() for length in ideal_length])
        result[f"ndcg@{k}"] = float((dcg / idcg.clamp_min(1e-12)).mean())
    return result


def aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    eligible = sum(row.get("eligible_users", 0) for row in rows)
    users = sum(row.get("users", row.get("eligible_users", 0)) for row in rows)
    result = {"users": users, "eligible_users": eligible}
    metric_names = {
        key
        for row in rows
        for key in row
        if key not in {"users", "eligible_users"}
    }
    for key in metric_names:
        result[key] = (
            sum(row.get(key, 0) * row.get("eligible_users", 0) for row in rows)
            / max(eligible, 1)
        )
    return result


@torch.no_grad()
def evaluate_checkpoint(
    manifest_path: Path,
    checkpoint_path: Path | None,
    split: str,
    alpha: float | None,
    batch_size: int,
    confirm_test: bool,
    frozen_selection: Path | None,
) -> dict[str, object]:
    if split == "test" and (not confirm_test or frozen_selection is None):
        raise RuntimeError(
            "Test evaluation requires --confirm-test and a --frozen-selection artifact"
        )
    if frozen_selection is not None and not frozen_selection.is_file():
        raise FileNotFoundError(frozen_selection)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arguments = manifest["arguments"]
    from .config import load_config
    from .models import HybridVAE
    from .train import (
        SparseRows,
        _checkpoint_model_state,
        _extra_inputs,
        genre_features,
        load_summary_tokens,
        make_model,
    )
    import pandas as pd
    from torch.utils.data import DataLoader

    config_path = Path(arguments["config"]) if arguments.get("config") else None
    config = load_config(config_path)
    matrix_dir = Path(arguments["matrix_dir"])
    name = arguments["model"]
    observed = sparse.load_npz(matrix_dir / f"{split}_observed.npz").tocsr()
    target = sparse.load_npz(matrix_dir / f"{split}_target.npz").tocsr()
    users = pd.read_csv(matrix_dir / "users.csv")
    rows = users.loc[users.split == split, "modelUserId"].to_numpy(np.int64)
    genre = genre_features(matrix_dir, observed)
    summaries = Path(arguments["summaries"]) if arguments.get("summaries") else None
    tokens = (
        load_summary_tokens(
            matrix_dir,
            summaries,
            config.model.backbone,
            config.model.max_text_tokens,
        )
        if name.startswith("tears") and summaries is not None
        else None
    )
    recvae_checkpoint = (
        Path(arguments["recvae_checkpoint"])
        if arguments.get("recvae_checkpoint")
        else None
    )
    model = make_model(
        name,
        observed.shape[1],
        genre.shape[1],
        config,
        recvae_checkpoint,
        float(arguments["dropout"]),
        float(arguments["gamma"]),
    )
    checkpoint = checkpoint_path or manifest_path.parent / "best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("fingerprint") != manifest["fingerprint"]:
        raise RuntimeError("Checkpoint and run manifest fingerprints differ")
    model.load_state_dict(_checkpoint_model_state(checkpoint), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    if isinstance(model, HybridVAE):
        alphas = [alpha] if alpha is not None else [index / 10 for index in range(11)]
    else:
        alphas = [None]
    metric_parts: dict[float | None, list[dict[str, float]]] = {
        value: [] for value in alphas
    }
    loader = DataLoader(SparseRows(observed, rows), batch_size=batch_size, shuffle=False)
    for row_ids, ratings in loader:
        ratings = ratings.to(device)
        extra = _extra_inputs(name, row_ids.long(), genre, tokens, device)
        truth = torch.from_numpy(
            target[row_ids.numpy()].toarray().astype(np.float32, copy=False)
        ).to(device)
        for value in alphas:
            if isinstance(model, HybridVAE):
                assert value is not None
                logits = model(ratings, *extra, alpha=value)["merged_logits"]
                assert isinstance(logits, torch.Tensor)
            else:
                logits, _ = model(ratings) if name == "recvae" else model(*extra)
            metric_parts[value].append(ranking_metrics(logits, ratings, truth))
    by_alpha = {
        ("base" if value is None else f"{value:.1f}"): aggregate_metric_rows(parts)
        for value, parts in metric_parts.items()
    }
    result: dict[str, object] = {
        "manifest": str(manifest_path.resolve()),
        "manifest_fingerprint": manifest["fingerprint"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split": split,
        "metrics": by_alpha,
    }
    if split == "validation" and isinstance(model, HybridVAE):
        result["selected_alpha"] = max(
            by_alpha, key=lambda value: by_alpha[value]["ndcg@50"]
        )
    if frozen_selection is not None:
        result["frozen_selection_sha256"] = sha256_file(frozen_selection)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tears_training.evaluate")
    parser.add_argument("--metrics-jsonl", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument("--frozen-selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        result: dict[str, object] = {
            "dry_run": True,
            "run_manifest": str(args.run_manifest) if args.run_manifest else None,
            "split": args.split,
            "output": str(args.output),
        }
    elif args.metrics_jsonl:
        rows = [
            json.loads(line)
            for line in args.metrics_jsonl.read_text(encoding="utf-8").splitlines()
            if line
        ]
        result = aggregate_metric_rows(rows)
    elif args.run_manifest:
        result = evaluate_checkpoint(
            args.run_manifest,
            args.checkpoint,
            args.split,
            args.alpha,
            args.batch_size,
            args.confirm_test,
            args.frozen_selection,
        )
    else:
        raise RuntimeError("Provide --metrics-jsonl or --run-manifest")
    if not args.dry_run:
        atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
