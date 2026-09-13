from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import signal
import tempfile
import time
from typing import Any

import numpy as np
from scipy import sparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .artifacts import (
    append_jsonl,
    atomic_write_json,
    base_manifest,
    sha256_file,
    stable_hash,
)
from .config import MODEL_NAMES, load_config
from .evaluate import aggregate_metric_rows, ranking_metrics
from .models import (
    EditableBase,
    GenreEncoder,
    HybridVAE,
    RecVAE,
    T5SummaryEncoder,
    gaussian_kl,
    hybrid_loss,
    multinomial_reconstruction,
)


STOP_REQUESTED = False


def _request_graceful_stop(signum: int, frame: object) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


class SparseRows(Dataset[tuple[int, torch.Tensor]]):
    def __init__(self, matrix: sparse.csr_matrix, rows: np.ndarray):
        self.matrix = matrix
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        row = int(self.rows[index])
        dense = self.matrix.getrow(row).toarray()[0].astype(np.float32, copy=False)
        return row, torch.from_numpy(dense)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def genre_features(matrix_dir: Path, observed: sparse.csr_matrix) -> torch.Tensor:
    import pandas as pd

    catalog = pd.read_csv(matrix_dir / "catalog.csv")
    names = sorted(
        {
            genre
            for value in catalog.genres.fillna("Unknown")
            for genre in str(value).split("|")
        }
    )
    mapping = {name: index for index, name in enumerate(names)}
    item_genres = np.zeros((len(catalog), len(names)), dtype=np.float32)
    for item, value in enumerate(catalog.genres.fillna("Unknown")):
        for genre in str(value).split("|"):
            item_genres[item, mapping[genre]] = 1
    binary = observed.copy()
    binary.data = (binary.data >= 4).astype(np.float32)
    binary.eliminate_zeros()
    values = np.asarray(binary @ item_genres, dtype=np.float32)
    values /= np.maximum(values.sum(1, keepdims=True), 1)
    return torch.from_numpy(values)


def load_summary_tokens(
    matrix_dir: Path,
    summaries: Path,
    backbone: str,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import pandas as pd
    from transformers import AutoTokenizer

    users = pd.read_csv(matrix_dir / "users.csv")
    records = {
        int(row["user_id"]): row["summary"]
        for row in (
            json.loads(line)
            for line in summaries.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    missing = [int(user) for user in users.userId if int(user) not in records]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} summaries; first user={missing[0]}")
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    encoded = tokenizer(
        [records[int(user)] for user in users.userId],
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    return encoded.input_ids, encoded.attention_mask


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_model_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid checkpoint state: {path}")
    return state


def make_model(
    name: str,
    items: int,
    genres: int,
    config: Any,
    recvae_checkpoint: Path | None,
    dropout: float,
    gamma: float,
) -> torch.nn.Module:
    recvae = RecVAE(items, config.model.latent_dim, dropout=dropout, gamma=gamma)
    if recvae_checkpoint:
        recvae.load_state_dict(_checkpoint_model_state(recvae_checkpoint), strict=True)
    if name == "recvae":
        return recvae
    is_hybrid = name in {"tears_recvae", "gers_recvae"}
    if is_hybrid and recvae_checkpoint is None:
        raise RuntimeError("Hybrid training requires --recvae-checkpoint from the same seed")
    editable: torch.nn.Module
    if name.startswith("tears"):
        editable = T5SummaryEncoder(
            items,
            config.model.latent_dim,
            config.model.backbone,
            config.model.lora_rank,
            config.model.lora_alpha,
            dropout,
        )
    else:
        editable = GenreEncoder(genres, items, config.model.latent_dim)
    return HybridVAE(recvae, editable) if is_hybrid else EditableBase(editable)


def _distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-process training requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _extra_inputs(
    name: str,
    row_ids: torch.Tensor,
    genre: torch.Tensor,
    tokens: tuple[torch.Tensor, torch.Tensor] | None,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    if name.startswith("tears"):
        if tokens is None:
            raise RuntimeError("TEARS training requires validated --summaries")
        return tokens[0][row_ids].to(device), tokens[1][row_ids].to(device)
    return (genre[row_ids].to(device),)


@torch.no_grad()
def validation_metrics(
    model: torch.nn.Module,
    name: str,
    observed: sparse.csr_matrix,
    target: sparse.csr_matrix,
    rows: np.ndarray,
    genre: torch.Tensor,
    tokens: tuple[torch.Tensor, torch.Tensor] | None,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    core = _unwrap(model)
    core.eval()
    alphas = (0.0, 0.5, 1.0) if isinstance(core, HybridVAE) else (None,)
    by_alpha: dict[float | None, list[dict[str, float]]] = {alpha: [] for alpha in alphas}
    recommended_items: dict[float | None, set[int]] = {alpha: set() for alpha in alphas}
    loader = DataLoader(SparseRows(observed, rows), batch_size=batch_size, shuffle=False)
    for row_ids, ratings in loader:
        ratings = ratings.to(device)
        extra = _extra_inputs(name, row_ids.long(), genre, tokens, device)
        target_batch = torch.from_numpy(
            target[row_ids.numpy()].toarray().astype(np.float32, copy=False)
        ).to(device)
        if isinstance(core, HybridVAE):
            for alpha in alphas:
                assert alpha is not None
                logits = core(ratings, *extra, alpha=alpha)["merged_logits"]
                assert isinstance(logits, torch.Tensor)
                by_alpha[alpha].append(ranking_metrics(logits, ratings, target_batch))
                masked = logits.masked_fill(ratings > 0, -torch.inf)
                recommended_items[alpha].update(
                    torch.topk(masked, min(20, masked.shape[1]), dim=1)
                    .indices.detach().cpu().reshape(-1).tolist()
                )
        else:
            logits, _ = core(ratings) if name == "recvae" else core(*extra)
            by_alpha[None].append(ranking_metrics(logits, ratings, target_batch))
            masked = logits.masked_fill(ratings > 0, -torch.inf)
            recommended_items[None].update(
                torch.topk(masked, min(20, masked.shape[1]), dim=1)
                .indices.detach().cpu().reshape(-1).tolist()
            )
    aggregated = {alpha: aggregate_metric_rows(parts) for alpha, parts in by_alpha.items()}
    for alpha, values in aggregated.items():
        values["coverage@20"] = len(recommended_items[alpha]) / max(observed.shape[1], 1)
    if isinstance(core, HybridVAE):
        result: dict[str, float] = {
            f"alpha_{alpha:g}_{metric}": value
            for alpha, values in aggregated.items()
            if alpha is not None
            for metric, value in values.items()
        }
        result["selection_ndcg@50"] = float(
            np.mean([values["ndcg@50"] for values in aggregated.values()])
        )
        result["eligible_users"] = aggregated[0.5]["eligible_users"]
        return result
    return aggregated[None] | {"selection_ndcg@50": aggregated[None]["ndcg@50"]}


def _serializable_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def wandb_tracking_identity(
    training_fingerprint: str, args: argparse.Namespace
) -> tuple[dict[str, Any], str]:
    """Build a schedule/attempt-specific execution and W&B identity."""
    payload = {
        "training_fingerprint": training_fingerprint,
        "profile": args.profile,
        "epochs": args.epochs,
        "minimum_epochs": args.minimum_epochs,
        "patience": args.patience,
        "attempt": getattr(args, "tracking_attempt", 0),
    }
    return payload, stable_hash(payload)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def resolve_resume_path(explicit: Path | None, run_dir: Path) -> Path | None:
    if explicit is not None:
        return explicit
    automatic = run_dir / "resume.pt"
    return automatic if automatic.is_file() else None


def _cpu_byte_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8)
    return torch.as_tensor(value, dtype=torch.uint8, device="cpu")


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_byte_tensor(state["torch"]))
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(
            [_cpu_byte_tensor(cuda_state) for cuda_state in state["cuda"]]
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_graceful_stop)
    config = load_config(args.config)
    if args.dry_run:
        return {
            "dry_run": True,
            "profile": args.profile,
            "model": args.model,
            "seed": args.seed,
            "matrix_dir": str(args.matrix_dir),
            "requires_summaries": args.model.startswith("tears"),
            "requires_recvae_checkpoint": args.model in {"tears_recvae", "gers_recvae"},
        }
    if args.model.startswith("tears") and args.summaries is None:
        raise RuntimeError("TEARS models require --summaries")
    seed_everything(args.seed)
    rank, world_size, local_rank, device = _distributed_context()
    is_primary = rank == 0
    try:
        import pandas as pd

        matrix_manifest_path = args.matrix_dir / "manifest.json"
        matrix_manifest = json.loads(matrix_manifest_path.read_text(encoding="utf-8"))
        observed = sparse.load_npz(args.matrix_dir / "train_observed.npz").tocsr()
        validation_observed = sparse.load_npz(
            args.matrix_dir / "validation_observed.npz"
        ).tocsr()
        validation_target = sparse.load_npz(
            args.matrix_dir / "validation_target.npz"
        ).tocsr()
        users = pd.read_csv(args.matrix_dir / "users.csv")
        train_rows = users.loc[users.split == "train", "modelUserId"].to_numpy(np.int64)
        validation_rows = users.loc[
            users.split == "validation", "modelUserId"
        ].to_numpy(np.int64)
        if not len(train_rows) or not len(validation_rows):
            raise RuntimeError("Training and validation rows must both be nonempty")
        # Genre inputs use each user's own observed history. Validation targets are
        # never included in this feature matrix.
        genre = genre_features(args.matrix_dir, observed + validation_observed)
        tokens = (
            load_summary_tokens(
                args.matrix_dir,
                args.summaries,
                config.model.backbone,
                config.model.max_text_tokens,
            )
            if args.model.startswith("tears")
            else None
        )
        model = make_model(
            args.model,
            observed.shape[1],
            genre.shape[1],
            config,
            args.recvae_checkpoint,
            args.dropout,
            args.gamma,
        ).to(device)
        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=isinstance(model, HybridVAE),
            )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        fingerprint_payload = {
            "model": args.model,
            "seed": args.seed,
            "matrix_fingerprint": matrix_manifest["fingerprint"],
            "hyperparameters": {
                "dropout": args.dropout,
                "gamma": args.gamma,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "ot_weight": args.ot_weight,
                "batch_size": args.batch_size,
            },
            "recvae_checkpoint_sha256": (
                sha256_file(args.recvae_checkpoint) if args.recvae_checkpoint else None
            ),
            "summaries_sha256": sha256_file(args.summaries) if args.summaries else None,
        }
        fingerprint = stable_hash(fingerprint_payload)
        tracking_payload, tracking_fingerprint = wandb_tracking_identity(
            fingerprint, args
        )
        run_dir = (
            config.output_root
            / "checkpoints"
            / args.profile
            / args.model
            / f"seed-{args.seed}"
            / tracking_fingerprint[:12]
        )
        artifact_references = {
            "run_dir": str(run_dir),
            "best_checkpoint": str(run_dir / "best.pt"),
            "final_checkpoint": str(run_dir / "final.pt"),
            "resume_checkpoint": str(run_dir / "resume.pt"),
            "metrics": str(run_dir / "metrics.jsonl"),
            "result": str(run_dir / "result.json"),
            "input_recvae_checkpoint": (
                str(args.recvae_checkpoint) if args.recvae_checkpoint else None
            ),
        }
        metrics_path = run_dir / "metrics.jsonl"
        start, best, stale = 0, -np.inf, 0
        loaded_resume = resolve_resume_path(args.resume, run_dir)
        if loaded_resume:
            saved = torch.load(loaded_resume, map_location="cpu", weights_only=False)
            if saved["fingerprint"] != fingerprint:
                raise RuntimeError("Checkpoint fingerprint mismatch")
            _unwrap(model).load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start = saved["epoch"] + 1
            best = saved["best"]
            stale = saved.get("stale", 0)
            _restore_rng_state(saved["rng"])
        manifest = base_manifest(Path.cwd(), config.to_dict()) | {
            "fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "arguments": _serializable_arguments(args),
            "world_size": world_size,
            "wandb_tracking_payload": tracking_payload,
            "wandb_tracking_fingerprint": tracking_fingerprint,
            "wandb_run_id": tracking_fingerprint[:16],
            "training_fingerprint": fingerprint,
            "execution_fingerprint": tracking_fingerprint,
            "resume_checkpoint_loaded": str(loaded_resume) if loaded_resume else None,
            "artifact_references": artifact_references,
        }
        if is_primary:
            atomic_write_json(run_dir / "manifest.json", manifest)

        wandb_run = None
        if is_primary and not args.no_wandb:
            import wandb

            wandb_run = wandb.init(
                entity=config.tracking.entity,
                project=config.tracking.project,
                name=(
                    f"{args.profile}-{args.model}-{args.seed}-e{args.epochs}-"
                    f"{tracking_fingerprint[:8]}"
                ),
                id=tracking_fingerprint[:16],
                resume="allow",
                group=f"{args.profile}-{args.model}-{args.seed}-{fingerprint[:8]}",
                job_type=f"{args.profile}-training",
                config=manifest,
                mode=config.tracking.mode,
                dir=str(config.output_root / "wandb"),
            )
        dataset = SparseRows(observed, train_rows)
        sampler = (
            DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
            )
            if world_size > 1
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        last_epoch = start - 1
        for epoch in range(start, args.epochs):
            last_epoch = epoch
            epoch_started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            losses: list[float] = []
            final_parts: dict[str, float] = {}
            for row_ids, ratings in loader:
                row_ids = row_ids.long()
                ratings = ratings.to(device, non_blocking=True)
                extra = _extra_inputs(args.model, row_ids, genre, tokens, device)
                optimizer.zero_grad(set_to_none=True)
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with context:
                    core = _unwrap(model)
                    if isinstance(core, HybridVAE):
                        outputs = model(
                            ratings, *extra, alpha=config.model.train_alpha
                        )
                        loss, final_parts = hybrid_loss(
                            outputs,
                            ratings,
                            args.ot_weight,
                            min(
                                config.model.kl_anneal_cap,
                                epoch
                                / max(args.epochs // 2, 1)
                                * config.model.kl_anneal_cap,
                            ),
                        )
                    else:
                        logits, latent = (
                            model(ratings) if args.model == "recvae" else model(*extra)
                        )
                        if isinstance(core, RecVAE):
                            loss = core.loss(logits, ratings, latent)
                        else:
                            anneal = min(
                                config.model.kl_anneal_cap,
                                epoch
                                / max(args.epochs // 2, 1)
                                * config.model.kl_anneal_cap,
                            )
                            loss = multinomial_reconstruction(logits, ratings) + anneal * gaussian_kl(latent)
                        final_parts = {}
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(float(loss.detach()))
            if isinstance(_unwrap(model), RecVAE):
                _unwrap(model).update_prior()
            if world_size > 1:
                dist.barrier()
            if is_primary:
                validation = validation_metrics(
                    model,
                    args.model,
                    validation_observed,
                    validation_target,
                    validation_rows,
                    genre,
                    tokens,
                    device,
                    args.validation_batch_size,
                )
                score = validation["selection_ndcg@50"]
                epoch_seconds = time.perf_counter() - epoch_started
                metrics = {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "resource/epoch_seconds": epoch_seconds,
                    "resource/train_users_per_second": len(train_rows)
                    / max(epoch_seconds, 1e-9),
                    "resource/peak_gpu_memory_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else 0
                    ),
                    **final_parts,
                    **{f"validation/{key}": value for key, value in validation.items()},
                }
                append_jsonl(metrics_path, [metrics])
                if wandb_run:
                    wandb_run.log(metrics, step=epoch)
                improved = score > best
                best = max(best, score)
                stale = 0 if improved else stale + 1
                payload = {
                    "fingerprint": fingerprint,
                    "epoch": epoch,
                    "best": best,
                    "stale": stale,
                    "model": _unwrap(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "metrics": metrics,
                    "rng": _rng_state(),
                }
                atomic_checkpoint(run_dir / "resume.pt", payload)
                if improved:
                    atomic_checkpoint(run_dir / "best.pt", payload)
            if world_size > 1:
                # Rank zero decides early stopping and broadcasts it.
                stop = torch.tensor(
                    [1 if is_primary and stale >= args.patience else 0], device=device
                )
                dist.broadcast(stop, src=0)
                should_stop = bool(stop.item())
            else:
                should_stop = stale >= args.patience
            if (should_stop and epoch + 1 >= args.minimum_epochs) or STOP_REQUESTED:
                break
        if is_primary:
            resume_path = run_dir / "resume.pt"
            if resume_path.exists():
                final_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
                atomic_checkpoint(run_dir / "final.pt", final_payload)
            result = {
                "run_dir": str(run_dir),
                "best_validation_selection_ndcg@50": best,
                "epochs_completed": max(0, last_epoch + 1),
                "fingerprint": fingerprint,
                "execution_fingerprint": tracking_fingerprint,
                "checkpoint_sha256": {
                    name: sha256_file(run_dir / name)
                    for name in ("best.pt", "final.pt", "resume.pt")
                    if (run_dir / name).exists()
                },
                "device": str(device),
                "gpu_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else None
                ),
            }
            atomic_write_json(run_dir / "result.json", result)
            if wandb_run:
                wandb_run.summary.update(
                    {
                        "artifacts/run_dir": artifact_references["run_dir"],
                        "artifacts/best_checkpoint": artifact_references["best_checkpoint"],
                        "artifacts/final_checkpoint": artifact_references["final_checkpoint"],
                        "artifacts/resume_checkpoint": artifact_references["resume_checkpoint"],
                        "artifacts/result": artifact_references["result"],
                        "artifacts/checkpoint_sha256": result["checkpoint_sha256"],
                    }
                )
                wandb_run.finish()
            return result
        return {"rank": rank, "fingerprint": fingerprint}
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tears_training.train")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minimum-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.0035)
    parser.add_argument("--ot-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--summaries", type=Path)
    parser.add_argument("--recvae-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tracking-attempt", type=int, default=0)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
