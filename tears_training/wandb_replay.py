from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from .artifacts import atomic_write_json, stable_hash
from .config import load_config
from .train import wandb_tracking_identity


def latest_schedule_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts = [index for index, row in enumerate(rows) if int(row["epoch"]) == 0]
    if not starts:
        raise RuntimeError("Metrics history has no epoch-0 schedule boundary")
    latest = rows[starts[-1] :]
    epochs = [int(row["epoch"]) for row in latest]
    if epochs != list(range(len(latest))):
        raise RuntimeError(f"Latest metrics schedule is not contiguous: {epochs[:5]}...")
    return latest


def replay_run(manifest_path: Path, config_path: Path | None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = manifest_path.parent
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise RuntimeError(f"Missing metrics history: {metrics_path}")
    metrics = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not metrics:
        raise RuntimeError(f"Empty metrics history: {metrics_path}")
    metrics = latest_schedule_metrics(metrics)

    arguments = argparse.Namespace(**manifest["arguments"])
    tracking_payload, tracking_fingerprint = wandb_tracking_identity(
        manifest["fingerprint"], arguments
    )
    canonical_run_id = tracking_fingerprint[:16]
    replay_fingerprint = stable_hash(
        {
            "tracking_fingerprint": tracking_fingerprint,
            "source_slurm_job_id": manifest.get("slurm", {}).get("SLURM_JOB_ID"),
            "replay_version": 1,
        }
    )
    run_id = replay_fingerprint[:16]
    replay_path = run_dir / f"wandb-replay-{run_id}.json"
    if replay_path.exists():
        return json.loads(replay_path.read_text(encoding="utf-8")) | {"skipped": True}

    config = load_config(config_path)
    import wandb

    replay_manifest = manifest | {
        "wandb_tracking_payload": tracking_payload,
        "wandb_tracking_fingerprint": tracking_fingerprint,
        "wandb_run_id": run_id,
        "wandb_canonical_run_id": canonical_run_id,
        "wandb_replayed_from_run_id": manifest.get("wandb_run_id"),
    }
    run = wandb.init(
        entity=config.tracking.entity,
        project=config.tracking.project,
        name=(
            f"{arguments.profile}-{arguments.model}-{arguments.seed}-"
            f"e{arguments.epochs}-{tracking_fingerprint[:8]}"
        ),
        id=run_id,
        resume="never",
        group=(
            f"{arguments.profile}-{arguments.model}-{arguments.seed}-"
            f"{manifest['fingerprint'][:8]}"
        ),
        job_type=f"{arguments.profile}-training-replay",
        config=replay_manifest,
        mode=config.tracking.mode,
        dir=str(config.output_root / "wandb"),
    )
    for row in metrics:
        run.log(row, step=int(row["epoch"]))
    run.summary["tracking/replayed"] = True
    run.summary["tracking/source_run_id"] = manifest.get("wandb_run_id")
    run.summary["tracking/source_slurm_job_id"] = manifest.get("slurm", {}).get(
        "SLURM_JOB_ID"
    )
    run.finish()

    record = {
        "replayed_at": int(time.time()),
        "epochs": len(metrics),
        "source_run_id": manifest.get("wandb_run_id"),
        "source_slurm_job_id": manifest.get("slurm", {}).get("SLURM_JOB_ID"),
        "wandb_canonical_run_id": canonical_run_id,
        "wandb_run_id": run_id,
        "wandb_url": (
            f"https://wandb.ai/{config.tracking.entity}/"
            f"{config.tracking.project}/runs/{run_id}"
        ),
    }
    atomic_write_json(replay_path, record)
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tears_training.wandb_replay")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    records = [replay_run(path, args.config) for path in args.manifest]
    print(json.dumps({"runs": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
