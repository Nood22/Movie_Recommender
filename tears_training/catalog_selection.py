from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, sha256_file, stable_hash


MEMORY_LIMIT_BYTES = 72 * 1024**3


def summarize_run(run_dir: Path, support: int) -> dict[str, Any]:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise RuntimeError(f"No epoch metrics in {run_dir}")
    best_row = max(rows, key=lambda row: float(row["validation/ndcg@20"]))
    return {
        "support": support,
        "run_dir": str(run_dir.resolve()),
        "run_fingerprint": result["fingerprint"],
        "checkpoint": str((run_dir / "best.pt").resolve()),
        "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "validation_ndcg@20": float(best_row["validation/ndcg@20"]),
        "validation_coverage@20": float(best_row["validation/coverage@20"]),
        "peak_gpu_memory_bytes": max(
            int(row["resource/peak_gpu_memory_bytes"]) for row in rows
        ),
        "mean_epoch_seconds": sum(float(row["resource/epoch_seconds"]) for row in rows)
        / len(rows),
        "mean_train_users_per_second": sum(
            float(row["resource/train_users_per_second"]) for row in rows
        )
        / len(rows),
        "epochs_completed": int(result["epochs_completed"]),
    }


def select_catalog(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["peak_gpu_memory_bytes"] < MEMORY_LIMIT_BYTES]
    if not eligible:
        raise RuntimeError("No catalog candidate passed the 72 GiB/GPU memory gate")
    best = max(float(row["validation_ndcg@20"]) for row in eligible)
    within_five_percent = [
        row
        for row in eligible
        if best <= 0 or (best - float(row["validation_ndcg@20"])) / best <= 0.05
    ]
    # Lower support means a larger catalog.
    selected = min(within_five_percent, key=lambda row: int(row["support"]))
    report: dict[str, Any] = {
        "selection_rule": "largest catalog within 5% relative of best validation NDCG@20 and below 72 GiB/GPU",
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "candidates": sorted(rows, key=lambda row: int(row["support"])),
        "selected_support": int(selected["support"]),
        "selected_run_dir": selected["run_dir"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_run_fingerprint": selected["run_fingerprint"],
    }
    report["fingerprint"] = stable_hash(report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tears_training.catalog_selection")
    parser.add_argument("--run", nargs=2, action="append", metavar=("SUPPORT", "RUN_DIR"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [summarize_run(Path(path), int(support)) for support, path in args.run]
    report = select_catalog(rows)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
