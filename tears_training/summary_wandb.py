from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from .artifacts import atomic_write_json, stable_hash
from .config import load_config


MONITOR_FILENAME = "wandb_monitor_config.json"


def poll_metrics(result: dict[str, Any], submission: dict[str, Any]) -> dict[str, Any]:
    total = completed = failed = 0
    statuses: set[str] = set()
    created_at: list[int] = []
    for batch in result.get("batches", []):
        counts = batch.get("request_counts") or {}
        total += int(counts.get("total", 0))
        completed += int(counts.get("completed", 0))
        failed += int(counts.get("failed", 0))
        statuses.add(str(batch.get("status", "unknown")))
        if batch.get("created_at") is not None:
            created_at.append(int(batch["created_at"]))
    polled_at = int(result.get("polled_at", time.time()))
    started_at = min(created_at) if created_at else int(submission["submitted_at"])
    usage = result.get("usage") or {}
    return {
        "batch/status": ",".join(sorted(statuses)) or "unknown",
        "batch/total_requests": total,
        "batch/completed_requests": completed,
        "batch/failed_requests": failed,
        "batch/completion_percent": completed / total * 100 if total else 0.0,
        "time/elapsed_seconds": max(0, polled_at - started_at),
        "usage/input_tokens": int(usage.get("input_tokens", 0)),
        "usage/output_tokens": int(usage.get("output_tokens", 0)),
        "cost/estimated_usd": float(submission["projected_cost_usd"]),
        "cost/actual_usd": float(result.get("actual_cost_usd", 0.0)),
        "cost/cumulative_usd": float(result.get("cumulative_cost_usd", 0.0)),
    }


def validation_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "validation/valid_summaries": int(result.get("valid", 0)),
        "validation/invalid_summaries": int(result.get("invalid", 0)),
    }


def quality_metrics(report: dict[str, Any]) -> dict[str, Any]:
    validation = report.get("validation") or {}
    lengths = report.get("summary_lengths_words") or {}
    cost = report.get("cost_usd") or {}
    return {
        "quality/missing_user_ids": len(validation.get("missing_user_ids", [])),
        "quality/duplicate_api_user_ids": len(
            validation.get("duplicate_api_user_ids", [])
        ),
        "quality/duplicate_output_user_ids": len(
            validation.get("duplicate_output_user_ids", [])
        ),
        "quality/duplicate_summary_groups": int(
            validation.get("duplicate_summary_groups", 0)
        ),
        "quality/summary_length_min": lengths.get("min"),
        "quality/summary_length_max": lengths.get("max"),
        "quality/summary_length_mean": lengths.get("mean"),
        "quality/summary_length_median": lengths.get("median"),
        "quality/summary_length_p05": lengths.get("p05"),
        "quality/summary_length_p95": lengths.get("p95"),
        "cost/per_planned_user_usd": cost.get("per_planned_user"),
        "cost/per_valid_user_usd": cost.get("per_valid_user"),
    }


def setup_monitor(
    run_dir: Path, config_path: Path, run_name: str
) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    config = load_config(config_path)
    plan = json.loads((root / "request_plan.json").read_text(encoding="utf-8"))
    submission = json.loads((root / "submission.json").read_text(encoding="utf-8"))
    run_id = stable_hash(
        {"purpose": "summary-batch-monitor-v1", "plan": plan["fingerprint"]}
    )[:16]
    metadata: dict[str, Any] = {
        "entity": config.tracking.entity,
        "project": config.tracking.project,
        "mode": config.tracking.mode,
        "run_name": run_name,
        "run_id": run_id,
        "run_url": (
            f"https://wandb.ai/{config.tracking.entity}/"
            f"{config.tracking.project}/runs/{run_id}"
        ),
        "run_root": str(root),
        "plan_fingerprint": plan["fingerprint"],
        "batch_ids": [entry["batch_id"] for entry in submission["batches"]],
        "created_at": int(time.time()),
    }
    atomic_write_json(root / MONITOR_FILENAME, metadata)
    return metadata


def log_monitor_event(root: Path, action: str, result: dict[str, Any]) -> dict[str, Any] | None:
    root = root.expanduser().resolve()
    metadata_path = root / MONITOR_FILENAME
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    submission = json.loads((root / "submission.json").read_text(encoding="utf-8"))
    metrics: dict[str, Any] = {}
    poll_path = root / "poll.json"
    if poll_path.is_file():
        metrics.update(
            poll_metrics(
                json.loads(poll_path.read_text(encoding="utf-8")), submission
            )
        )
    if action == "validate":
        metrics.update(validation_metrics(result))
    elif action == "quality":
        metrics.update(quality_metrics(result))
        validation_path = root / "validated" / "validation_report.json"
        if validation_path.is_file():
            metrics.update(
                validation_metrics(
                    json.loads(validation_path.read_text(encoding="utf-8"))
                )
            )
    if not metrics:
        return metadata

    import wandb

    wandb_dir = root / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=metadata["entity"],
        project=metadata["project"],
        name=metadata["run_name"],
        id=metadata["run_id"],
        resume="allow",
        job_type="summary-batch-monitor",
        config=metadata,
        mode=metadata["mode"],
        dir=str(wandb_dir),
    )
    run.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    run.summary["monitor/last_action"] = action
    run.finish()
    metadata["last_logged_at"] = int(time.time())
    metadata["last_action"] = action
    atomic_write_json(metadata_path, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="tears-ml32m-summaries-10k")
    args = parser.parse_args()
    metadata = setup_monitor(args.run_dir, args.config, args.run_name)
    poll_path = args.run_dir / "poll.json"
    if poll_path.is_file():
        log_monitor_event(
            args.run_dir,
            "poll",
            json.loads(poll_path.read_text(encoding="utf-8")),
        )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
