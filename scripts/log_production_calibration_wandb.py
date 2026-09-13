"""Log the completed production summary calibration to standard W&B.

This deliberately uses the project's existing wandb package, entity, project,
metric namespaces, and summary-batch-monitor job type. It has no Weave usage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from tears_training.artifacts import atomic_write_json, stable_hash
from tears_training.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    plan = json.loads((args.source_root / "request_plan.json").read_text())
    submission = json.loads((args.source_root / "submission.json").read_text())
    poll = json.loads((args.source_root / "poll.json").read_text())["batch"]
    validation = json.loads(
        (args.candidate_root / "reports" / "automated_validation_report.json").read_text()
    )
    run_id = stable_hash(
        {
            "purpose": "summary-batch-monitor-v1",
            "plan": plan["fingerprint"],
            "candidate_protocol": json.loads(
                (args.candidate_root / "protocol.json").read_text()
            )["fingerprint"],
        }
    )[:16]
    run_name = f"production-summary-calibration-1000-{run_id[:8]}"
    run_url = (
        f"https://wandb.ai/{config.tracking.entity}/"
        f"{config.tracking.project}/runs/{run_id}"
    )
    metadata = {
        "entity": config.tracking.entity,
        "project": config.tracking.project,
        "mode": config.tracking.mode,
        "run_id": run_id,
        "run_name": run_name,
        "run_url": run_url,
        "job_type": "summary-batch-monitor",
        "source_root": str(args.source_root),
        "candidate_root": str(args.candidate_root),
        "plan_fingerprint": plan["fingerprint"],
        "batch_id": submission["batch_id"],
        "weave_enabled": False,
        "post_hoc_completed_job": True,
    }
    metrics = {
        "batch/total_requests": 1000,
        "batch/completed_requests": int(poll["request_counts"]["completed"]),
        "batch/failed_requests": int(poll["request_counts"]["failed"]),
        "batch/completion_percent": 100.0,
        "time/elapsed_seconds": int(poll["completed_at"] - poll["created_at"]),
        "usage/input_tokens": validation["usage"]["input_tokens"],
        "usage/output_tokens": validation["usage"]["output_tokens"],
        "cost/estimated_usd": submission["projected_reserved_cost_usd"],
        "cost/actual_usd": validation["cost"]["exact_calibration_usd"],
        "cost/cumulative_usd": validation["cost"]["exact_calibration_usd"],
        "validation/valid_summaries": validation["successful_api_responses"],
        "validation/invalid_summaries": validation["missing"],
        "quality/missing_user_ids": validation["missing"],
        "quality/duplicate_summary_groups": validation["duplicates"][
            "exact_group_count"
        ],
        "quality/near_duplicate_pairs": validation["duplicates"]["near_pair_count"],
        "quality/summary_length_min": validation["length"]["final"]["minimum"],
        "quality/summary_length_max": validation["length"]["final"]["maximum"],
        "quality/summary_length_mean": validation["length"]["final"]["mean"],
        "quality/summary_length_median": validation["length"]["final"]["median"],
        "quality/summary_length_p95": validation["length"]["final"]["p95_linear"],
        "quality/repaired_summaries": validation["repaired"],
        "quality/materially_compressed": validation["length"][
            "materially_compressed"
        ],
        "quality/formatting_failures": validation["formatting_failures"],
        "quality/placeholders": validation["placeholders"],
    }
    import wandb

    wandb_dir = args.candidate_root / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=config.tracking.entity,
        project=config.tracking.project,
        name=run_name,
        id=run_id,
        resume="allow",
        job_type="summary-batch-monitor",
        config=metadata,
        mode=config.tracking.mode,
        dir=str(wandb_dir),
    )
    run.log(metrics, step=1000)
    for key, value in metrics.items():
        run.summary[key] = value
    run.summary["batch/status"] = "completed"
    run.summary["monitor/post_hoc"] = True
    run.summary["monitor/live_generation_available"] = False
    run.finish()
    metadata["logged_at"] = int(time.time())
    metadata["metrics"] = metrics
    atomic_write_json(args.candidate_root / "wandb_monitor_config.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
