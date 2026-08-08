from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import statistics

import numpy as np

from tears_training.artifacts import atomic_write_json
from tears_training.summaries import write_jsonl
from tears_training.summary_wandb import log_monitor_event


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sample-seed", type=int, default=2024)
    parser.add_argument("--sample-size", type=int, default=25)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    plan = json.loads((root / "request_plan.json").read_text(encoding="utf-8"))
    poll = json.loads((root / "poll.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (root / "validated" / "validation_report.json").read_text(encoding="utf-8")
    )
    raw = []
    for path in sorted((root / "responses").glob("*.jsonl")):
        raw.extend(read_jsonl(path))
    valid = read_jsonl(root / "validated" / "summaries.jsonl")
    invalid = read_jsonl(root / "validated" / "invalid.jsonl")

    records = {record["custom_id"]: record for record in plan["records"]}
    planned_users = [int(record["user_id"]) for record in plan["records"]]
    response_ids = [str(row.get("custom_id")) for row in raw]
    response_counts = Counter(response_ids)
    missing_custom_ids = sorted(set(records) - set(response_ids))
    unexpected_custom_ids = sorted(set(response_ids) - set(records))
    duplicate_custom_ids = sorted(key for key, count in response_counts.items() if count > 1)
    duplicate_api_users = sorted(
        {int(records[key]["user_id"]) for key in duplicate_custom_ids if key in records}
    )
    missing_users = sorted(int(records[key]["user_id"]) for key in missing_custom_ids)
    output_users = [int(row["user_id"]) for row in valid + invalid]
    output_user_counts = Counter(output_users)
    duplicate_output_users = sorted(user for user, count in output_user_counts.items() if count > 1)

    word_counts = [len(row["summary"].split()) for row in valid]
    normalized_counts = Counter(" ".join(row["summary"].lower().split()) for row in valid)
    duplicate_summary_groups = sum(count > 1 for count in normalized_counts.values())
    error_counts = Counter(error for row in invalid for error in row.get("errors", []))
    batch_counts = Counter()
    for batch in poll["batches"]:
        counts = batch.get("request_counts") or {}
        for key in ("total", "completed", "failed"):
            batch_counts[key] += int(counts.get(key, 0))

    rng = random.Random(args.sample_seed)
    sample = rng.sample(sorted(valid, key=lambda row: int(row["user_id"])), min(args.sample_size, len(valid)))
    sample_rows = [
        {
            "user_id": int(row["user_id"]),
            "custom_id": row["custom_id"],
            "word_count": len(row["summary"].split()),
            "summary": row["summary"],
        }
        for row in sample
    ]
    sample_path = root / "quality" / f"manual_sample_seed-{args.sample_seed}_n-{len(sample_rows)}.jsonl"
    write_jsonl(sample_path, sample_rows)

    actual_cost = float(poll.get("actual_cost_usd", 0.0))
    report = {
        "plan_fingerprint": plan["fingerprint"],
        "validation_fingerprint": validation["validation_fingerprint"],
        "cohort": {
            "planned_users": len(planned_users),
            "unique_planned_users": len(set(planned_users)),
            "excluded_calibration_users": plan.get("excluded_user_ids"),
            "calibration_overlap": plan.get("excluded_overlap"),
        },
        "api": dict(batch_counts),
        "validation": {
            "valid": len(valid),
            "invalid": len(invalid),
            "error_counts": dict(sorted(error_counts.items())),
            "missing_user_ids": missing_users,
            "duplicate_api_user_ids": duplicate_api_users,
            "duplicate_output_user_ids": duplicate_output_users,
            "unexpected_custom_ids": unexpected_custom_ids,
            "duplicate_summary_groups": duplicate_summary_groups,
        },
        "summary_lengths_words": (
            {
                "min": min(word_counts),
                "max": max(word_counts),
                "mean": statistics.fmean(word_counts),
                "median": statistics.median(word_counts),
                "p05": float(np.percentile(word_counts, 5)),
                "p95": float(np.percentile(word_counts, 95)),
            }
            if word_counts
            else None
        ),
        "usage": poll.get("usage", {}),
        "cost_usd": {
            "total": actual_cost,
            "per_planned_user": actual_cost / len(planned_users) if planned_users else None,
            "per_valid_user": actual_cost / len(valid) if valid else None,
        },
        "manual_sample": {
            "seed": args.sample_seed,
            "size": len(sample_rows),
            "path": str(sample_path),
            "inspection_status": "pending_manual_review",
        },
    }
    output = root / "quality" / "quality_report.json"
    atomic_write_json(output, report)
    log_monitor_event(root, "quality", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
