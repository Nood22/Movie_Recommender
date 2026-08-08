from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from .artifacts import atomic_write_json, sha256_file


IDENTITY_FIELDS = {"model", "seed", "run", "fingerprint"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def aggregate_seed_metrics(
    rows: list[dict[str, Any]], baseline: str = "recvae"
) -> dict[str, Any]:
    models = sorted({str(row["model"]) for row in rows})
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in IDENTITY_FIELDS and isinstance(value, (int, float))
        }
    )
    summary: dict[str, Any] = {}
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        summary[model] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in model_rows if metric in row])),
                "std": float(
                    np.std(
                        [row[metric] for row in model_rows if metric in row],
                        ddof=1,
                    )
                )
                if sum(metric in row for row in model_rows) > 1
                else 0.0,
                "n": sum(metric in row for row in model_rows),
            }
            for metric in metric_names
            if any(metric in row for row in model_rows)
        }

    tests: dict[str, Any] = {}
    baseline_rows = {int(row["seed"]): row for row in rows if row["model"] == baseline}
    for model in models:
        if model == baseline:
            continue
        comparison = {int(row["seed"]): row for row in rows if row["model"] == model}
        shared_seeds = sorted(set(baseline_rows) & set(comparison))
        tests[model] = {}
        for metric in metric_names:
            seeds = [
                seed
                for seed in shared_seeds
                if metric in baseline_rows[seed] and metric in comparison[seed]
            ]
            if len(seeds) < 2:
                continue
            first = np.asarray([baseline_rows[seed][metric] for seed in seeds], dtype=float)
            second = np.asarray([comparison[seed][metric] for seed in seeds], dtype=float)
            ttest = stats.ttest_rel(second, first)
            try:
                wilcoxon = stats.wilcoxon(second, first)
                wilcoxon_result = {
                    "statistic": float(wilcoxon.statistic),
                    "pvalue": float(wilcoxon.pvalue),
                }
            except ValueError:
                wilcoxon_result = {"statistic": 0.0, "pvalue": 1.0}
            tests[model][metric] = {
                "seeds": seeds,
                "mean_paired_difference": float(np.mean(second - first)),
                "paired_t": {
                    "statistic": float(ttest.statistic),
                    "pvalue": float(ttest.pvalue),
                },
                "wilcoxon": wilcoxon_result,
            }
    return {
        "models": models,
        "metrics": metric_names,
        "summary": summary,
        "paired_tests_vs_baseline": tests,
    }


def promotion_report(
    execution_manifest: Path,
    results_jsonl: Path | None,
    summary_poll: Path | None,
) -> dict[str, Any]:
    document = json.loads(execution_manifest.read_text(encoding="utf-8"))
    plan = document.get("execution_plan", document)
    results = load_jsonl(results_jsonl) if results_jsonl else []
    completed = {row.get("job_name") for row in results if row.get("status") == "complete"}
    expected = {job["name"] for job in plan["jobs"]}
    summary = (
        json.loads(summary_poll.read_text(encoding="utf-8")) if summary_poll else None
    )
    gates = {
        "all_planned_jobs_complete": expected <= completed,
        "job_results_finite": all(
            np.isfinite(value)
            for row in results
            for value in row.values()
            if isinstance(value, float)
        ),
        "summary_batches_complete": (
            summary is None
            or all(batch.get("status") == "completed" for batch in summary["batches"])
        ),
        "cost_below_cap": summary is None
        or summary.get("cumulative_cost_usd", 0)
        <= float(plan["config"]["summaries"]["spend_cap_usd"]),
    }
    return {
        "plan_fingerprint": plan["fingerprint"],
        "phase": plan["phase"],
        "jobs_expected": len(expected),
        "jobs_complete": len(expected & completed),
        "summary_cost": summary.get("cumulative_cost_usd") if summary else None,
        "gates": gates,
        "ready_for_manual_promotion": all(gates.values()),
        "source_hashes": {
            "execution_manifest": sha256_file(execution_manifest),
            "results_jsonl": sha256_file(results_jsonl) if results_jsonl else None,
            "summary_poll": sha256_file(summary_poll) if summary_poll else None,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tears_training.report")
    sub = parser.add_subparsers(dest="action", required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--metrics-jsonl", type=Path, required=True)
    aggregate.add_argument("--baseline", default="recvae")
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--dry-run", action="store_true")
    promotion = sub.add_parser("promotion")
    promotion.add_argument("--execution-manifest", type=Path, required=True)
    promotion.add_argument("--results-jsonl", type=Path)
    promotion.add_argument("--summary-poll", type=Path)
    promotion.add_argument("--output", type=Path, required=True)
    promotion.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        result: dict[str, Any] = {"dry_run": True, "action": args.action}
    elif args.action == "aggregate":
        result = aggregate_seed_metrics(load_jsonl(args.metrics_jsonl), args.baseline)
    else:
        result = promotion_report(
            args.execution_manifest, args.results_jsonl, args.summary_poll
        )
    if not args.dry_run:
        atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
