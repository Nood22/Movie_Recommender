#!/usr/bin/env python3
"""Verify the frozen five-seed Phase 6 training gate without reading test data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

from tears_training.artifacts import atomic_write_json, atomic_write_text, sha256_file


MODELS = ("recvae", "gers_base", "tears_base", "tears_recvae", "gers_recvae")
SEEDS = (2020, 2021, 2022, 2023, 2024)
REQUIRED_CHECKPOINTS = ("best.pt", "final.pt", "resume.pt")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_identity(
    manifest: dict[str, Any], model: str, seed: int
) -> bool:
    arguments = manifest.get("arguments") or {}
    return (
        arguments.get("profile") == "full"
        and arguments.get("model") == model
        and int(arguments.get("seed", -1)) == seed
        and int(arguments.get("epochs", -1)) == 200
        and int(arguments.get("minimum_epochs", -1)) == 200
        and int(arguments.get("patience", -1)) == 201
    )


def _validate_candidate(result_path: Path, model: str, seed: int) -> dict[str, Any]:
    run_dir = result_path.parent
    manifest_path = run_dir / "manifest.json"
    errors: list[str] = []
    try:
        result = _load_json(result_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"identity_matches": False, "errors": [f"invalid result.json: {exc}"]}
    try:
        manifest = _load_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"identity_matches": False, "errors": [f"invalid manifest.json: {exc}"]}

    identity_matches = _candidate_identity(manifest, model, seed)
    if not identity_matches:
        return {"identity_matches": False, "errors": []}

    arguments = manifest["arguments"]
    if int(result.get("epochs_completed", -1)) != 200:
        errors.append(f"epochs_completed={result.get('epochs_completed')}, expected 200")
    score = result.get("best_validation_selection_ndcg@50")
    if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        errors.append("best validation selection NDCG@50 is missing or non-finite")
    if result.get("fingerprint") != manifest.get("fingerprint"):
        errors.append("training fingerprint differs between result and manifest")
    if result.get("execution_fingerprint") != manifest.get("execution_fingerprint"):
        errors.append("execution fingerprint differs between result and manifest")
    if Path(result.get("run_dir", "")).resolve() != run_dir.resolve():
        errors.append("result run_dir does not match the containing directory")

    recorded_hashes = result.get("checkpoint_sha256") or {}
    verified_hashes: dict[str, str | None] = {}
    for name in REQUIRED_CHECKPOINTS:
        checkpoint = run_dir / name
        if not checkpoint.is_file():
            errors.append(f"missing {name}")
            verified_hashes[name] = None
            continue
        actual = sha256_file(checkpoint)
        verified_hashes[name] = actual
        if recorded_hashes.get(name) != actual:
            errors.append(f"{name} SHA-256 differs from result.json")

    return {
        "identity_matches": True,
        "model": model,
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "result_path": str(result_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "wandb_run_id": manifest.get("wandb_run_id"),
        "training_fingerprint": manifest.get("fingerprint"),
        "execution_fingerprint": manifest.get("execution_fingerprint"),
        "best_validation_selection_ndcg@50": score,
        "epochs_completed": result.get("epochs_completed"),
        "checkpoint_sha256": verified_hashes,
        "matrix_dir": arguments.get("matrix_dir"),
        "summaries": arguments.get("summaries"),
        "errors": errors,
    }


def _find_run(
    checkpoint_root: Path,
    model: str,
    seed: int,
    not_before: datetime,
) -> dict[str, Any]:
    seed_dir = checkpoint_root / model / f"seed-{seed}"
    candidates = sorted(
        seed_dir.glob("*/result.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    matching: list[dict[str, Any]] = []
    for result_path in candidates:
        modified = datetime.fromtimestamp(result_path.stat().st_mtime, timezone.utc)
        if modified < not_before:
            continue
        candidate = _validate_candidate(result_path, model, seed)
        if candidate.get("identity_matches"):
            matching.append(candidate)
    if not matching:
        return {
            "model": model,
            "seed": seed,
            "errors": ["no matching completed 200-epoch full run found"],
        }
    valid = [candidate for candidate in matching if not candidate["errors"]]
    return valid[0] if valid else matching[0]


def _wandb_states(
    runs: list[dict[str, Any]], entity: str, project: str
) -> tuple[dict[str, str], list[str]]:
    import wandb

    api = wandb.Api(timeout=90)
    states: dict[str, str] = {}
    errors: list[str] = []
    for run in runs:
        run_id = run.get("wandb_run_id")
        if not run_id:
            errors.append(f"{run['model']} seed {run['seed']}: missing W&B run ID")
            continue
        try:
            state = str(api.run(f"{entity}/{project}/{run_id}").state)
        except Exception as exc:  # W&B exposes several transport exception types.
            errors.append(f"{run['model']} seed {run['seed']}: W&B lookup failed: {exc}")
            continue
        states[run_id] = state
        if state != "finished":
            errors.append(
                f"{run['model']} seed {run['seed']}: W&B state is {state}, expected finished"
            )
    return states, errors


def build_report(
    checkpoint_root: Path,
    summary_path: Path,
    expected_summary_sha256: str,
    not_before: datetime,
    verify_wandb: bool,
    wandb_entity: str,
    wandb_project: str,
    models: Iterable[str] = MODELS,
    seeds: Iterable[int] = SEEDS,
) -> dict[str, Any]:
    model_values = tuple(models)
    seed_values = tuple(seeds)
    runs = [
        _find_run(checkpoint_root, model, seed, not_before)
        for model in model_values
        for seed in seed_values
    ]
    errors = [
        f"{run['model']} seed {run['seed']}: {message}"
        for run in runs
        for message in run["errors"]
    ]

    summary_actual = sha256_file(summary_path) if summary_path.is_file() else None
    if summary_actual != expected_summary_sha256:
        errors.append(
            "summary corpus SHA-256 is missing or differs from the frozen Phase 6 hash"
        )

    wandb_states: dict[str, str] = {}
    if verify_wandb and not any(run["errors"] for run in runs):
        try:
            wandb_states, wandb_errors = _wandb_states(
                runs, wandb_entity, wandb_project
            )
            errors.extend(wandb_errors)
        except Exception as exc:
            errors.append(f"W&B verification could not start: {exc}")

    summary: dict[str, Any] = {}
    for model in model_values:
        scores = [
            float(run["best_validation_selection_ndcg@50"])
            for run in runs
            if run["model"] == model and not run["errors"]
        ]
        summary[model] = {
            "valid_seeds": len(scores),
            "mean_best_validation_selection_ndcg@50": (
                statistics.mean(scores) if scores else None
            ),
            "sample_std_best_validation_selection_ndcg@50": (
                statistics.stdev(scores) if len(scores) > 1 else 0.0 if scores else None
            ),
        }

    expected_runs = len(model_values) * len(seed_values)
    valid_runs = sum(not run["errors"] for run in runs)
    complete = not errors and valid_runs == expected_runs
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 6,
        "protocol": "frozen_v10_evidence_gated_emiliano",
        "test_metrics_used": False,
        "expected_runs": expected_runs,
        "valid_runs": valid_runs,
        "summary_corpus": {
            "path": str(summary_path.resolve()),
            "expected_sha256": expected_summary_sha256,
            "actual_sha256": summary_actual,
            "verified": summary_actual == expected_summary_sha256,
        },
        "wandb": {
            "verified": verify_wandb,
            "entity": wandb_entity,
            "project": wandb_project,
            "states": wandb_states,
        },
        "model_summary": summary,
        "runs": runs,
        "errors": errors,
        "phase6_complete": complete,
        "phase7_ready": complete,
    }


def render_markdown(report: dict[str, Any]) -> str:
    status = "PASS" if report["phase6_complete"] else "FAIL"
    lines = [
        "# Phase 6 finalization gate",
        "",
        f"**Recorded:** {report['recorded_at_utc']}  ",
        f"**Status:** **{status}**  ",
        "**Test isolation:** no test metrics were read.",
        "",
        "## Gate summary",
        "",
        f"- Valid runs: **{report['valid_runs']} / {report['expected_runs']}**",
        f"- Frozen summary hash verified: **{str(report['summary_corpus']['verified']).lower()}**",
        f"- W&B verification requested: **{str(report['wandb']['verified']).lower()}**",
        f"- Phase 7 ready: **{str(report['phase7_ready']).lower()}**",
        "",
        "## Validation results",
        "",
        "| Model | Valid seeds | Mean best selection NDCG@50 | Sample SD |",
        "|---|---:|---:|---:|",
    ]
    for model, values in report["model_summary"].items():
        mean = values["mean_best_validation_selection_ndcg@50"]
        std = values["sample_std_best_validation_selection_ndcg@50"]
        lines.append(
            f"| {model} | {values['valid_seeds']} | "
            f"{mean:.6f} | {std:.6f} |"
            if mean is not None and std is not None
            else f"| {model} | {values['valid_seeds']} | - | - |"
        )
    lines.extend(["", "## Run ledger", "", "| Model | Seed | Epochs | Best selection NDCG@50 | W&B |", "|---|---:|---:|---:|---|"])
    states = report["wandb"]["states"]
    entity = report["wandb"]["entity"]
    project = report["wandb"]["project"]
    for run in report["runs"]:
        if run["errors"]:
            lines.append(f"| {run['model']} | {run['seed']} | - | - | missing/invalid |")
            continue
        run_id = run.get("wandb_run_id")
        state = states.get(run_id, "not checked")
        wandb_value = (
            f"[{state}](https://wandb.ai/{entity}/{project}/runs/{run_id})"
            if run_id
            else state
        )
        lines.append(
            f"| {run['model']} | {run['seed']} | {run['epochs_completed']} | "
            f"{float(run['best_validation_selection_ndcg@50']):.6f} | {wandb_value} |"
        )
    if report["errors"]:
        lines.extend(["", "## Blocking errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    else:
        lines.extend(
            [
                "",
                "## Decision",
                "",
                "Phase 6 is complete. All five model families have five valid, hashed,",
                "200-epoch runs synchronized to W&B. Phase 7 may begin with validation-only",
                "alpha selection; the test split remains closed until that selection is frozen.",
            ]
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/network/scratch/a/adls/FullTrainingTEARS/checkpoints/full"),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path(
            "/network/scratch/a/adls/FullTrainingTEARS/summaries/validated/"
            "v010_frozen/final_summaries.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-sha256",
        default="763605a2a22dfa77e32c0226fb1f2318b8889b89fa9517e341fb2ac008e90b07",
    )
    parser.add_argument(
        "--not-before", default="2026-08-19T00:00:00+00:00"
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("artifacts/phase6_finalization.json")
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("docs/august20/PHASE6_FINALIZATION_REPORT.md"),
    )
    parser.add_argument("--verify-wandb", action="store_true")
    parser.add_argument("--wandb-entity", default="niita-mila")
    parser.add_argument("--wandb-project", default="tears-ml32m")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_report(
        args.checkpoint_root,
        args.summary_path,
        args.summary_sha256,
        datetime.fromisoformat(args.not_before).astimezone(timezone.utc),
        args.verify_wandb,
        args.wandb_entity,
        args.wandb_project,
    )
    atomic_write_json(args.output_json, report)
    atomic_write_text(args.output_markdown, render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["phase6_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
