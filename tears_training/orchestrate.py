from __future__ import annotations

import argparse
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from .artifacts import atomic_write_json, base_manifest, stable_hash
from .config import MODEL_NAMES, load_config


SLURM_PROFILES = {
    "pilot": "tears_pilot.sbatch",
    "full_2gpu": "tears_full_2gpu.sbatch",
    "full_4gpu": "tears_full_4gpu.sbatch",
}

SECRET_PATTERNS = (re.compile(r"sk-[A-Za-z0-9_-]{20,}"),)
KEY_ASSIGNMENT = re.compile(r"OPENAI_API_KEY\s*=\s*['\"]([^'\"]+)['\"]")


def scan_workspace_secrets(root: Path) -> list[dict[str, object]]:
    try:
        output = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return [{"file": None, "line": None, "kind": "git_scan_unavailable"}]
    findings: list[dict[str, object]] = []
    for encoded in output.split(b"\0"):
        if not encoded:
            continue
        relative = Path(os.fsdecode(encoded))
        path = root / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            assignment = KEY_ASSIGNMENT.search(line)
            assigned_secret = bool(
                assignment
                and assignment.group(1).strip().lower()
                not in {"...", "your-key", "your_api_key", "<secret>"}
            )
            if assigned_secret or any(pattern.search(line) for pattern in SECRET_PATTERNS):
                findings.append(
                    {"file": str(relative), "line": number, "kind": "possible_api_key"}
                )
    return findings


def preflight(root: Path, config: Any) -> dict[str, object]:
    return {
        "raw_data_exists": config.raw_data.is_dir(),
        "scratch_configured": bool(os.environ.get("SCRATCH")),
        "wandb_credentials_available": bool(
            os.environ.get("WANDB_API_KEY") or (Path.home() / ".netrc").is_file()
        ),
        "openai_key_in_environment": bool(os.environ.get("OPENAI_API_KEY")),
        "workspace_secret_findings": scan_workspace_secrets(root),
        "paid_submission_performed": False,
        "slurm_submission_performed": False,
    }


def tuning_grid(model: str) -> list[dict[str, float]]:
    if model == "recvae":
        return [
            {
                "dropout": dropout,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "weight_decay": weight_decay,
                "ot_weight": 0.1,
            }
            for dropout, learning_rate, gamma, weight_decay in product(
                (0.1, 0.2, 0.4),
                (1e-3, 1e-4, 1e-5),
                (0.0035, 0.004, 0.005),
                (0.0, 1e-5),
            )
        ]
    if model in {"tears_recvae", "gers_recvae"}:
        return [
            {
                "dropout": dropout,
                "learning_rate": learning_rate,
                "gamma": 0.0035,
                "weight_decay": 0.0,
                "ot_weight": ot_weight,
            }
            for dropout, learning_rate, ot_weight in product(
                (0.1, 0.2, 0.4), (1e-3, 1e-4), (0.1, 0.5, 1.0)
            )
        ]
    return [
        {
            "dropout": dropout,
            "learning_rate": learning_rate,
            "gamma": 0.0035,
            "weight_decay": 0.0,
            "ot_weight": 0.1,
        }
        for dropout, learning_rate in product((0.1, 0.2, 0.4), (1e-3, 1e-4))
    ]


def candidate_id(model: str, hyperparameters: dict[str, float]) -> str:
    return stable_hash({"model": model, "hyperparameters": hyperparameters})[:12]


def select_survivors(
    path: Path, models: tuple[str, ...], final: bool = False
) -> dict[str, list[dict[str, float]]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected: dict[str, list[dict[str, float]]] = {}
    for model in models:
        candidates = [row for row in rows if row["model"] == model]
        if not candidates:
            raise RuntimeError(f"No prior-rung results for {model}")
        candidates.sort(
            key=lambda row: (
                float(row["validation_selection_ndcg@50"]),
                -float(row.get("gpu_hours", math.inf)),
            ),
            reverse=True,
        )
        keep = 1 if final else max(1, math.ceil(len(candidates) / 3))
        if final and len(candidates) > 1:
            best = float(candidates[0]["validation_selection_ndcg@50"])
            near = [
                row
                for row in candidates
                if best <= 0
                or (best - float(row["validation_selection_ndcg@50"])) / best < 0.01
            ]
            winner = min(near, key=lambda row: float(row.get("gpu_hours", math.inf)))
            candidates = [winner]
        selected[model] = [row["hyperparameters"] for row in candidates[:keep]]
    return selected


def _training_arguments(
    model: str,
    profile: str,
    matrix_dir: Path,
    seed: int,
    epochs: int,
    hyperparameters: dict[str, float],
    config_path: Path | None,
    summaries: Path | None,
    recvae_checkpoint: Path | None,
    fixed_epochs: bool,
) -> list[str]:
    arguments = [
        "--profile",
        profile,
        "--matrix-dir",
        str(matrix_dir.resolve()),
        "--model",
        model,
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--minimum-epochs",
        str(epochs if fixed_epochs else min(30, epochs)),
        "--patience",
        str(epochs + 1 if fixed_epochs else 20),
    ]
    if config_path:
        arguments += ["--config", str(config_path.resolve())]
    for key in ("dropout", "learning_rate", "gamma", "weight_decay", "ot_weight"):
        arguments += [f"--{key.replace('_', '-')}", str(hyperparameters[key])]
    if model.startswith("tears"):
        if summaries is None:
            raise RuntimeError(f"{model} requires --summaries")
        arguments += ["--summaries", str(summaries.resolve())]
    if model in {"tears_recvae", "gers_recvae"}:
        if recvae_checkpoint is None:
            raise RuntimeError(f"{model} requires a same-seed RecVAE checkpoint")
        arguments += ["--recvae-checkpoint", str(recvae_checkpoint.resolve())]
    return arguments


def build_execution_plan(args: argparse.Namespace, config: Any, root: Path) -> dict[str, Any]:
    default_models = (
        ("recvae", "tears_base", "gers_base")
        if args.phase == "smoke"
        else MODEL_NAMES
    )
    models = tuple(args.models or default_models)
    unknown = set(models) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    selections: dict[str, list[dict[str, float]]]
    if args.phase == "smoke":
        if args.rung == 5:
            selections = {model: tuning_grid(model) for model in models}
        else:
            if args.previous_results is None:
                raise RuntimeError("Later smoke rungs require --previous-results")
            selections = select_survivors(
                args.previous_results, models, final=args.rung in {200, 300}
            )
        seeds = (2024,)
    else:
        if args.selection is None:
            raise RuntimeError("Pilot/full planning requires a frozen --selection JSON")
        raw_selection = json.loads(args.selection.read_text(encoding="utf-8"))
        selections = {model: [raw_selection[model]] for model in models}
        seeds = (2024,) if args.phase == "pilot" else config.seeds

    jobs: list[dict[str, Any]] = []
    schedules = (
        ((200, True), (300, False)) if args.phase == "smoke" and args.rung in {200, 300} else ((args.rung, False),)
    )
    for seed in seeds:
        for model in models:
            recvae_checkpoint = None
            if model in {"tears_recvae", "gers_recvae"}:
                if args.recvae_checkpoint_template is None:
                    raise RuntimeError(
                        "Hybrid jobs require --recvae-checkpoint-template with a {seed} field"
                    )
                recvae_checkpoint = Path(
                    args.recvae_checkpoint_template.format(seed=seed)
                )
            for hyperparameters in selections[model]:
                for epochs, fixed in schedules:
                    identifier = candidate_id(model, hyperparameters)
                    job_name = f"tears-{args.phase}-{model}-{seed}-{identifier}-e{epochs}"
                    jobs.append(
                        {
                            "name": job_name,
                            "model": model,
                            "seed": seed,
                            "candidate_id": identifier,
                            "epochs": epochs,
                            "fixed_epochs": fixed,
                            "hyperparameters": hyperparameters,
                            "arguments": _training_arguments(
                                model,
                                args.phase,
                                args.matrix_dir,
                                seed,
                                epochs,
                                hyperparameters,
                                args.config,
                                args.summaries,
                                recvae_checkpoint,
                                fixed,
                            ),
                        }
                    )
    plan: dict[str, Any] = base_manifest(root, config.to_dict()) | {
        "phase": args.phase,
        "rung": args.rung,
        "slurm_profile": args.slurm_profile,
        "models": models,
        "jobs": jobs,
        "manual_gates": {
            "paid_api_submission": "not_performed",
            "slurm_submission": "requires --submit --confirm-submit",
            "full_promotion": "requires matching approval JSON",
        },
    }
    plan["fingerprint"] = stable_hash(
        {
            "phase": plan["phase"],
            "rung": plan["rung"],
            "slurm_profile": plan["slurm_profile"],
            "models": plan["models"],
            "jobs": plan["jobs"],
            "git": plan["git"],
            "config_sha256": plan["config_sha256"],
        }
    )
    return plan


def _verify_promotion(path: Path, fingerprint: str) -> None:
    approval = json.loads(path.read_text(encoding="utf-8"))
    if approval.get("approved") is not True or approval.get("plan_fingerprint") != fingerprint:
        raise RuntimeError("Promotion approval does not match this execution plan")


def submit_plan(
    plan: dict[str, Any],
    root: Path,
    confirm: bool,
    promotion: Path | None,
    afterok: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    if not confirm:
        raise RuntimeError("Slurm submission requires --confirm-submit")
    if plan["phase"] == "full":
        if promotion is None:
            raise RuntimeError("Full submission requires --promotion-approval")
        _verify_promotion(promotion, plan["fingerprint"])
    findings = scan_workspace_secrets(root)
    if findings:
        raise RuntimeError(
            "Tracked secret scan failed; inspect the preflight report (secret values are never printed)"
        )
    script = root / "slurm" / SLURM_PROFILES[plan["slurm_profile"]]
    submitted: list[dict[str, str]] = []
    concurrency = {"pilot": 2, "full_2gpu": 8, "full_4gpu": 4}[
        plan["slurm_profile"]
    ]
    for index, job in enumerate(plan["jobs"]):
        command = [
            "sbatch",
            "--parsable",
            "--job-name",
            job["name"],
        ]
        dependencies = [f"afterok:{job_id}" for job_id in afterok]
        if index >= concurrency:
            dependencies.append(f"afterany:{submitted[index - concurrency]['job_id']}")
        if dependencies:
            command += ["--dependency", ",".join(dependencies)]
        log_root = Path(plan["config"]["output_root"]) / "logs"
        command += ["--output", str(log_root / "%x-%j.out")]
        command += [str(script), *job["arguments"]]
        job_id = subprocess.run(
            command, cwd=root, check=True, text=True, capture_output=True
        ).stdout.strip().split(";")[0]
        submitted.append({"name": job["name"], "job_id": job_id})
    return submitted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tears_training.orchestrate")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--phase", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--summaries", type=Path)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--rung", type=int, choices=(5, 20, 60, 200, 300), default=5)
    parser.add_argument("--previous-results", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--recvae-checkpoint-template")
    parser.add_argument("--slurm-profile", choices=tuple(SLURM_PROFILES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-submit", action="store_true")
    parser.add_argument("--promotion-approval", type=Path)
    parser.add_argument("--afterok", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path.cwd()
    config = load_config(args.config)
    if args.slurm_profile is None:
        args.slurm_profile = "full_4gpu" if args.phase == "full" else "pilot"
    plan = build_execution_plan(args, config, root)
    report = {"preflight": preflight(root, config), "execution_plan": plan}
    output = args.output or config.output_root / "manifests" / f"{args.phase}-{plan['fingerprint'][:12]}.json"
    if args.submit:
        report["submitted"] = submit_plan(
            plan,
            root,
            args.confirm_submit,
            args.promotion_approval,
            tuple(args.afterok),
        )
        report["preflight"]["slurm_submission_performed"] = True
        report["execution_plan"]["manual_gates"]["slurm_submission"] = "performed"
    if not args.dry_run:
        report["output"] = str(output)
        atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
