from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.finalize_phase6 import build_report, render_markdown
from tears_training.artifacts import sha256_file


def _write_run(root: Path, *, corrupt_hash: bool = False) -> None:
    run_dir = root / "recvae" / "seed-2020" / "run-id"
    run_dir.mkdir(parents=True)
    for name in ("best.pt", "final.pt", "resume.pt"):
        (run_dir / name).write_bytes(name.encode())
    hashes = {name: sha256_file(run_dir / name) for name in ("best.pt", "final.pt", "resume.pt")}
    if corrupt_hash:
        hashes["best.pt"] = "0" * 64
    manifest = {
        "fingerprint": "training",
        "execution_fingerprint": "execution",
        "wandb_run_id": "execution",
        "arguments": {
            "profile": "full",
            "model": "recvae",
            "seed": 2020,
            "epochs": 200,
            "minimum_epochs": 200,
            "patience": 201,
            "matrix_dir": "/matrix",
            "summaries": None,
        },
    }
    result = {
        "run_dir": str(run_dir),
        "epochs_completed": 200,
        "best_validation_selection_ndcg@50": 0.25,
        "fingerprint": "training",
        "execution_fingerprint": "execution",
        "checkpoint_sha256": hashes,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")


def _report(tmp_path: Path, *, corrupt_hash: bool = False) -> dict[str, object]:
    checkpoint_root = tmp_path / "checkpoints"
    _write_run(checkpoint_root, corrupt_hash=corrupt_hash)
    summary_path = tmp_path / "summaries.jsonl"
    summary_path.write_text("{}\n", encoding="utf-8")
    return build_report(
        checkpoint_root,
        summary_path,
        sha256_file(summary_path),
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        False,
        "entity",
        "project",
        models=("recvae",),
        seeds=(2020,),
    )


def test_build_report_passes_only_with_complete_hashed_run(tmp_path: Path) -> None:
    report = _report(tmp_path)

    assert report["phase6_complete"] is True
    assert report["phase7_ready"] is True
    assert report["valid_runs"] == 1
    assert report["test_metrics_used"] is False
    assert "**Status:** **PASS**" in render_markdown(report)


def test_build_report_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    report = _report(tmp_path, corrupt_hash=True)

    assert report["phase6_complete"] is False
    assert report["phase7_ready"] is False
    assert report["valid_runs"] == 0
    assert any("best.pt SHA-256 differs" in error for error in report["errors"])
