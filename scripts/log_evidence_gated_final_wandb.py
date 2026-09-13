#!/usr/bin/env python3
"""Log final evidence-gated audit metrics to the existing standard W&B run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    metadata = json.loads((root / "wandb_monitor_config.json").read_text())
    report = json.loads((root / "reports/final_audit_report.json").read_text())
    manual = report["manual_semantic_audit"]
    corrected = report["corrected_evidence_contract_review"]
    metrics = {
        "audit/manual_reviewed": manual["reviewed"],
        "audit/unsupported_negative_claims": manual["unsupported_negative_claims"],
        "audit/overstated_negative_claims": manual["overstated_negative_claims"],
        "audit/inappropriate_abstentions": manual["inappropriate_abstentions"],
        "audit/strict_positive_grounding_pass": manual[
            "strict_positive_genre_grounding_pass"
        ],
        "audit/strict_positive_grounding_fail": manual[
            "strict_positive_genre_grounding_fail"
        ],
        "audit/mixed_genre_positive_failures": manual[
            "mixed_genre_categorical_positive_failures"
        ],
        "audit/partial_negative_omissions_sample": manual[
            "supported_negative_evidence_omitted_users"
        ],
        "audit/partial_negative_omissions_all": corrected[
            "confirmed_partial_supported_negative_omissions"
        ]["users"],
        "audit/semantic_complete": manual["semantic_completeness"],
        "audit/semantic_richness_acceptable": manual[
            "semantic_richness_acceptable"
        ],
        "audit/promotion_pass": 0,
    }

    import wandb

    run = wandb.init(
        entity=metadata["entity"],
        project=metadata["project"],
        name=metadata["run_name"],
        id=metadata["run_id"],
        resume="allow",
        job_type=metadata["job_type"],
        mode=metadata["mode"],
        dir=str(root / "wandb"),
        config={"weave_enabled": False, "final_manual_audit": True},
    )
    run.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    run.summary["promotion/recommend_full_cohort"] = False
    run.summary["audit/final_report"] = str(root / "reports/final_audit_report.json")
    run.finish()
    print(json.dumps({"run_url": metadata["run_url"], "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
