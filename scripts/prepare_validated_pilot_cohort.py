from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tears_training.artifacts import atomic_write_json, sha256_file, stable_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = [
        json.loads(line)
        for line in args.summaries.read_text(encoding="utf-8").splitlines()
        if line
    ]
    valid_ids = [int(row["user_id"]) for row in summaries]
    if len(valid_ids) != len(set(valid_ids)):
        raise RuntimeError("validated summaries contain duplicate user IDs")

    cohort = pd.read_csv(args.cohort)
    selected = cohort.loc[cohort.userId.astype(int).isin(valid_ids)].copy()
    if len(selected) != len(valid_ids):
        raise RuntimeError(
            f"validated/cohort mismatch: summaries={len(valid_ids)} cohort_rows={len(selected)}"
        )
    selected = selected.sort_values(["split", "userId"], kind="stable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    counts = {key: int(value) for key, value in selected.split.value_counts().items()}
    manifest = {
        "decision": "August 7 validated-summary pilot amendment",
        "users": len(selected),
        "split_counts": counts,
        "cohort_source": str(args.cohort.resolve()),
        "cohort_source_sha256": sha256_file(args.cohort),
        "summaries": str(args.summaries.resolve()),
        "summaries_sha256": sha256_file(args.summaries),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    manifest["fingerprint"] = stable_hash(manifest)
    atomic_write_json(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
