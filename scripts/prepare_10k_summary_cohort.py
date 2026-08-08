from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tears_training.artifacts import atomic_write_json, sha256_file, stable_hash
from tears_training.config import load_config


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare(
    config_path: Path,
    dataset_dir: Path,
    calibration_plan_path: Path,
    output_dir: Path,
    seed: int,
) -> dict[str, object]:
    config = load_config(config_path)
    splits_path = dataset_dir / "user_splits.csv"
    pilot_path = dataset_dir / "pilot_users.csv"
    splits = pd.read_csv(splits_path)
    pilot = pd.read_csv(pilot_path)
    calibration = json.loads(calibration_plan_path.read_text(encoding="utf-8"))
    excluded = {int(record["user_id"]) for record in calibration["records"]}
    if len(excluded) != len(calibration["records"]):
        raise RuntimeError("Calibration plan contains duplicate user IDs")
    if int(pilot.userId.isin(excluded).sum()) != len(excluded):
        raise RuntimeError("Every calibration user must belong to the pilot cohort")

    remaining = pilot.loc[~pilot.userId.isin(excluded)].copy()
    pilot_ids = set(map(int, pilot.userId))
    replacements: list[pd.DataFrame] = []
    replacement_records: list[dict[str, object]] = []
    removed = pilot.loc[pilot.userId.isin(excluded)]
    for offset, ((split, band), count) in enumerate(
        sorted(removed.groupby(["split", "activity_band"]).size().items())
    ):
        candidates = splits.loc[
            (splits.split == split)
            & (splits.activity_band == band)
            & (~splits.userId.isin(pilot_ids))
        ].copy()
        values = candidates.userId.to_numpy(np.int64)
        np.random.default_rng(seed + offset).shuffle(values)
        if len(values) < int(count):
            raise RuntimeError(f"Not enough replacements for {split}/{band}")
        chosen = set(map(int, values[: int(count)]))
        part = candidates.loc[candidates.userId.isin(chosen)].copy()
        part["pilot_split"] = split
        replacements.append(part)
        replacement_records.append(
            {
                "split": split,
                "activity_band": band,
                "count": int(count),
                "user_ids": sorted(chosen),
            }
        )

    cohort = pd.concat([remaining, *replacements], ignore_index=True)
    cohort = cohort.sort_values("userId").reset_index(drop=True)
    if len(cohort) != 10_000 or cohort.userId.nunique() != 10_000:
        raise AssertionError("Exclusive summary cohort must contain 10,000 unique users")
    overlap = sorted(set(map(int, cohort.userId)) & excluded)
    if overlap:
        raise AssertionError(f"Calibration overlap detected: {overlap[:5]}")
    split_counts = cohort.split.value_counts().to_dict()
    if split_counts != {"train": 9_000, "validation": 500, "test": 500}:
        raise AssertionError(f"Unexpected cohort split counts: {split_counts}")

    users = cohort[["userId", "split"]].copy()
    users.insert(1, "modelUserId", np.arange(len(users), dtype=np.int64))
    users_path = output_dir / "users.csv"
    atomic_write_csv(users_path, users)

    catalog = pd.read_csv(config.raw_data / "movies.csv").sort_values("movieId").reset_index(drop=True)
    catalog["genres"] = catalog.genres.replace("(no genres listed)", "Unknown")
    catalog.insert(1, "modelItemId", np.arange(len(catalog), dtype=np.int64))
    catalog_path = output_dir / "catalog.csv"
    atomic_write_csv(catalog_path, catalog)

    manifest: dict[str, object] = {
        "seed": seed,
        "users": len(users),
        "unique_user_ids": int(users.userId.nunique()),
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "activity_band_counts": {
            key: int(value) for key, value in cohort.activity_band.value_counts().to_dict().items()
        },
        "calibration_plan": str(calibration_plan_path.resolve()),
        "calibration_plan_fingerprint": calibration["fingerprint"],
        "excluded_calibration_users": len(excluded),
        "excluded_user_ids_sha256": stable_hash(sorted(excluded)),
        "calibration_overlap": 0,
        "replacement_users": len(cohort) - len(remaining),
        "replacement_records": replacement_records,
        "sources": {
            "user_splits": {"path": str(splits_path.resolve()), "sha256": sha256_file(splits_path)},
            "pilot_users": {"path": str(pilot_path.resolve()), "sha256": sha256_file(pilot_path)},
        },
        "artifacts": {
            "users": {"path": str(users_path.resolve()), "sha256": sha256_file(users_path)},
            "catalog": {"path": str(catalog_path.resolve()), "sha256": sha256_file(catalog_path)},
        },
    }
    manifest["fingerprint"] = stable_hash(manifest)
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--calibration-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    result = prepare(
        args.config,
        args.dataset_dir,
        args.calibration_plan,
        args.output_dir,
        args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
