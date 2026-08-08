from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from .artifacts import atomic_write_json, sha256_file, stable_hash
from .config import ExperimentConfig, load_config


REQUIRED_FILES = ("ratings.csv", "movies.csv", "links.csv", "tags.csv")


def parse_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) == 2:
                checksums[Path(parts[1]).name] = parts[0]
    return checksums


def md5_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_data(config: ExperimentConfig) -> dict[str, object]:
    raw = config.raw_data
    if not raw.is_dir():
        raise FileNotFoundError(f"ML-32M directory does not exist: {raw}")
    expected = parse_checksums(raw / "checksums.txt")
    files: dict[str, object] = {}
    for name in REQUIRED_FILES:
        path = raw / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = md5_file(path)
        wanted = expected.get(name)
        if wanted and actual != wanted:
            raise RuntimeError(f"MD5 mismatch for {path}: {actual} != {wanted}")
        files[name] = {
            "bytes": path.stat().st_size,
            "md5": actual,
            "sha256": sha256_file(path),
        }
    ratings_header = pd.read_csv(raw / "ratings.csv", nrows=0).columns.tolist()
    movies_header = pd.read_csv(raw / "movies.csv", nrows=0).columns.tolist()
    if ratings_header != ["userId", "movieId", "rating", "timestamp"]:
        raise RuntimeError(f"Unexpected ratings schema: {ratings_header}")
    if movies_header != ["movieId", "title", "genres"]:
        raise RuntimeError(f"Unexpected movies schema: {movies_header}")
    return {"raw_data": str(raw), "files": files}


def _activity_band(count: int, boundaries: tuple[int, ...]) -> str:
    for lower, upper in zip(boundaries, boundaries[1:]):
        if lower <= count < upper:
            return f"{lower}-{upper - 1}"
    return f"{boundaries[-1]}+"


def _proportional_quotas(sizes: dict[str, int], total: int) -> dict[str, int]:
    population = sum(sizes.values())
    raw = {key: total * size / population for key, size in sizes.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(sizes, key=lambda key: (raw[key] - quotas[key], sizes[key]), reverse=True)
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def stratified_sample(
    users: pd.DataFrame,
    count: int,
    seed: int,
    excluded: set[int] | None = None,
) -> np.ndarray:
    available = users[~users.userId.isin(excluded or set())]
    sizes = available.groupby("activity_band").size().to_dict()
    quotas = _proportional_quotas(sizes, count)
    chosen: list[np.ndarray] = []
    for offset, band in enumerate(sorted(quotas)):
        values = available.loc[available.activity_band == band, "userId"].to_numpy()
        rng = np.random.default_rng(seed + offset)
        rng.shuffle(values)
        if quotas[band] > len(values):
            raise RuntimeError(f"Not enough users in activity band {band}")
        chosen.append(values[: quotas[band]])
    result = np.concatenate(chosen).astype(np.int64)
    np.random.default_rng(seed + 10_000).shuffle(result)
    return result


def scan_user_counts(config: ExperimentConfig) -> pd.DataFrame:
    counts: Counter[int] = Counter()
    path = config.raw_data / "ratings.csv"
    for chunk in pd.read_csv(path, usecols=["userId"], chunksize=1_000_000):
        counts.update(chunk.userId.value_counts().to_dict())
    users = pd.DataFrame({"userId": list(counts), "interaction_count": list(counts.values())})
    users["activity_band"] = users.interaction_count.map(
        lambda value: _activity_band(int(value), config.data.activity_bands)
    )
    return users.sort_values("userId").reset_index(drop=True)


def make_user_splits(config: ExperimentConfig, users: pd.DataFrame) -> pd.DataFrame:
    expected = config.data.train_users + config.data.validation_users + config.data.test_users
    if len(users) != expected:
        raise RuntimeError(f"Expected {expected} users, found {len(users)}")
    validation = stratified_sample(users, config.data.validation_users, config.data.split_seed)
    used = set(map(int, validation))
    test = stratified_sample(users, config.data.test_users, config.data.split_seed + 1, used)
    labels = pd.Series("train", index=users.index)
    labels.loc[users.userId.isin(validation)] = "validation"
    labels.loc[users.userId.isin(test)] = "test"
    result = users.copy()
    result["split"] = labels
    counts = result.split.value_counts().to_dict()
    wanted = {
        "train": config.data.train_users,
        "validation": config.data.validation_users,
        "test": config.data.test_users,
    }
    if counts != wanted:
        raise AssertionError(f"Split counts differ: {counts} != {wanted}")
    return result


def make_pilot_cohort(config: ExperimentConfig, splits: pd.DataFrame) -> pd.DataFrame:
    requested = {
        "train": config.data.pilot_train_users,
        "validation": config.data.pilot_validation_users,
        "test": config.data.pilot_test_users,
    }
    selected: list[pd.DataFrame] = []
    for offset, (split, count) in enumerate(requested.items()):
        candidates = splits[splits.split == split]
        ids = stratified_sample(candidates, count, config.data.split_seed + 100 + offset)
        part = candidates[candidates.userId.isin(ids)].copy()
        part["pilot_split"] = split
        selected.append(part)
    return pd.concat(selected).sort_values(["pilot_split", "userId"]).reset_index(drop=True)


def scan_training_item_support(
    config: ExperimentConfig, train_users: set[int]
) -> Counter[int]:
    counts: Counter[int] = Counter()
    path = config.raw_data / "ratings.csv"
    for chunk in pd.read_csv(path, usecols=["userId", "movieId"], chunksize=1_000_000):
        chunk = chunk[chunk.userId.isin(train_users)]
        counts.update(chunk.movieId.value_counts().to_dict())
    return counts


def build_catalogs(
    config: ExperimentConfig, support: Counter[int], output: Path
) -> dict[str, object]:
    movies = pd.read_csv(config.raw_data / "movies.csv")
    total_ratings = sum(support.values())
    candidates: dict[str, object] = {}
    for minimum in config.data.item_support_candidates:
        movie_ids = sorted(movie for movie, count in support.items() if count >= minimum)
        frame = movies[movies.movieId.isin(movie_ids)].copy().sort_values("movieId")
        frame["modelItemId"] = np.arange(len(frame), dtype=np.int64)
        frame["genres"] = frame.genres.replace("(no genres listed)", "Unknown")
        path = output / f"catalog_support_{minimum}.csv"
        frame.to_csv(path, index=False)
        retained = sum(support[movie] for movie in movie_ids)
        candidates[str(minimum)] = {
            "movies": len(frame),
            "ratings": retained,
            "rating_fraction": retained / total_ratings,
            "path": str(path),
            "sha256": sha256_file(path),
        }
    return candidates


def _chronological_eval_rows(
    frame: pd.DataFrame, observed_fraction: float, positive_rating: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed: list[pd.DataFrame] = []
    targets: list[pd.DataFrame] = []
    for _, history in frame.groupby("userId", sort=False):
        history = history.sort_values(["timestamp", "movieId"], kind="stable")
        boundary = max(1, min(len(history) - 1, int(np.floor(len(history) * observed_fraction))))
        observed.append(history.iloc[:boundary])
        held_out = history.iloc[boundary:]
        targets.append(held_out.loc[held_out.rating >= positive_rating])
    return pd.concat(observed, ignore_index=True), pd.concat(targets, ignore_index=True)


def _atomic_save_sparse(path: Path, matrix: sparse.csr_matrix) -> None:
    temporary = path.with_name(f".{path.name}.tmp.npz")
    sparse.save_npz(temporary, matrix, compressed=True)
    temporary.replace(path)


def _save_mmap_csr(directory: Path, name: str, matrix: sparse.csr_matrix) -> None:
    """Persist CSR components as .npy files that can be opened with mmap_mode."""
    component_root = directory / f"{name}_csr"
    component_root.mkdir(parents=True, exist_ok=True)
    for component, values in (
        ("data", matrix.data),
        ("indices", matrix.indices),
        ("indptr", matrix.indptr),
        ("shape", np.asarray(matrix.shape, dtype=np.int64)),
    ):
        path = component_root / f"{component}.npy"
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        temporary.replace(path)


def _chunk_matrix(
    frame: pd.DataFrame,
    user_map: dict[int, int],
    item_map: dict[int, int],
    shape: tuple[int, int],
) -> sparse.csr_matrix:
    if frame.empty:
        return sparse.csr_matrix(shape, dtype=np.float32)
    rows = frame.userId.map(user_map).to_numpy(np.int64)
    columns = frame.movieId.map(item_map).to_numpy(np.int64)
    return sparse.csr_matrix(
        (frame.rating.to_numpy(np.float32), (rows, columns)), shape=shape
    )


def build_sparse_profile(
    config: ExperimentConfig,
    splits: pd.DataFrame,
    catalog_path: Path,
    output: Path,
    restrict_users: set[int] | None = None,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(catalog_path)
    item_map = dict(zip(catalog.movieId.astype(int), catalog.modelItemId.astype(int)))
    split_map = dict(zip(splits.userId.astype(int), splits.split))
    if restrict_users is not None:
        split_map = {user: split for user, split in split_map.items() if user in restrict_users}
    user_ids = sorted(split_map)
    user_map = {user: index for index, user in enumerate(user_ids)}
    shape = (len(user_ids), len(catalog))
    train_observed = sparse.csr_matrix(shape, dtype=np.float32)
    histories: dict[str, dict[int, list[tuple[int, int, float]]]] = {
        "validation": {},
        "test": {},
    }
    ratings_count = 0
    for chunk in pd.read_csv(config.raw_data / "ratings.csv", chunksize=1_000_000):
        chunk = chunk[chunk.userId.isin(user_map) & chunk.movieId.isin(item_map)]
        if chunk.empty:
            continue
        ratings_count += len(chunk)
        chunk["split"] = chunk.userId.map(split_map)
        training = chunk.loc[chunk.split == "train"]
        if not training.empty:
            train_observed = train_observed + _chunk_matrix(
                training, user_map, item_map, shape
            )
        for split in ("validation", "test"):
            part = chunk.loc[chunk.split == split]
            for row in part.itertuples(index=False):
                histories[split].setdefault(int(row.userId), []).append(
                    (int(row.timestamp), int(row.movieId), float(row.rating))
                )
    train_observed.sum_duplicates()
    train_observed.sort_indices()
    matrices: dict[str, sparse.csr_matrix] = {}
    eligibility: dict[str, int] = {}
    matrices["train_observed"] = train_observed
    matrices["train_target"] = sparse.csr_matrix(shape, dtype=np.float32)
    for split in ("validation", "test"):
        observed_rows: list[int] = []
        observed_columns: list[int] = []
        observed_values: list[float] = []
        target_rows: list[int] = []
        target_columns: list[int] = []
        target_values: list[float] = []
        for user, history in histories[split].items():
            ordered = sorted(history, key=lambda value: (value[0], value[1]))
            boundary = max(
                1,
                min(
                    len(ordered) - 1,
                    int(np.floor(len(ordered) * config.data.observed_fraction)),
                ),
            )
            model_user = user_map[user]
            for _, movie, rating in ordered[:boundary]:
                observed_rows.append(model_user)
                observed_columns.append(item_map[movie])
                observed_values.append(rating)
            for _, movie, rating in ordered[boundary:]:
                if rating >= config.data.positive_rating:
                    target_rows.append(model_user)
                    target_columns.append(item_map[movie])
                    target_values.append(rating)
        matrices[f"{split}_observed"] = sparse.csr_matrix(
            (observed_values, (observed_rows, observed_columns)), shape=shape, dtype=np.float32
        )
        matrices[f"{split}_target"] = sparse.csr_matrix(
            (target_values, (target_rows, target_columns)), shape=shape, dtype=np.float32
        )
        split_rows = np.asarray(
            [user_map[user] for user, value in split_map.items() if value == split],
            dtype=np.int64,
        )
        target_counts = np.diff(matrices[f"{split}_target"].indptr)[split_rows]
        eligibility[split] = int((target_counts > 0).sum())
        overlap = matrices[f"{split}_observed"].multiply(matrices[f"{split}_target"])
        if overlap.nnz:
            raise AssertionError(f"Observed/target leakage detected in {split}")
    matrix_files: dict[str, dict[str, object]] = {}
    for name, matrix in matrices.items():
        matrix.eliminate_zeros()
        matrix.sort_indices()
        path = output / f"{name}.npz"
        _atomic_save_sparse(path, matrix)
        _save_mmap_csr(output, name, matrix)
        matrix_files[name] = {
            "nnz": int(matrix.nnz),
            "sha256": sha256_file(path),
        }
    pd.DataFrame(
        {"userId": user_ids, "modelUserId": range(len(user_ids)), "split": [split_map[u] for u in user_ids]}
    ).to_csv(output / "users.csv", index=False)
    shutil.copy2(catalog_path, output / "catalog.csv")
    manifest = {
        "users": len(user_ids),
        "items": len(catalog),
        "ratings": ratings_count,
        "eligible_evaluation_users": eligibility,
        "matrices": matrix_files,
        "catalog_sha256": sha256_file(output / "catalog.csv"),
        "user_mapping_sha256": sha256_file(output / "users.csv"),
    }
    manifest["fingerprint"] = stable_hash(manifest)
    atomic_write_json(output / "manifest.json", manifest)
    return manifest


def prepare(
    config: ExperimentConfig,
    profile: str,
    item_support: int | None,
    cohort_users: Path | None = None,
) -> dict[str, object]:
    output = config.output_root / "datasets" / profile
    output.mkdir(parents=True, exist_ok=True)
    users = scan_user_counts(config)
    splits = make_user_splits(config, users)
    splits.to_csv(output / "user_splits.csv", index=False)
    pilot = make_pilot_cohort(config, splits)
    pilot.to_csv(output / "pilot_users.csv", index=False)
    train_users = set(splits.loc[splits.split == "train", "userId"].astype(int))
    support = scan_training_item_support(config, train_users)
    catalogs = build_catalogs(config, support, output)
    result: dict[str, object] = {
        "profile": profile,
        "user_splits_sha256": sha256_file(output / "user_splits.csv"),
        "pilot_users_sha256": sha256_file(output / "pilot_users.csv"),
        "catalogs": catalogs,
    }
    if profile == "smoke":
        ranked = [movie for movie, _ in support.most_common(config.data.smoke_items)]
        movies = pd.read_csv(config.raw_data / "movies.csv")
        smoke_catalog = movies[movies.movieId.isin(ranked)].copy()
        order = {movie: index for index, movie in enumerate(ranked)}
        smoke_catalog["modelItemId"] = smoke_catalog.movieId.map(order)
        smoke_catalog["genres"] = smoke_catalog.genres.replace("(no genres listed)", "Unknown")
        smoke_catalog = smoke_catalog.sort_values("modelItemId")
        catalog_path = output / "catalog_smoke_4000.csv"
        smoke_catalog.to_csv(catalog_path, index=False)
        cohort = set(pilot.userId.astype(int))
        result["matrix"] = build_sparse_profile(
            config, splits, catalog_path, output / "matrix", cohort
        )
    elif item_support is not None:
        key = str(item_support)
        if key not in catalogs:
            raise ValueError(f"item support must be one of {sorted(catalogs)}")
        matrix_output = output / f"support_{item_support}" / "matrix"
        if cohort_users is not None:
            if profile != "pilot":
                raise ValueError("--cohort-users is only valid for the pilot profile")
            explicit = pd.read_csv(cohort_users)
            if "userId" not in explicit:
                raise ValueError("cohort user file must contain userId")
            if explicit.userId.duplicated().any():
                raise ValueError("cohort user file contains duplicate userId values")
            cohort = set(explicit.userId.astype(int))
            known = set(splits.userId.astype(int))
            unknown = sorted(cohort - known)
            if unknown:
                raise ValueError(f"cohort contains unknown users; first={unknown[0]}")
            result["explicit_cohort"] = {
                "path": str(cohort_users.resolve()),
                "sha256": sha256_file(cohort_users),
                "users": len(cohort),
            }
        else:
            cohort = set(pilot.userId.astype(int)) if profile == "pilot" else None
        result["matrix"] = build_sparse_profile(
            config,
            splits,
            Path(catalogs[key]["path"]),  # type: ignore[index]
            matrix_output,
            cohort,
        )
    result["fingerprint"] = stable_hash(result)
    manifest_path = (
        output / f"support_{item_support}" / "prepare_manifest.json"
        if profile != "smoke" and item_support is not None
        else output / "prepare_manifest.json"
    )
    atomic_write_json(manifest_path, result)
    return result


def prepare_catalog_selection(config: ExperimentConfig) -> dict[str, object]:
    """Prepare the three Phase-4 full-user matrices without overwriting them."""
    output = config.output_root / "datasets" / "catalog_selection"
    output.mkdir(parents=True, exist_ok=True)
    users = scan_user_counts(config)
    splits = make_user_splits(config, users)
    splits.to_csv(output / "user_splits.csv", index=False)
    pilot = make_pilot_cohort(config, splits)
    pilot.to_csv(output / "pilot_users.csv", index=False)
    train_users = set(splits.loc[splits.split == "train", "userId"].astype(int))
    support = scan_training_item_support(config, train_users)
    catalogs = build_catalogs(config, support, output)
    matrices: dict[str, object] = {}
    for minimum in config.data.item_support_candidates:
        key = str(minimum)
        matrices[key] = build_sparse_profile(
            config,
            splits,
            Path(catalogs[key]["path"]),  # type: ignore[index]
            output / f"support_{minimum}" / "matrix",
        )
    result: dict[str, object] = {
        "phase": "catalog_selection",
        "profile": "full",
        "user_splits_sha256": sha256_file(output / "user_splits.csv"),
        "pilot_users_sha256": sha256_file(output / "pilot_users.csv"),
        "catalogs": catalogs,
        "matrices": matrices,
    }
    result["fingerprint"] = stable_hash(result)
    atomic_write_json(output / "prepare_manifest.json", result)
    return result


def ensure_layout(config: ExperimentConfig, dry_run: bool) -> dict[str, str]:
    landing = config.scratch_root / "new_dataset"
    link = landing / "ml-32m"
    directories = [
        config.output_root / name
        for name in ("manifests", "datasets", "summaries", "checkpoints", "metrics", "logs", "wandb", "reports")
    ]
    if not dry_run:
        landing.mkdir(parents=True, exist_ok=True)
        if link.exists() and link.resolve() != config.raw_data.resolve():
            raise RuntimeError(f"Dataset landing path points elsewhere: {link}")
        if not link.exists():
            link.symlink_to(config.raw_data, target_is_directory=True)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        config.paid_backup_root.mkdir(parents=True, exist_ok=True)
    return {"landing": str(link), "output_root": str(config.output_root)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tears_training.data")
    parser.add_argument("--config", type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--initialize-layout", action="store_true")
    verify.add_argument("--dry-run", action="store_true")
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--profile", choices=("smoke", "pilot", "full"), required=True)
    prepare_parser.add_argument("--item-support", type=int)
    prepare_parser.add_argument("--cohort-users", type=Path)
    prepare_parser.add_argument("--dry-run", action="store_true")
    catalog_selection = sub.add_parser("catalog-selection")
    catalog_selection.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.action == "verify":
        result = verify_raw_data(config)
        if args.initialize_layout:
            result["layout"] = ensure_layout(config, args.dry_run)
    elif args.dry_run:
        result = {
            "profile": getattr(args, "profile", "full"),
            "item_support": getattr(args, "item_support", None),
            "cohort_users": str(args.cohort_users.resolve()) if getattr(args, "cohort_users", None) else None,
            "raw_data": str(config.raw_data),
            "output": str(
                config.output_root
                / "datasets"
                / ("catalog_selection" if args.action == "catalog-selection" else args.profile)
            ),
        }
    else:
        ensure_layout(config, False)
        result = (
            prepare_catalog_selection(config)
            if args.action == "catalog-selection"
            else prepare(config, args.profile, args.item_support, args.cohort_users)
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
