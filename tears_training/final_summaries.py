"""Calibration-only generation for the final ML-32M Emiliano-prompt corpus.

This module deliberately has no full-cohort submission command.  It can select,
plan, submit, poll, and validate exactly one 100-user calibration batch.  Full
generation remains a separate, explicit promotion step that is not implemented
until the calibration and budget are approved.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    mirror_paid_artifact,
    sha256_file,
    stable_hash,
)
from .config import ExperimentConfig, load_config


PROTOCOL_VERSION = "ml32m-support20-emiliano-original-v1-gpt5mini-2025-08-07"
MODEL = "gpt-5-mini-2025-08-07"
CALIBRATION_USERS = 100
MAX_HISTORY_ITEMS = 50
MAX_OUTPUT_TOKENS = 450
MIN_WORDS = 120
MAX_WORDS = 260
SEED = 2024
EXPECTED_MATRIX_FINGERPRINT = (
    "27596ef71a4ac0f46e81ca97a40c4a9475bb1c2cc62c9db13819fe3cbeb714ce"
)
EXPECTED_CATALOG_SHA256 = (
    "ac8b369749f8305bd712ec011150ff57b21459fe48be052f515ce5e291e00961"
)
EXPECTED_USERS_SHA256 = (
    "a0bc4801c7c0ea940d3e3e304c5acdd46f9b8c9fc15870165bbe1d32c3b137a7"
)
SOURCE_NOTEBOOK_RELATIVE = Path("TEARS_Project/Code4Neda/prompt_gpt.ipynb")
EXPECTED_SOURCE_NOTEBOOK_SHA256 = (
    "7eb9afb84e027f7cf9b315e68801e498c2e08765c4461d5f7d460fa0488fc089"
)

# Extracted from the executed prompt construction cell in prompt_gpt.ipynb.
# Do not reflow, correct punctuation, or add instructions to this string.
EMILIANO_SYSTEM_PROMPT = (
    "Task: You will now help me generate a highly detailed summary based on the broad common elements of movies.\n"
    "Do not comment on the year of production. Do not mention any specific movie titles or actors.\n"
    "Do not comment on the ratings but use qualitative speech such as the user likes, or the user does not enjoy\n"
    "Remember you are an expert crafter of these summaries so any other expert should be able to craft a similar summary to yours given this task\n"
    "Keep the summary short at about 200 words. The summary should have the following format:\n"
    "Summary: {Specific details about genres the user enjoys}. {Specific details of plot points the user seems to enjoy}. "
    "{Specific details about genres the user does not enjoy}. {Specific details of plot points the user does not enjoy but other users may}."
)
EMILIANO_PROMPT_SHA256 = hashlib.sha256(
    EMILIANO_SYSTEM_PROMPT.encode("utf-8")
).hexdigest()
EXPECTED_EMILIANO_PROMPT_SHA256 = (
    "94c0db114eaf2c984dc9d156f2e4e2c37018577eef150ab76e3f2d348ecf164a"
)

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

GENRE_TERMS = {
    "action": ("action",),
    "adventure": ("adventure",),
    "animation": ("animation", "animated"),
    "children": ("children", "family", "family friendly"),
    "comedy": ("comedy", "comedies", "comic", "humor", "humour"),
    "crime": ("crime", "criminal"),
    "documentary": ("documentary", "documentaries"),
    "drama": ("drama", "dramas", "dramatic"),
    "fantasy": ("fantasy", "fantastical"),
    "film-noir": ("film noir", "noir"),
    "horror": ("horror",),
    "imax": ("imax",),
    "musical": ("musical", "musicals"),
    "mystery": ("mystery", "mysteries"),
    "romance": ("romance", "romantic"),
    "sci-fi": ("sci fi", "science fiction", "science-fiction"),
    "thriller": ("thriller", "thrillers", "suspense"),
    "war": ("war", "wartime"),
    "western": ("western", "westerns"),
}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count, path.stat().st_size


def _write_immutable_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError(f"Refusing to overwrite different artifact: {path}")
        return
    atomic_write_json(path, value)


def _write_immutable_jsonl(path: Path, rows: list[dict[str, Any]]) -> tuple[int, int]:
    encoded = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Refusing to overwrite different artifact: {path}")
        return len(rows), len(encoded)
    return _write_jsonl(path, rows)


def _proportional_quotas(sizes: dict[str, int], total: int) -> dict[str, int]:
    if total < 0 or total > sum(sizes.values()):
        raise ValueError("Invalid sample quota")
    if not sizes:
        return {}
    # Cover every nonempty activity band whenever the split quota permits it,
    # then allocate the remainder proportionally.
    quotas = {key: 1 if total >= len(sizes) else 0 for key in sizes}
    remaining = total - sum(quotas.values())
    if remaining < 0:
        quotas = {key: 0 for key in sizes}
        remaining = total
    capacity = {key: sizes[key] - quotas[key] for key in sizes}
    capacity_total = sum(capacity.values())
    raw = {
        key: (remaining * capacity[key] / capacity_total if capacity_total else 0)
        for key in sizes
    }
    floors = {key: int(raw[key]) for key in sizes}
    for key in sizes:
        quotas[key] += floors[key]
    leftover = total - sum(quotas.values())
    order = sorted(
        sizes,
        key=lambda key: (raw[key] - floors[key], capacity[key], key),
        reverse=True,
    )
    for key in order[:leftover]:
        quotas[key] += 1
    return quotas


def select_calibration_users(
    user_splits: pd.DataFrame,
    matrix_users: pd.DataFrame,
    count: int = CALIBRATION_USERS,
    seed: int = SEED,
) -> pd.DataFrame:
    required = {"userId", "interaction_count", "activity_band", "split"}
    if not required.issubset(user_splits.columns):
        raise RuntimeError(f"user_splits.csv is missing {sorted(required - set(user_splits))}")
    if set(user_splits.userId.astype(int)) != set(matrix_users.userId.astype(int)):
        raise RuntimeError("Frozen user_splits and matrix users contain different user IDs")
    if count != CALIBRATION_USERS:
        raise RuntimeError("This module may plan exactly 100 calibration users only")
    split_quotas = {"train": 90, "validation": 5, "test": 5}
    if sum(split_quotas.values()) != count:
        raise AssertionError("Calibration split quotas do not sum to 100")

    selected: list[pd.DataFrame] = []
    stratum_records: list[tuple[str, str, int]] = []
    offset = 0
    for split in ("train", "validation", "test"):
        available = user_splits.loc[user_splits.split == split]
        sizes = {
            str(key): int(value)
            for key, value in available.groupby("activity_band").size().items()
        }
        quotas = _proportional_quotas(sizes, split_quotas[split])
        for band in sorted(quotas):
            quota = quotas[band]
            candidates = available.loc[
                available.activity_band.astype(str) == band
            ].copy()
            values = candidates.userId.to_numpy(np.int64)
            np.random.default_rng(seed + offset).shuffle(values)
            offset += 1
            chosen = set(map(int, values[:quota]))
            part = candidates.loc[candidates.userId.isin(chosen)].copy()
            selected.append(part)
            stratum_records.append((split, band, len(part)))

    result = pd.concat(selected, ignore_index=True).sort_values("userId")
    if len(result) != count or result.userId.nunique() != count:
        raise AssertionError("Calibration must contain exactly 100 unique users")
    if result.split.value_counts().to_dict() != split_quotas:
        raise AssertionError("Calibration split quotas changed")
    if any(actual <= 0 for _, _, actual in stratum_records):
        raise AssertionError("A selected calibration stratum is empty")
    return result.reset_index(drop=True)


def verify_frozen_matrix(matrix_dir: Path, dataset_dir: Path) -> dict[str, Any]:
    manifest_path = matrix_dir / "manifest.json"
    users_path = matrix_dir / "users.csv"
    catalog_path = matrix_dir / "catalog.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "fingerprint": manifest.get("fingerprint") == EXPECTED_MATRIX_FINGERPRINT,
        "users": manifest.get("users") == 200_948,
        "items": manifest.get("items") == 22_343,
        "catalog_manifest": manifest.get("catalog_sha256")
        == EXPECTED_CATALOG_SHA256,
        "catalog_file": sha256_file(catalog_path) == EXPECTED_CATALOG_SHA256,
        "users_file": sha256_file(users_path) == EXPECTED_USERS_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen support-20 matrix verification failed: {checks}")
    split_path = dataset_dir / "user_splits.csv"
    return {
        "matrix_dir": str(matrix_dir.resolve()),
        "matrix_manifest_path": str(manifest_path.resolve()),
        "matrix_manifest_sha256": sha256_file(manifest_path),
        "matrix_fingerprint": manifest["fingerprint"],
        "users": manifest["users"],
        "items": manifest["items"],
        "users_path": str(users_path.resolve()),
        "users_sha256": sha256_file(users_path),
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "user_splits_path": str(split_path.resolve()),
        "user_splits_sha256": sha256_file(split_path),
    }


def load_histories(
    config: ExperimentConfig,
    matrix_dir: Path,
    selected: pd.DataFrame,
) -> dict[int, list[tuple[int, int, str, float, str]]]:
    wanted = set(map(int, selected.userId))
    split_by_user = {
        int(row.userId): str(row.split) for row in selected.itertuples(index=False)
    }
    catalog = pd.read_csv(
        matrix_dir / "catalog.csv", usecols=["movieId", "title", "genres"]
    )
    evidence = {
        int(row.movieId): (str(row.title), str(row.genres))
        for row in catalog.itertuples(index=False)
    }
    histories: dict[int, list[tuple[int, int, str, float, str]]] = {
        user_id: [] for user_id in wanted
    }
    for chunk in pd.read_csv(config.raw_data / "ratings.csv", chunksize=1_000_000):
        part = chunk[chunk.userId.isin(wanted) & chunk.movieId.isin(evidence)]
        for row in part.itertuples(index=False):
            title, genres = evidence[int(row.movieId)]
            histories[int(row.userId)].append(
                (int(row.timestamp), int(row.movieId), title, float(row.rating), genres)
            )

    result: dict[int, list[tuple[int, int, str, float, str]]] = {}
    for user_id in sorted(wanted):
        chronological = sorted(histories[user_id], key=lambda item: (item[0], item[1]))
        if split_by_user[user_id] in {"validation", "test"}:
            boundary = max(
                1,
                min(
                    len(chronological) - 1,
                    int(np.floor(len(chronological) * config.data.observed_fraction)),
                ),
            )
            chronological = chronological[:boundary]
        # Emiliano's prompt shows the most recent title first.  movieId is a
        # deterministic tie-breaker where timestamps are equal.
        recent_newest_first = list(reversed(chronological[-MAX_HISTORY_ITEMS:]))
        if not recent_newest_first:
            raise RuntimeError(f"Calibration user {user_id} has no observed history")
        result[user_id] = recent_newest_first
    return result


def render_emiliano_history(
    history: Iterable[tuple[int, int, str, float, str]],
) -> str:
    prompt = ""
    for _, _, title, rating, genres in history:
        prompt += f"\n{title}"
        prompt += f"\nRating: {rating}\n"
        prompt += f"\\Genres: {genres}\n"
    return prompt


def protocol_manifest(repository_root: Path | None = None) -> dict[str, Any]:
    if EMILIANO_PROMPT_SHA256 != EXPECTED_EMILIANO_PROMPT_SHA256:
        raise AssertionError("Emiliano prompt bytes changed")
    source_path = (
        (repository_root / SOURCE_NOTEBOOK_RELATIVE).resolve()
        if repository_root is not None
        else SOURCE_NOTEBOOK_RELATIVE
    )
    if repository_root is not None:
        source_sha256 = sha256_file(source_path)
        if source_sha256 != EXPECTED_SOURCE_NOTEBOOK_SHA256:
            raise RuntimeError("Original Emiliano notebook bytes changed")
    else:
        source_sha256 = EXPECTED_SOURCE_NOTEBOOK_SHA256
    value: dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "label": "Emiliano prompt + newer OpenAI model",
        "source": {
            "repository_path": str(source_path),
            "notebook_sha256": source_sha256,
            "prompt_construction_cell_index": 3,
            "prompt_construction_execution_count": 2,
            "request_cell_index": 5,
            "request_cell_execution_count": 4,
            "original_executed_api": "openai.ChatCompletion.create",
            "original_default_model": "gpt-4-turbo",
            "original_max_tokens": 300,
            "original_temperature": 0,
            "original_seed": 0,
        },
        "system_prompt": EMILIANO_SYSTEM_PROMPT,
        "system_prompt_sha256": EMILIANO_PROMPT_SHA256,
        "history": {
            "maximum_items": MAX_HISTORY_ITEMS,
            "order": "timestamp_descending_movieId_descending_tiebreak",
            "template": "\\n{title}\\nRating: {rating}\\n\\\\Genres: {genres}\\n",
            "validation_test_scope": "frozen_observed_70_percent_only",
        },
        "generation": {
            "model": MODEL,
            "endpoint": "/v1/responses",
            "reasoning_effort": "minimal",
            "temperature": None,
            "seed": None,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "structured_output": SUMMARY_SCHEMA,
            "text_verbosity": "low",
            "store": False,
            "omitted_parameters": {
                "temperature": "not sent to the selected GPT-5 reasoning model",
                "seed": "not sent by the Responses API calibration protocol",
            },
        },
        "validation": {
            "word_range": [MIN_WORDS, MAX_WORDS],
            "requires_literal_summary_prefix": True,
            "requires_exact_sentence_count": False,
            "requires_four_content_categories": True,
            "privacy": ["movie_title", "year", "numeric_rating"],
            "cross_user_duplicate_check": "exact normalized text",
            "near_duplicate_check": {
                "sequence_match_threshold": 0.92,
                "token_jaccard_threshold": 0.85,
            },
            "grounding_check": "per-user supplied-history genre evidence plus manual theme/plot review",
            "normative_note": "No exact sentence-count rule is applied.",
        },
    }
    value["fingerprint"] = stable_hash(value)
    return value


def response_request(
    user_id: int,
    split: str,
    activity_band: str,
    history: list[tuple[int, int, str, float, str]],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    history_text = render_emiliano_history(history)
    history_hash = hashlib.sha256(history_text.encode("utf-8")).hexdigest()
    cache_key = stable_hash(
        {
            "user_id": user_id,
            "history_hash": history_hash,
            "protocol_fingerprint": protocol["fingerprint"],
        }
    )
    custom_id = f"cal-user-{user_id}-{cache_key[:16]}-a0"
    body = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "input": [
                {"role": "system", "content": EMILIANO_SYSTEM_PROMPT},
                {"role": "user", "content": history_text},
            ],
            "reasoning": {"effort": "minimal"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "viewer_profile",
                    "strict": True,
                    "schema": SUMMARY_SCHEMA,
                },
                "verbosity": "low",
            },
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "store": False,
        },
    }
    record = {
        "user_id": user_id,
        "split": split,
        "activity_band": activity_band,
        "custom_id": custom_id,
        "cache_key": cache_key,
        "history_hash": history_hash,
        "history_items": len(history),
        "movie_ids": [item[1] for item in history],
        "titles": [item[2] for item in history],
        "ratings": [item[3] for item in history],
        "genres": [item[4] for item in history],
    }
    return body, record


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4 * 1.20))


def plan_calibration(
    config_path: Path,
    matrix_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    seed: int = SEED,
) -> dict[str, Any]:
    config = load_config(config_path)
    frozen = verify_frozen_matrix(matrix_dir, dataset_dir)
    matrix_users = pd.read_csv(matrix_dir / "users.csv")
    user_splits = pd.read_csv(dataset_dir / "user_splits.csv")
    selected = select_calibration_users(user_splits, matrix_users, seed=seed)
    histories = load_histories(config, matrix_dir, selected)
    repository_root = Path(__file__).resolve().parents[1]
    protocol = protocol_manifest(repository_root)

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        request, record = response_request(
            int(row.userId),
            str(row.split),
            str(row.activity_band),
            histories[int(row.userId)],
            protocol,
        )
        rows.append(request)
        records.append(record)
    if len(rows) != CALIBRATION_USERS:
        raise AssertionError("Paid calibration plan is not exactly 100 requests")
    if len({row["custom_id"] for row in rows}) != CALIBRATION_USERS:
        raise AssertionError("Paid calibration request IDs are not unique")

    output_dir = output_dir.resolve()
    requests_path = output_dir / "requests" / "calibration-100.jsonl"
    count, size = _write_immutable_jsonl(requests_path, rows)
    request_sha256 = sha256_file(requests_path)
    estimated_input = sum(
        estimate_tokens(
            row["body"]["input"][0]["content"]
            + row["body"]["input"][1]["content"]
        )
        for row in rows
    )
    plan: dict[str, Any] = {
        "scope": "calibration_only",
        "expected_requests": CALIBRATION_USERS,
        "requests": count,
        "unique_users": len({record["user_id"] for record in records}),
        "seed": seed,
        "protocol": protocol,
        "dataset": frozen,
        "sample": {
            "split_counts": {
                key: int(value) for key, value in selected.split.value_counts().items()
            },
            "activity_band_counts": {
                key: int(value)
                for key, value in selected.activity_band.value_counts().items()
            },
            "stratum_counts": {
                f"{split}/{band}": int(value)
                for (split, band), value in selected.groupby(
                    ["split", "activity_band"]
                ).size().items()
            },
            "user_ids": [int(value) for value in selected.userId],
            "user_ids_sha256": stable_hash([int(value) for value in selected.userId]),
        },
        "request_file": {
            "path": str(requests_path),
            "requests": count,
            "bytes": size,
            "sha256": request_sha256,
        },
        "records": records,
        "estimated_input_tokens": estimated_input,
        "reserved_output_tokens": CALIBRATION_USERS * MAX_OUTPUT_TOKENS,
    }
    plan["fingerprint"] = stable_hash(plan)
    _write_immutable_json(output_dir / "request_plan.json", plan)
    _write_immutable_json(output_dir / "protocol.json", protocol)
    prompt_path = output_dir / "prompt" / "emiliano_system_prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_bytes = EMILIANO_SYSTEM_PROMPT.encode("utf-8")
    if prompt_path.exists() and prompt_path.read_bytes() != prompt_bytes:
        raise RuntimeError(f"Refusing to overwrite different prompt artifact: {prompt_path}")
    if not prompt_path.exists():
        atomic_write_bytes(prompt_path, prompt_bytes)
    _write_immutable_json(
        output_dir / "prompt" / "provenance.json",
        {
            "protocol_version": protocol["version"],
            "protocol_fingerprint": protocol["fingerprint"],
            "prompt_sha256": EMILIANO_PROMPT_SHA256,
            "prompt_path": str(prompt_path),
            "source": protocol["source"],
            "system_message": EMILIANO_SYSTEM_PROMPT,
            "user_history_construction": protocol["history"],
            "generation": protocol["generation"],
            "validation": protocol["validation"],
            "security_note": "The source notebook is referenced and hashed, not copied, because it contains a historical embedded credential.",
        },
    )
    try:
        import openai

        openai_version = openai.__version__
    except (ImportError, AttributeError):
        openai_version = None
    _write_immutable_json(
        output_dir / "runtime_environment.json",
        {
            "python_version": sys.version,
            "openai_sdk_version": openai_version,
            "api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
            "api_key_value_recorded": False,
            "repository_root": str(repository_root),
        },
    )
    _write_immutable_json(
        output_dir / "calibration_user_ids.json",
        {
            "seed": seed,
            "count": CALIBRATION_USERS,
            "user_ids": plan["sample"]["user_ids"],
            "user_ids_sha256": plan["sample"]["user_ids_sha256"],
            "split_counts": plan["sample"]["split_counts"],
            "activity_band_counts": plan["sample"]["activity_band_counts"],
            "stratum_counts": plan["sample"]["stratum_counts"],
        },
    )
    return plan


def _load_verified_plan(plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    fingerprint = plan.pop("fingerprint", None)
    actual = stable_hash(plan)
    plan["fingerprint"] = fingerprint
    if fingerprint != actual:
        raise RuntimeError("Calibration plan fingerprint mismatch")
    if (
        plan.get("scope") != "calibration_only"
        or plan.get("requests") != CALIBRATION_USERS
        or plan.get("unique_users") != CALIBRATION_USERS
        or plan.get("expected_requests") != CALIBRATION_USERS
    ):
        raise RuntimeError("Refusing a plan that is not exactly 100 calibration users")
    request_file = plan["request_file"]
    request_path = Path(request_file["path"])
    if sha256_file(request_path) != request_file["sha256"]:
        raise RuntimeError("Calibration request file hash mismatch")
    if sum(1 for _ in request_path.open("r", encoding="utf-8")) != CALIBRATION_USERS:
        raise RuntimeError("Calibration request file does not contain exactly 100 lines")
    return plan


def projected_reserved_cost(
    plan: dict[str, Any], input_price: float, output_price: float
) -> float:
    return (
        plan["estimated_input_tokens"] * input_price
        + plan["reserved_output_tokens"] * output_price
    ) / 1_000_000


def submit_calibration(
    plan_path: Path,
    artifact_mirror_root: Path | None,
    input_price: float,
    cached_input_price: float,
    output_price: float,
    max_cost: float,
    confirmed: bool,
) -> dict[str, Any]:
    if not confirmed:
        raise RuntimeError("Paid calibration requires --confirm-calibration-spend")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required through the environment")
    plan = _load_verified_plan(plan_path)
    root = plan_path.parent
    submission_path = root / "submission.json"
    if submission_path.exists():
        raise RuntimeError(
            "Calibration was already submitted; refusing a duplicate paid request"
        )
    projection = projected_reserved_cost(plan, input_price, output_price)
    if projection <= 0 or projection > max_cost:
        raise RuntimeError(
            f"Calibration projection ${projection:.6f} exceeds ${max_cost:.6f} cap"
        )

    from openai import OpenAI

    request_path = Path(plan["request_file"]["path"])
    if artifact_mirror_root is not None:
        relative = Path("final_calibration") / plan["fingerprint"]
        mirror_paid_artifact(plan_path, artifact_mirror_root, relative / plan_path.name)
        mirror_paid_artifact(
            root / "protocol.json", artifact_mirror_root, relative / "protocol.json"
        )
        mirror_paid_artifact(
            root / "calibration_user_ids.json",
            artifact_mirror_root,
            relative / "calibration_user_ids.json",
        )
        mirror_paid_artifact(
            request_path,
            artifact_mirror_root,
            relative / "requests" / request_path.name,
        )

    client = OpenAI()
    with request_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "scope": "calibration_only",
            "plan": plan["fingerprint"],
            "protocol": plan["protocol"]["fingerprint"],
        },
    )
    result = {
        "scope": "calibration_only",
        "plan": plan["fingerprint"],
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "status": batch.status,
        "submitted_requests": CALIBRATION_USERS,
        "pricing_usd_per_million": {
            "input": input_price,
            "cached_input": cached_input_price,
            "output": output_price,
        },
        "pricing_source": "https://developers.openai.com/api/docs/models/gpt-5-mini",
        "pricing_verified_at": datetime.now(timezone.utc).isoformat(),
        "projected_reserved_cost_usd": projection,
        "maximum_authorized_calibration_cost_usd": max_cost,
        "submitted_at": int(time.time()),
    }
    atomic_write_json(submission_path, result)
    if artifact_mirror_root is not None:
        mirror_paid_artifact(
            submission_path, artifact_mirror_root, relative / submission_path.name
        )
    return result


def poll_calibration(
    submission_path: Path, artifact_mirror_root: Path | None
) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required through the environment")
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    if submission.get("scope") != "calibration_only":
        raise RuntimeError("Refusing to poll a non-calibration submission")
    from openai import OpenAI

    client = OpenAI()
    batch = client.batches.retrieve(submission["batch_id"])
    status = batch.model_dump(mode="json")
    root = submission_path.parent
    relative = Path("final_calibration") / submission["plan"]
    for kind, file_id in (
        ("responses", batch.output_file_id),
        ("errors", batch.error_file_id),
    ):
        if not file_id:
            continue
        payload = client.files.content(file_id).content
        path = root / kind / f"{batch.id}.jsonl"
        if path.exists() and path.read_bytes() != payload:
            raise RuntimeError(f"Downloaded {kind} changed for {batch.id}")
        if not path.exists():
            atomic_write_bytes(path, payload)
        if artifact_mirror_root is not None:
            mirror_paid_artifact(
                path, artifact_mirror_root, relative / kind / path.name
            )
    usage = status.get("usage") or {}
    prices = submission["pricing_usd_per_million"]
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    provisional_cost = (
        input_tokens * prices["input"] + output_tokens * prices["output"]
    ) / 1_000_000
    result = {
        "scope": "calibration_only",
        "plan": submission["plan"],
        "batch": status,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "provisional_batch_level_cost_usd": provisional_cost,
        "cost_note": "Exact cost is calculated from per-response usage, including cached input, during validation.",
        "polled_at": int(time.time()),
    }
    atomic_write_json(root / "poll.json", result)
    return result


def extract_output_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str):
        return body["output_text"].strip()
    parts: list[str] = []
    for output in body.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                parts.append(str(content.get("text", "")))
    return "\n".join(parts).strip()


def _normalized_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _title_without_year(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


def _contains_phrase(normalized: str, phrase: str) -> bool:
    return f" {_normalized_text(phrase)} " in f" {normalized} "


def analyze_summary(summary: str, record: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    stripped = summary.strip()
    words = stripped.split()
    normalized = _normalized_text(stripped)
    if not stripped:
        return {
            "errors": ["empty"],
            "privacy": {"movie_title": [], "year": [], "numeric_rating": []},
            "content_structure": {
                "liked_genres": False,
                "liked_themes_plots": False,
                "disliked_genres_styles": False,
                "disliked_plots_content": False,
            },
            "grounding": {
                "history_genres": [],
                "mentioned_genres": [],
                "supported_genres": [],
                "unsupported_genres": [],
                "genre_rating_evidence": {},
            },
        }
    if not stripped.startswith("Summary:"):
        errors.append("format_prefix")
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        errors.append(f"word_count:{len(words)}")
    years = sorted(set(re.findall(r"\b(?:18|19|20)\d{2}\b", stripped)))
    if years:
        errors.append("year_leakage")
    numeric_ratings = sorted(set(re.findall(
        r"(?:\b(?:rating|rated)\s*:?\s*[0-5](?:\.0|\.5)?|\b[0-5](?:\.0|\.5)?\s*/\s*5\b|\b[0-5](?:\.0|\.5)?\s+stars?\b|\b[0-5]\.5\b)",
        stripped,
        re.IGNORECASE,
    )))
    if numeric_ratings:
        errors.append("rating_leakage")
    leaked_titles: list[str] = []
    for title in record["titles"]:
        candidate = _title_without_year(str(title))
        if len(_normalized_text(candidate)) >= 4 and _contains_phrase(
            normalized, candidate
        ):
            leaked_titles.append(str(title))
    if leaked_titles:
        errors.append(f"title_leakage:{leaked_titles[0]}")

    positive_markers = (
        "enjoy",
        "like",
        "prefer",
        "appreciate",
        "favor",
        "favour",
        "drawn to",
        "affinity",
    )
    negative_markers = (
        "dislike",
        "does not enjoy",
        "do not enjoy",
        "less interested",
        "less appealing",
        "avoid",
        "averse",
        "not drawn",
        "not prefer",
    )
    plot_markers = (
        "plot",
        "story",
        "stories",
        "narrative",
        "theme",
        "character",
        "content",
        "pacing",
        "tone",
        "conflict",
        "ending",
        "arc",
    )
    other_viewer_markers = (
        "other viewers",
        "other users",
        "other audiences",
        "some viewers",
        "some audiences",
        "others may",
    )
    style_markers = ("style", "styles", "tone", "pacing", "approach")
    has_positive = any(marker in normalized for marker in positive_markers)
    has_negative = any(marker in normalized for marker in negative_markers)
    has_plot = any(marker in normalized for marker in plot_markers)
    has_other_viewers = any(marker in normalized for marker in other_viewer_markers)

    history_genres = {
        genre.strip().lower()
        for value in record["genres"]
        for genre in str(value).split("|")
        if genre.strip() and genre.strip() != "(no genres listed)"
    }
    mentioned: set[str] = set()
    for canonical, aliases in GENRE_TERMS.items():
        if any(_contains_phrase(normalized, alias) for alias in aliases):
            mentioned.add(canonical)
    supported = mentioned & history_genres
    unsupported = mentioned - history_genres
    content_structure = {
        "liked_genres": bool(has_positive and mentioned),
        "liked_themes_plots": bool(has_positive and has_plot),
        "disliked_genres_styles": bool(
            has_negative
            and (mentioned or any(marker in normalized for marker in style_markers))
        ),
        "disliked_plots_content": bool(has_negative and has_plot and has_other_viewers),
    }
    for category, present in content_structure.items():
        if not present:
            errors.append(f"category_{category}")
    if not supported:
        errors.append("grounding_no_history_genre")
    if unsupported:
        errors.append(f"grounding_unsupported_genre:{sorted(unsupported)[0]}")

    evidence: dict[str, list[float]] = {}
    for genres, rating in zip(record["genres"], record.get("ratings", [])):
        for genre in str(genres).split("|"):
            canonical = genre.strip().lower()
            if canonical and canonical != "(no genres listed)":
                evidence.setdefault(canonical, []).append(float(rating))
    genre_rating_evidence = {
        genre: {
            "count": len(values),
            "mean_rating": sum(values) / len(values),
            "liked_rating_count": sum(value >= 4.0 for value in values),
            "disliked_rating_count": sum(value <= 2.5 for value in values),
        }
        for genre, values in sorted(evidence.items())
    }
    return {
        "errors": errors,
        "privacy": {
            "movie_title": leaked_titles,
            "year": years,
            "numeric_rating": numeric_ratings,
        },
        "content_structure": content_structure,
        "grounding": {
            "history_genres": sorted(history_genres),
            "mentioned_genres": sorted(mentioned),
            "supported_genres": sorted(supported),
            "unsupported_genres": sorted(unsupported),
            "genre_rating_evidence": genre_rating_evidence,
        },
    }


def validate_summary(summary: str, record: dict[str, Any]) -> list[str]:
    return list(analyze_summary(summary, record)["errors"])


def _read_response_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "responses").glob("*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    return rows


def _percentile(values: list[int], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _near_duplicate_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row["summary"]]
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(candidates):
        left_normalized = _normalized_text(left["summary"])
        left_tokens = set(left_normalized.split())
        for right in candidates[left_index + 1 :]:
            right_normalized = _normalized_text(right["summary"])
            if left_normalized == right_normalized:
                continue
            right_tokens = set(right_normalized.split())
            union = left_tokens | right_tokens
            jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
            if jaccard < 0.75:
                continue
            sequence = SequenceMatcher(
                None, left_normalized, right_normalized, autojunk=False
            ).ratio()
            if sequence >= 0.92 or jaccard >= 0.85:
                pairs.append(
                    {
                        "left_user_id": left["user_id"],
                        "right_user_id": right["user_id"],
                        "sequence_similarity": sequence,
                        "token_jaccard": jaccard,
                    }
                )
    return sorted(
        pairs,
        key=lambda pair: (
            -max(pair["sequence_similarity"], pair["token_jaccard"]),
            pair["left_user_id"],
            pair["right_user_id"],
        ),
    )


def validate_calibration(
    plan_path: Path,
    artifact_mirror_root: Path | None,
    *,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    plan = _load_verified_plan(plan_path)
    root = plan_path.parent
    submission = json.loads((root / "submission.json").read_text(encoding="utf-8"))
    prices = submission["pricing_usd_per_million"]
    records = {record["custom_id"]: record for record in plan["records"]}
    response_rows = _read_response_rows(root)
    parsed: list[dict[str, Any]] = []
    input_usages: list[int] = []
    cached_input_usages: list[int] = []
    output_usages: list[int] = []
    reasoning_usages: list[int] = []
    per_user_costs: list[float] = []
    received: set[str] = set()

    for row in response_rows:
        custom_id = row.get("custom_id")
        if custom_id not in records or custom_id in received:
            continue
        received.add(custom_id)
        record = records[custom_id]
        errors: list[str] = []
        summary = ""
        response = row.get("response") or {}
        body = response.get("body") or {}
        if response.get("status_code") != 200:
            errors.append(f"http_status:{response.get('status_code')}")
        else:
            raw_text = extract_output_text(body)
            try:
                structured = json.loads(raw_text)
                summary = str(structured["summary"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                errors.append("invalid_structured_output")
        diagnostics = analyze_summary(summary, record)
        if summary:
            errors.extend(diagnostics["errors"])
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0))
        input_details = usage.get("input_tokens_details") or {}
        cached_input_tokens = int(input_details.get("cached_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        output_details = usage.get("output_tokens_details") or {}
        reasoning_tokens = int(output_details.get("reasoning_tokens", 0))
        if cached_input_tokens > input_tokens:
            raise RuntimeError("Cached input tokens exceed total input tokens")
        input_usages.append(input_tokens)
        cached_input_usages.append(cached_input_tokens)
        output_usages.append(output_tokens)
        reasoning_usages.append(reasoning_tokens)
        per_user_costs.append(
            (
                (input_tokens - cached_input_tokens) * prices["input"]
                + cached_input_tokens * prices["cached_input"]
                + output_tokens * prices["output"]
            )
            / 1_000_000
        )
        parsed.append(
            {
                "user_id": record["user_id"],
                "split": record["split"],
                "activity_band": record["activity_band"],
                "custom_id": custom_id,
                "summary": summary,
                "word_count": len(summary.split()),
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "privacy": diagnostics["privacy"],
                "content_structure": diagnostics["content_structure"],
                "grounding": diagnostics["grounding"],
                "errors": errors,
            }
        )

    for custom_id, record in records.items():
        if custom_id not in received:
            parsed.append(
                {
                    "user_id": record["user_id"],
                    "split": record["split"],
                    "activity_band": record["activity_band"],
                    "custom_id": custom_id,
                    "summary": "",
                    "word_count": 0,
                    "input_tokens": 0,
                    "cached_input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                    "privacy": {
                        "movie_title": [],
                        "year": [],
                        "numeric_rating": [],
                    },
                    "content_structure": {
                        "liked_genres": False,
                        "liked_themes_plots": False,
                        "disliked_genres_styles": False,
                        "disliked_plots_content": False,
                    },
                    "grounding": {
                        "history_genres": [],
                        "mentioned_genres": [],
                        "supported_genres": [],
                        "unsupported_genres": [],
                        "genre_rating_evidence": {},
                    },
                    "errors": ["missing_response"],
                }
            )

    duplicate_groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in parsed:
        if candidate["summary"]:
            key = " ".join(candidate["summary"].lower().split())
            duplicate_groups.setdefault(key, []).append(candidate)
    for group in duplicate_groups.values():
        if len({candidate["user_id"] for candidate in group}) > 1:
            for candidate in group:
                candidate["errors"].append("cross_user_duplicate")

    near_duplicate_pairs = _near_duplicate_pairs(parsed)
    valid = [row for row in parsed if not row["errors"]]
    invalid = [row for row in parsed if row["errors"]]
    error_counts = Counter(
        error.split(":", 1)[0] for row in invalid for error in row["errors"]
    )
    total_input = sum(input_usages)
    total_cached_input = sum(cached_input_usages)
    total_uncached_input = total_input - total_cached_input
    total_output = sum(output_usages)
    actual_cost = (
        total_uncached_input * prices["input"]
        + total_cached_input * prices["cached_input"]
        + total_output * prices["output"]
    ) / 1_000_000
    if input_usages:
        mean_input = total_input / len(input_usages)
        mean_output = total_output / len(output_usages)
        mean_cost = actual_cost / CALIBRATION_USERS
        corpus_expected = mean_cost * 200_948
        remaining_expected = actual_cost + mean_cost * (200_948 - CALIBRATION_USERS)
        wave_expected = mean_cost * 50_000
        max_observed_cost = max(per_user_costs)
        conservative_upper = max_observed_cost * 200_948 * 1.15
        conservative_wave = max_observed_cost * 50_000 * 1.15
    else:
        mean_input = mean_output = mean_cost = 0.0
        corpus_expected = remaining_expected = wave_expected = 0.0
        max_observed_cost = conservative_upper = conservative_wave = 0.0

    word_counts = [row["word_count"] for row in parsed if row["summary"]]
    report: dict[str, Any] = {
        "scope": "calibration_only",
        "stop_before_full_cohort": True,
        "plan": plan["fingerprint"],
        "protocol_version": plan["protocol"]["version"],
        "protocol_fingerprint": plan["protocol"]["fingerprint"],
        "prompt_sha256": plan["protocol"]["system_prompt_sha256"],
        "model": plan["protocol"]["generation"]["model"],
        "dataset_fingerprint": plan["dataset"]["matrix_fingerprint"],
        "expected": CALIBRATION_USERS,
        "received": len(received),
        "successful_api_responses": sum(
            not any(error.startswith("http_status") for error in row["errors"])
            and "missing_response" not in row["errors"]
            for row in parsed
        ),
        "successful_structured_responses": sum(
            bool(row["summary"])
            and "invalid_structured_output" not in row["errors"]
            and not any(error.startswith("http_status") for error in row["errors"])
            for row in parsed
        ),
        "valid": len(valid),
        "invalid": len(invalid),
        "missing_user_ids": [
            row["user_id"] for row in invalid if "missing_response" in row["errors"]
        ],
        "invalid_user_ids": [row["user_id"] for row in invalid],
        "error_counts": dict(sorted(error_counts.items())),
        "cross_user_duplicate_groups": sum(
            len({candidate["user_id"] for candidate in group}) > 1
            for group in duplicate_groups.values()
        ),
        "cross_user_near_duplicate_pairs": len(near_duplicate_pairs),
        "near_duplicate_pairs": near_duplicate_pairs,
        "privacy_leakage": {
            "movie_title_user_ids": sorted(
                row["user_id"] for row in parsed if row["privacy"]["movie_title"]
            ),
            "year_user_ids": sorted(
                row["user_id"] for row in parsed if row["privacy"]["year"]
            ),
            "numeric_rating_user_ids": sorted(
                row["user_id"]
                for row in parsed
                if row["privacy"]["numeric_rating"]
            ),
        },
        "content_structure_preserved": {
            category: sum(
                bool(row["summary"]) and row["content_structure"][category]
                for row in parsed
            )
            for category in (
                "liked_genres",
                "liked_themes_plots",
                "disliked_genres_styles",
                "disliked_plots_content",
            )
        },
        "grounding": {
            "users_with_supported_history_genre": sum(
                bool(row["grounding"]["supported_genres"]) for row in parsed
            ),
            "users_with_unsupported_genre_claim": sum(
                bool(row["grounding"]["unsupported_genres"]) for row in parsed
            ),
            "manual_theme_plot_grounding_required": True,
        },
        "word_counts": {
            "minimum": min(word_counts) if word_counts else 0,
            "p05": _percentile(word_counts, 5) if word_counts else 0,
            "p25": _percentile(word_counts, 25) if word_counts else 0,
            "mean": float(np.mean(word_counts)) if word_counts else 0,
            "median": _percentile(word_counts, 50) if word_counts else 0,
            "p75": _percentile(word_counts, 75) if word_counts else 0,
            "p95": _percentile(word_counts, 95) if word_counts else 0,
            "maximum": max(word_counts) if word_counts else 0,
            "standard_deviation_population": float(np.std(word_counts))
            if word_counts
            else 0,
        },
        "usage": {
            "input_tokens": total_input,
            "uncached_input_tokens": total_uncached_input,
            "cached_input_tokens": total_cached_input,
            "output_tokens": total_output,
            "reasoning_tokens_in_output": sum(reasoning_usages),
            "mean_input_tokens_per_user": mean_input,
            "mean_output_tokens_per_user": mean_output,
            "p95_input_tokens_per_user": _percentile(input_usages, 95)
            if input_usages
            else 0,
            "p95_output_tokens_per_user": _percentile(output_usages, 95)
            if output_usages
            else 0,
            "maximum_input_tokens_per_user": max(input_usages) if input_usages else 0,
            "maximum_output_tokens_per_user": max(output_usages)
            if output_usages
            else 0,
        },
        "cost": {
            "pricing_usd_per_million": prices,
            "exact_calibration_usd": actual_cost,
            "mean_per_user_usd": mean_cost,
            "estimated_all_200948_users_usd": corpus_expected,
            "estimated_incremental_remaining_after_calibration_usd": max(
                0.0, remaining_expected - actual_cost
            ),
            "estimated_complete_corpus_including_calibration_usd": remaining_expected,
            "estimated_50000_user_wave_usd": wave_expected,
            "maximum_observed_per_user_usd": max_observed_cost,
            "conservative_upper_all_200948_usd": conservative_upper,
            "conservative_upper_50000_user_wave_usd": conservative_wave,
            "conservative_method": "maximum observed per-user calibration cost multiplied by population and a 15% reserve",
        },
        "calibration_passed": (
            len(received) == CALIBRATION_USERS
            and len(valid) == CALIBRATION_USERS
            and len(invalid) == 0
        ),
    }
    report["fingerprint"] = stable_hash(report)
    if write_artifacts:
        _write_jsonl(root / "validated" / "all_summaries.jsonl", parsed)
        _write_jsonl(root / "validated" / "summaries.jsonl", valid)
        _write_jsonl(root / "validated" / "invalid.jsonl", invalid)
        atomic_write_json(root / "validated" / "calibration_report.json", report)
        atomic_write_json(root / "reports" / "validation_report.json", report)
        atomic_write_json(
            root / "reports" / "cost_report.json",
            {
                "scope": "calibration_only",
                "model": report["model"],
                "pricing_source": submission["pricing_source"],
                "pricing_verified_at": submission["pricing_verified_at"],
                "usage": report["usage"],
                "cost": report["cost"],
            },
        )
        if artifact_mirror_root is not None:
            relative = Path("final_calibration") / plan["fingerprint"] / "validated"
            for path in (
                root / "validated" / "all_summaries.jsonl",
                root / "validated" / "summaries.jsonl",
                root / "validated" / "invalid.jsonl",
                root / "validated" / "calibration_report.json",
            ):
                mirror_paid_artifact(path, artifact_mirror_root, relative / path.name)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tears_training.final_summaries"
    )
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan-calibration")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--matrix-dir", type=Path, required=True)
    plan.add_argument("--dataset-dir", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--seed", type=int, default=SEED)
    submit = sub.add_parser("submit-calibration")
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--artifact-mirror-root", type=Path)
    submit.add_argument("--input-price-per-million", type=float, required=True)
    submit.add_argument("--cached-input-price-per-million", type=float, required=True)
    submit.add_argument("--output-price-per-million", type=float, required=True)
    submit.add_argument("--max-calibration-cost-usd", type=float, default=1.0)
    submit.add_argument("--confirm-calibration-spend", action="store_true")
    poll = sub.add_parser("poll-calibration")
    poll.add_argument("--submission", type=Path, required=True)
    poll.add_argument("--artifact-mirror-root", type=Path)
    validate = sub.add_parser("validate-calibration")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--artifact-mirror-root", type=Path)
    validate.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.action == "plan-calibration":
        result = plan_calibration(
            args.config,
            args.matrix_dir,
            args.dataset_dir,
            args.output_dir,
            args.seed,
        )
    elif args.action == "submit-calibration":
        result = submit_calibration(
            args.plan,
            args.artifact_mirror_root,
            args.input_price_per_million,
            args.cached_input_price_per_million,
            args.output_price_per_million,
            args.max_calibration_cost_usd,
            args.confirm_calibration_spend,
        )
    elif args.action == "poll-calibration":
        result = poll_calibration(args.submission, args.artifact_mirror_root)
    else:
        result = validate_calibration(
            args.plan,
            args.artifact_mirror_root,
            write_artifacts=not args.dry_run,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
