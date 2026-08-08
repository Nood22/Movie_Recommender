from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .artifacts import (
    append_jsonl,
    atomic_write_bytes,
    atomic_write_json,
    mirror_paid_artifact,
    stable_hash,
)
from .config import ExperimentConfig, load_config


SYSTEM_PROMPT = """You describe a movie viewer's preferences for the TEARS recommender system.
Use only preferences supported by the supplied private viewing history and write in third person.
Return exactly four declarative sentences in the summary field, in this order:
1. Begin with "Summary:" and describe liked genres.
2. Describe liked plot points, themes, or content preferences where supported.
3. Describe disliked genres, themes, or styles where supported.
4. Describe disliked plot points or content preferences, including what other viewers may enjoy, where supported.
If the history does not support a like or dislike, say that no strong preference is supported instead of inventing one.
Never name movies, actors, years, or numeric ratings. Do not recommend titles or invent facts.
Use periods only to end the four required sentences; avoid abbreviations containing periods."""

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}

VALIDATION_VERSION = "tears-summary-privacy-format-v3-emiliano"


@dataclass(frozen=True)
class SummaryRequest:
    user_id: int
    history: tuple[tuple[str, float, str], ...]
    prompt_version: str
    model: str
    max_output_tokens: int
    attempt: int = 0

    @property
    def history_hash(self) -> str:
        return stable_hash(self.history)

    @property
    def prompt_hash(self) -> str:
        return stable_hash({"version": self.prompt_version, "system": SYSTEM_PROMPT})

    @property
    def cache_key(self) -> str:
        return stable_hash(
            {
                "user_id": self.user_id,
                "ordered_history_hash": self.history_hash,
                "prompt_hash": self.prompt_hash,
                "model": self.model,
                "generation": {
                    "max_output_tokens": self.max_output_tokens,
                    "reasoning_effort": "minimal",
                    "schema": SUMMARY_SCHEMA,
                },
            }
        )

    @property
    def custom_id(self) -> str:
        return f"user-{self.user_id}-{self.cache_key[:16]}-a{self.attempt}"


def render_history_prompt(history_lines: str, min_words: int, max_words: int) -> str:
    return (
        f"Return a {min_words}-{max_words} word TEARS profile in the summary field. "
        "Follow the exact four-sentence order in the system instructions. "
        "Treat titles and ratings as private evidence that must not appear in the profile.\n\n"
        + history_lines
    )


def render_prompt(request: SummaryRequest, min_words: int, max_words: int) -> str:
    lines = "\n".join(
        f"- title={title!r}; private_rating={rating:g}/5; genres={genres}"
        for title, rating, genres in request.history
    )
    return render_history_prompt(lines, min_words, max_words)


def batch_body(request: SummaryRequest, config: ExperimentConfig) -> dict[str, Any]:
    return {
        "custom_id": request.custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": request.model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": render_prompt(
                        request,
                        config.summaries.min_words,
                        config.summaries.max_words,
                    ),
                },
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
            "max_output_tokens": request.max_output_tokens,
            "store": False,
        },
    }


def load_requests(
    config: ExperimentConfig, matrix_dir: Path, split: str
) -> list[SummaryRequest]:
    users = pd.read_csv(matrix_dir / "users.csv")
    selected_users = users if split == "all" else users.loc[users.split == split]
    wanted = set(selected_users.userId.astype(int))
    user_splits = {
        int(row.userId): str(row.split) for row in selected_users.itertuples(index=False)
    }
    catalog = pd.read_csv(
        matrix_dir / "catalog.csv", usecols=["movieId", "title", "genres"]
    )
    evidence = {
        int(row.movieId): (str(row.title), str(row.genres))
        for row in catalog.itertuples(index=False)
    }
    histories: dict[int, list[tuple[int, int, str, float, str]]] = {
        user: [] for user in wanted
    }
    for chunk in pd.read_csv(config.raw_data / "ratings.csv", chunksize=1_000_000):
        part = chunk[chunk.userId.isin(wanted) & chunk.movieId.isin(evidence)]
        for row in part.itertuples(index=False):
            title, genres = evidence[int(row.movieId)]
            histories[int(row.userId)].append(
                (int(row.timestamp), int(row.movieId), title, float(row.rating), genres)
            )
    result = []
    for user in sorted(wanted):
        history = sorted(histories[user], key=lambda value: (value[0], value[1]))
        if user_splits[user] in {"validation", "test"} and history:
            boundary = max(
                1,
                min(
                    len(history) - 1,
                    int(np.floor(len(history) * config.data.observed_fraction)),
                ),
            )
            history = history[:boundary]
        history = history[-config.summaries.max_history_items :]
        result.append(
            SummaryRequest(
                user_id=user,
                history=tuple((title, rating, genres) for _, _, title, rating, genres in history),
                prompt_version=config.summaries.prompt_version,
                model=config.summaries.model,
                max_output_tokens=config.summaries.max_output_tokens,
            )
        )
    return result


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count, path.stat().st_size


def estimate_tokens(text: str) -> int:
    # Deliberately conservative dependency-free estimate.
    return max(1, int(len(text) / 4 * 1.20))


def _request_record(request: SummaryRequest) -> dict[str, Any]:
    return {
        "custom_id": request.custom_id,
        "user_id": request.user_id,
        "cache_key": request.cache_key,
        "history_hash": request.history_hash,
        "prompt_hash": request.prompt_hash,
        "attempt": request.attempt,
        "titles": [title for title, _, _ in request.history],
    }


def _write_plan(
    config: ExperimentConfig,
    root: Path,
    requests: list[SummaryRequest],
    filename: str = "request_plan.json",
    dry_run: bool = False,
) -> dict[str, Any]:
    chunks: list[list[SummaryRequest]] = []
    current: list[SummaryRequest] = []
    current_bytes = 0
    input_tokens = 0
    for request in requests:
        body = batch_body(request, config)
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        if len(encoded) > config.summaries.max_batch_bytes:
            raise RuntimeError(f"One request exceeds the Batch file limit: {request.custom_id}")
        if current and (
            len(current) >= config.summaries.max_batch_requests
            or current_bytes + len(encoded) > config.summaries.max_batch_bytes
        ):
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(request)
        current_bytes += len(encoded)
        input_tokens += estimate_tokens(
            SYSTEM_PROMPT + body["body"]["input"][1]["content"]
        )
    if current:
        chunks.append(current)

    files = []
    prefix = Path(filename).stem
    for index, chunk in enumerate(chunks):
        path = root / "requests" / f"{prefix}-wave-{index:04d}.jsonl"
        if dry_run:
            encoded_rows = [
                json.dumps(
                    batch_body(request, config),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
                + b"\n"
                for request in chunk
            ]
            count, size = len(chunk), sum(map(len, encoded_rows))
        else:
            count, size = write_jsonl(
                path, (batch_body(request, config) for request in chunk)
            )
        files.append({"path": str(path.resolve()), "requests": count, "bytes": size})
    plan: dict[str, Any] = {
        "users": len(requests),
        "requests": len(requests),
        "estimated_input_tokens": input_tokens,
        "reserved_output_tokens": len(requests) * config.summaries.max_output_tokens,
        "files": files,
        "records": [_request_record(request) for request in requests],
        "model": config.summaries.model,
        "endpoint": "/v1/responses",
    }
    plan["fingerprint"] = stable_hash(plan)
    if not dry_run:
        atomic_write_json(root / filename, plan)
    return plan


def plan_batches(
    config: ExperimentConfig,
    matrix_dir: Path,
    split: str,
    limit: int | None,
    dry_run: bool = False,
    output_dir: Path | None = None,
    exclude_plans: tuple[Path, ...] = (),
) -> dict[str, Any]:
    requests = load_requests(config, matrix_dir, split)
    excluded_user_ids: set[int] = set()
    exclusion_fingerprints: list[str] = []
    for path in exclude_plans:
        exclusion = json.loads(path.read_text(encoding="utf-8"))
        excluded_user_ids.update(int(record["user_id"]) for record in exclusion["records"])
        exclusion_fingerprints.append(str(exclusion["fingerprint"]))
    requests = [request for request in requests if request.user_id not in excluded_user_ids]
    if limit is not None and len(requests) < limit:
        raise RuntimeError(
            f"Only {len(requests)} users remain after exclusions; cannot plan {limit}"
        )
    if limit is not None:
        requests = requests[:limit]
    root = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else config.output_root / "summaries" / matrix_dir.parent.name / split
    )
    cache = root / "cache"
    if not dry_run:
        cache.mkdir(parents=True, exist_ok=True)
    uncached = [r for r in requests if not (cache / f"{r.cache_key}.json").exists()]
    plan = _write_plan(config, root, uncached, dry_run=dry_run)
    plan["cohort_users"] = len(requests)
    plan["cached"] = len(requests) - len(uncached)
    plan["excluded_user_ids"] = len(excluded_user_ids)
    plan["excluded_overlap"] = sum(
        request.user_id in excluded_user_ids for request in requests
    )
    plan["exclusion_plan_fingerprints"] = exclusion_fingerprints
    if plan["excluded_overlap"]:
        raise AssertionError("Excluded users remain in the paid request plan")
    plan["dry_run"] = dry_run
    plan["fingerprint"] = stable_hash({k: v for k, v in plan.items() if k != "fingerprint"})
    if not dry_run:
        atomic_write_json(root / "request_plan.json", plan)
    return plan


def operation_event(action: str, result: dict[str, Any]) -> dict[str, Any]:
    """Return a compact, secret-free event suitable for a durable run log."""
    event: dict[str, Any] = {"action": action, "recorded_at": int(time.time())}
    for key in (
        "plan",
        "fingerprint",
        "cohort_users",
        "cached",
        "excluded_user_ids",
        "excluded_overlap",
        "users",
        "requests",
        "estimated_input_tokens",
        "reserved_output_tokens",
        "projected_cost_usd",
        "actual_cost_usd",
        "cumulative_cost_usd",
        "expected",
        "valid",
        "invalid",
        "validation_version",
        "validation_fingerprint",
        "dry_run",
        "external_request_performed",
    ):
        if key in result:
            event[key] = result[key]
    if "batches" in result:
        event["batches"] = [
            {
                "batch_id": batch.get("batch_id", batch.get("id")),
                "status": batch.get("status"),
                "request_counts": batch.get("request_counts"),
            }
            for batch in result["batches"]
        ]
    if "usage" in result:
        event["usage"] = result["usage"]
    return event


def projected_cost_usd(plan: dict[str, Any], input_per_million: float, output_per_million: float) -> float:
    if input_per_million < 0 or output_per_million < 0:
        raise ValueError("Token prices cannot be negative")
    return (
        plan["estimated_input_tokens"] * input_per_million
        + plan["reserved_output_tokens"] * output_per_million
    ) / 1_000_000


def enforce_cost_gate(
    config: ExperimentConfig,
    projected_cost: float,
    actual_spend: float,
) -> None:
    if projected_cost <= 0:
        raise RuntimeError("Projected cost must be positive")
    protected = (actual_spend + projected_cost) * (1 + config.summaries.reserve_fraction)
    if protected > config.summaries.spend_cap_usd:
        raise RuntimeError(
            f"Spend plus projection and reserve is ${protected:.2f}, above the "
            f"${config.summaries.spend_cap_usd:.2f} cap"
        )


def _require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY must be supplied through the job environment; never store it in files"
        )


def submit_batches(
    config: ExperimentConfig,
    plan_path: Path,
    input_price: float,
    output_price: float,
    actual_spend: float,
    confirm_spend: bool,
) -> dict[str, Any]:
    if not confirm_spend:
        raise RuntimeError("Paid submission requires the explicit --confirm-spend flag")
    _require_api_key()
    from openai import OpenAI

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected_fingerprint = stable_hash({k: v for k, v in plan.items() if k != "fingerprint"})
    if plan.get("fingerprint") != expected_fingerprint:
        raise RuntimeError("Request plan fingerprint mismatch")
    estimate = projected_cost_usd(plan, input_price, output_price)
    enforce_cost_gate(config, estimate, actual_spend)
    if not plan["files"]:
        raise RuntimeError("The request plan contains no uncached requests")

    paid_relative = Path(plan["fingerprint"])
    mirror_paid_artifact(plan_path, config.paid_backup_root, paid_relative / plan_path.name)
    for entry in plan["files"]:
        request_path = Path(entry["path"])
        if request_path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Request file changed after planning: {request_path}")
        mirror_paid_artifact(
            request_path, config.paid_backup_root, paid_relative / "requests" / request_path.name
        )

    client = OpenAI()
    submitted = []
    for entry in plan["files"]:
        request_path = Path(entry["path"])
        with request_path.open("rb") as handle:
            uploaded = client.files.create(file=handle, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"plan": plan["fingerprint"]},
        )
        submitted.append(
            {"batch_id": batch.id, "input_file_id": uploaded.id, "status": batch.status}
        )
    record = {
        "plan": plan["fingerprint"],
        "plan_path": str(plan_path.resolve()),
        "projected_cost_usd": estimate,
        "actual_spend_before_usd": actual_spend,
        "pricing_usd_per_million": {"input": input_price, "output": output_price},
        "submitted_at": int(time.time()),
        "batches": submitted,
    }
    output = plan_path.parent / "submission.json"
    atomic_write_json(output, record)
    mirror_paid_artifact(output, config.paid_backup_root, paid_relative / output.name)
    return record


def poll_batches(config: ExperimentConfig, submission_path: Path) -> dict[str, Any]:
    _require_api_key()
    from openai import OpenAI

    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    client = OpenAI()
    root = submission_path.parent
    paid_relative = Path(submission["plan"])
    statuses = []
    for entry in submission["batches"]:
        batch = client.batches.retrieve(entry["batch_id"])
        status = batch.model_dump(mode="json")
        statuses.append(status)
        for kind, file_id in (
            ("responses", batch.output_file_id),
            ("errors", batch.error_file_id),
        ):
            if not file_id:
                continue
            payload = client.files.content(file_id).content
            path = root / kind / f"{batch.id}.jsonl"
            atomic_write_bytes(path, payload)
            mirror_paid_artifact(path, config.paid_backup_root, paid_relative / kind / path.name)
    input_tokens = sum(
        int((status.get("usage") or {}).get("input_tokens", 0)) for status in statuses
    )
    output_tokens = sum(
        int((status.get("usage") or {}).get("output_tokens", 0)) for status in statuses
    )
    prices = submission["pricing_usd_per_million"]
    current_cost = (
        input_tokens * prices["input"] + output_tokens * prices["output"]
    ) / 1_000_000
    result = {
        "polled_at": int(time.time()),
        "plan": submission["plan"],
        "batches": statuses,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "actual_cost_usd": current_cost,
        "cumulative_cost_usd": submission["actual_spend_before_usd"] + current_cost,
    }
    atomic_write_json(root / "poll.json", result)
    return result


def extract_output_text(response_body: dict[str, Any]) -> str:
    if isinstance(response_body.get("output_text"), str):
        return response_body["output_text"].strip()
    parts = []
    for output in response_body.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "\n".join(parts).strip()


def _normalized_title(title: str) -> str:
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def validate_text(
    text: str,
    config: ExperimentConfig,
    titles: Iterable[str] = (),
) -> list[str]:
    words = text.split()
    lowered = re.sub(r"[^a-z0-9]+", " ", text.lower())
    padded_lowered = f" {lowered} "
    errors = []
    if not text.strip():
        errors.append("empty")
    stripped = text.strip()
    if not stripped.startswith("Summary:"):
        errors.append("format_prefix")
    else:
        body = stripped[len("Summary:") :].strip()
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", body)
            if sentence.strip()
        ]
        if (
            len(sentences) != 4
            or any(sentence[-1:] not in ".!?" for sentence in sentences)
        ):
            errors.append(f"format_parts:{len(sentences)}")
    if re.search(r"\b(?:i|me|my|mine|we|us|our|ours)\b", text, re.IGNORECASE):
        errors.append("not_third_person")
    if not re.search(r"\b(?:viewer|user|they|their|them)\b", text, re.IGNORECASE):
        errors.append("not_third_person")
    if not (config.summaries.min_words <= len(words) <= config.summaries.max_words):
        errors.append(f"word_count:{len(words)}")
    if re.search(r"\b(?:18|19|20)\d{2}\b", text):
        errors.append("year_leakage")
    if re.search(r"(?:\s*/\s*5\b|\brating\s*:|\brated\b|\bstars\b)", text.lower()):
        errors.append("rating_leakage")
    leaking = [
        title
        for title in titles
        if len(_normalized_title(title)) >= 4
        and f" {_normalized_title(title)} " in padded_lowered
    ]
    if leaking:
        errors.append(f"title_leakage:{leaking[0]}")
    return errors


def _read_response_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "responses").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return rows


def _build_retry_plan(
    config: ExperimentConfig,
    root: Path,
    source_plan: dict[str, Any],
    retryable: list[dict[str, Any]],
) -> Path:
    inputs: dict[str, dict[str, Any]] = {}
    for entry in source_plan["files"]:
        for line in Path(entry["path"]).read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                inputs[row["custom_id"]] = row
    source_records = {record["custom_id"]: record for record in source_plan["records"]}
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for invalid in retryable:
        old_id = invalid["custom_id"]
        if old_id not in inputs or old_id not in source_records:
            continue
        attempt = int(invalid.get("attempt", 0)) + 1
        new_id = re.sub(r"-a\d+$", f"-a{attempt}", old_id)
        row = json.loads(json.dumps(inputs[old_id]))
        row["custom_id"] = new_id
        record = dict(source_records[old_id])
        record["custom_id"] = new_id
        record["attempt"] = attempt
        rows.append(row)
        records.append(record)
    files: list[dict[str, Any]] = []
    for start in range(0, len(rows), config.summaries.max_batch_requests):
        chunk = rows[start : start + config.summaries.max_batch_requests]
        index = start // config.summaries.max_batch_requests
        path = root / "requests" / f"retry-wave-{index:04d}.jsonl"
        count, size = write_jsonl(path, chunk)
        if size > config.summaries.max_batch_bytes:
            raise RuntimeError(
                "Retry wave exceeds the byte limit; lower max_batch_requests and revalidate"
            )
        files.append({"path": str(path.resolve()), "requests": count, "bytes": size})
    retry_plan: dict[str, Any] = {
        "users": len(records),
        "requests": len(records),
        "estimated_input_tokens": sum(
            estimate_tokens(
                SYSTEM_PROMPT + row["body"]["input"][1]["content"]
            )
            for row in rows
        ),
        "reserved_output_tokens": len(records) * config.summaries.max_output_tokens,
        "files": files,
        "records": records,
        "model": config.summaries.model,
        "endpoint": "/v1/responses",
        "retry_of": source_plan["fingerprint"],
    }
    retry_plan["fingerprint"] = stable_hash(retry_plan)
    path = root / "retry_plan.json"
    atomic_write_json(path, retry_plan)
    return path


def validate_responses(config: ExperimentConfig, plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    root = plan_path.parent
    records = {record["custom_id"]: record for record in plan["records"]}
    parsed: list[dict[str, Any]] = []
    response_rows = _read_response_rows(root)
    for row in response_rows:
        custom_id = row.get("custom_id")
        record = records.get(custom_id)
        errors: list[str] = []
        summary = ""
        if record is None:
            continue
        response = row.get("response") or {}
        if response.get("status_code") != 200:
            errors.append(f"http_status:{response.get('status_code')}")
        else:
            raw_text = extract_output_text(response.get("body") or {})
            try:
                structured = json.loads(raw_text)
                summary = structured["summary"].strip()
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                errors.append("invalid_structured_output")
            if summary:
                errors.extend(validate_text(summary, config, record.get("titles", [])))
        canonical = {
            "user_id": record["user_id"],
            "summary": summary,
            "cache_key": record["cache_key"],
            "custom_id": custom_id,
            "attempt": record.get("attempt", 0),
        }
        parsed.append(canonical | {"errors": errors})

    duplicate_groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in parsed:
        if candidate["summary"]:
            normalized = " ".join(candidate["summary"].lower().split())
            duplicate_groups.setdefault(normalized, []).append(candidate)
    for group in duplicate_groups.values():
        if len({candidate["user_id"] for candidate in group}) > 1:
            ids = ",".join(candidate["custom_id"] for candidate in group)
            for candidate in group:
                candidate["errors"].append(f"cross_user_duplicate:{ids}")

    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for candidate in parsed:
        if candidate["errors"]:
            invalid.append(candidate)
        else:
            candidate.pop("errors")
            valid.append(candidate)
            atomic_write_json(
                root / "cache" / f"{candidate['cache_key']}.json", candidate
            )

    received = {row.get("custom_id") for row in response_rows}
    for custom_id, record in records.items():
        if custom_id not in received:
            invalid.append(
                {
                    "user_id": record["user_id"],
                    "custom_id": custom_id,
                    "cache_key": record["cache_key"],
                    "attempt": record.get("attempt", 0),
                    "summary": "",
                    "errors": ["missing_response"],
                }
            )

    write_jsonl(root / "validated" / "summaries.jsonl", valid)
    write_jsonl(root / "validated" / "invalid.jsonl", invalid)
    validation_fingerprint = stable_hash(
        {"plan": plan["fingerprint"], "validation_version": VALIDATION_VERSION}
    )
    result: dict[str, Any] = {
        "plan": plan["fingerprint"],
        "validation_version": VALIDATION_VERSION,
        "validation_fingerprint": validation_fingerprint,
        "expected": len(records),
        "valid": len(valid),
        "invalid": len(invalid),
    }
    # A retry plan is deliberately explicit and still needs a separate paid submit command.
    retryable = [row for row in invalid if int(row.get("attempt", 0)) < config.summaries.retry_limit]
    if retryable:
        result["retry_required"] = len(retryable)
        retry_path = _build_retry_plan(config, root, plan, retryable)
        result["retry_plan"] = str(retry_path)
        result["retry_note"] = "Retry submission remains behind the normal explicit cost gate."
    atomic_write_json(root / "validated" / "validation_report.json", result)
    paid_relative = Path(plan["fingerprint"])
    for name in ("summaries.jsonl", "invalid.jsonl", "validation_report.json"):
        path = root / "validated" / name
        mirror_paid_artifact(
            path,
            config.paid_backup_root,
            paid_relative / "validated" / validation_fingerprint / name,
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tears_training.summaries")
    parser.add_argument("--config", type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--matrix-dir", type=Path, required=True)
    plan.add_argument(
        "--split", choices=("train", "validation", "test", "all"), default="train"
    )
    plan.add_argument("--limit", type=int)
    plan.add_argument("--output-dir", type=Path)
    plan.add_argument("--exclude-plan", type=Path, action="append", default=[])
    plan.add_argument("--dry-run", action="store_true")
    submit = sub.add_parser("submit")
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--input-price-per-million", type=float, required=True)
    submit.add_argument("--output-price-per-million", type=float, required=True)
    submit.add_argument("--actual-spend-usd", type=float, required=True)
    submit.add_argument("--confirm-spend", action="store_true")
    submit.add_argument("--dry-run", action="store_true")
    poll = sub.add_parser("poll")
    poll.add_argument("--submission", type=Path, required=True)
    poll.add_argument("--dry-run", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.action == "plan":
        result = plan_batches(
            config,
            args.matrix_dir,
            args.split,
            args.limit,
            args.dry_run,
            args.output_dir,
            tuple(args.exclude_plan),
        )
        event_root = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else config.output_root / "summaries" / args.matrix_dir.parent.name / args.split
        )
    elif args.action == "submit":
        event_root = args.plan.resolve().parent
        if args.dry_run:
            plan_value = json.loads(args.plan.read_text(encoding="utf-8"))
            estimate = projected_cost_usd(
                plan_value,
                args.input_price_per_million,
                args.output_price_per_million,
            )
            enforce_cost_gate(config, estimate, args.actual_spend_usd)
            result = {
                "dry_run": True,
                "requests": plan_value["requests"],
                "projected_cost_usd": estimate,
                "external_request_performed": False,
            }
        else:
            result = submit_batches(
                config,
                args.plan,
                args.input_price_per_million,
                args.output_price_per_million,
                args.actual_spend_usd,
                args.confirm_spend,
            )
    elif args.action == "poll":
        event_root = args.submission.resolve().parent
        result = (
            {
                "dry_run": True,
                "submission": str(args.submission),
                "external_request_performed": False,
            }
            if args.dry_run
            else poll_batches(config, args.submission)
        )
    else:
        event_root = args.plan.resolve().parent
        result = (
            {"dry_run": True, "plan": str(args.plan), "files_written": False}
            if args.dry_run
            else validate_responses(config, args.plan)
        )
    if not getattr(args, "dry_run", False):
        append_jsonl(event_root / "logs" / "events.jsonl", [operation_event(args.action, result)])
        if args.action in {"poll", "validate"}:
            from .summary_wandb import log_monitor_event

            log_monitor_event(event_root, args.action, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
