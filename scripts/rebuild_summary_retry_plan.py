from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tears_training.artifacts import atomic_write_json, stable_hash
from tears_training.config import load_config
from tears_training.summaries import (
    SUMMARY_SCHEMA,
    SYSTEM_PROMPT,
    estimate_tokens,
    render_history_prompt,
    write_jsonl,
)


def rebuild_retry_plan(config_path: Path, source_path: Path, output_dir: Path) -> Path:
    config = load_config(config_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    expected = stable_hash({key: value for key, value in source.items() if key != "fingerprint"})
    if source.get("fingerprint") != expected:
        raise RuntimeError("Source retry plan fingerprint mismatch")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")

    source_records = {record["custom_id"]: record for record in source["records"]}
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    prompt_hash = stable_hash(
        {"version": config.summaries.prompt_version, "system": SYSTEM_PROMPT}
    )

    for file_entry in source["files"]:
        for raw_line in Path(file_entry["path"]).read_text(encoding="utf-8").splitlines():
            if not raw_line:
                continue
            row = json.loads(raw_line)
            old_id = row["custom_id"]
            record = dict(source_records[old_id])
            old_user_prompt = row["body"]["input"][1]["content"]
            separator = old_user_prompt.find("\n\n")
            if separator < 0:
                raise RuntimeError(f"Request has no private-history separator: {old_id}")
            history_lines = old_user_prompt[separator + 2 :]
            if not history_lines.startswith("- title="):
                raise RuntimeError(f"Request has no private history: {old_id}")

            cache_key = stable_hash(
                {
                    "user_id": record["user_id"],
                    "ordered_history_hash": record["history_hash"],
                    "prompt_hash": prompt_hash,
                    "model": config.summaries.model,
                    "generation": {
                        "max_output_tokens": config.summaries.max_output_tokens,
                        "reasoning_effort": "minimal",
                        "schema": SUMMARY_SCHEMA,
                    },
                }
            )
            attempt = int(record.get("attempt", 1))
            custom_id = f"user-{record['user_id']}-{cache_key[:16]}-a{attempt}"

            row["custom_id"] = custom_id
            row["body"]["model"] = config.summaries.model
            row["body"]["input"][0]["content"] = SYSTEM_PROMPT
            row["body"]["input"][1]["content"] = render_history_prompt(
                history_lines,
                config.summaries.min_words,
                config.summaries.max_words,
            )
            row["body"]["max_output_tokens"] = config.summaries.max_output_tokens
            record.update(
                {
                    "attempt": attempt,
                    "cache_key": cache_key,
                    "custom_id": custom_id,
                    "prompt_hash": prompt_hash,
                }
            )
            rows.append(row)
            records.append(record)

    request_entries: list[dict[str, Any]] = []
    for start in range(0, len(rows), config.summaries.max_batch_requests):
        chunk = rows[start : start + config.summaries.max_batch_requests]
        index = start // config.summaries.max_batch_requests
        request_path = output_dir / "requests" / f"request_plan-wave-{index:04d}.jsonl"
        count, size = write_jsonl(request_path, chunk)
        if size > config.summaries.max_batch_bytes:
            raise RuntimeError("Rebuilt retry wave exceeds the Batch byte limit")
        request_entries.append(
            {"path": str(request_path.resolve()), "requests": count, "bytes": size}
        )

    plan: dict[str, Any] = {
        "users": len(records),
        "requests": len(records),
        "estimated_input_tokens": sum(
            estimate_tokens(SYSTEM_PROMPT + row["body"]["input"][1]["content"])
            for row in rows
        ),
        "reserved_output_tokens": len(records) * config.summaries.max_output_tokens,
        "files": request_entries,
        "records": records,
        "model": config.summaries.model,
        "endpoint": "/v1/responses",
        "retry_of": source["fingerprint"],
        "prompt_version": config.summaries.prompt_version,
    }
    plan["fingerprint"] = stable_hash(plan)
    plan_path = output_dir / "request_plan.json"
    atomic_write_json(plan_path, plan)
    return plan_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan_path = rebuild_retry_plan(args.config, args.source_plan, args.output_dir)
    print(plan_path)


if __name__ == "__main__":
    main()
