"""Second 100-user calibration with one negative-grounding safeguard.

This module derives the v2 request plan from the immutable v1 plan so the
sample and rendered histories cannot drift.  It deliberately exposes no
full-cohort operation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from . import final_summaries as core
from .artifacts import atomic_write_bytes, atomic_write_json, sha256_file, stable_hash


PROTOCOL_VERSION = (
    "ml32m-support20-emiliano-v2-negative-grounding-safeguard-"
    "gpt5mini-2025-08-07"
)
SAFEGUARD = (
    "If the user's rating history does not provide sufficient evidence for a "
    "negative preference, do not infer or invent one. Instead, state that no "
    "strong negative preference is supported by the available history."
)
V1_PROMPT = core.EMILIANO_SYSTEM_PROMPT
V2_PROMPT = V1_PROMPT + "\n" + SAFEGUARD
V2_PROMPT_SHA256 = hashlib.sha256(V2_PROMPT.encode("utf-8")).hexdigest()


def _configure_core() -> None:
    """Apply the sole generation-protocol change for this process."""
    core.PROTOCOL_VERSION = PROTOCOL_VERSION
    core.EMILIANO_SYSTEM_PROMPT = V2_PROMPT
    core.EMILIANO_PROMPT_SHA256 = V2_PROMPT_SHA256
    core.EXPECTED_EMILIANO_PROMPT_SHA256 = V2_PROMPT_SHA256


def _write_immutable_text(path: Path, value: str) -> None:
    payload = value.encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _protocol_delta(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    allowed = {"version", "system_prompt", "system_prompt_sha256", "fingerprint"}
    differing = sorted(key for key in set(v1) | set(v2) if v1.get(key) != v2.get(key))
    unexpected = sorted(set(differing) - allowed)
    if unexpected:
        raise RuntimeError(f"Unexpected protocol changes: {unexpected}")
    if v2["system_prompt"] != v1["system_prompt"] + "\n" + SAFEGUARD:
        raise RuntimeError("V2 prompt is not exactly v1 plus the safeguard")
    return {
        "allowed_changed_top_level_fields": sorted(allowed),
        "actual_changed_top_level_fields": differing,
        "unexpected_changed_top_level_fields": unexpected,
        "v1_prompt_sha256": v1["system_prompt_sha256"],
        "v2_prompt_sha256": v2["system_prompt_sha256"],
        "added_text": SAFEGUARD,
        "added_text_sha256": hashlib.sha256(SAFEGUARD.encode("utf-8")).hexdigest(),
        "prompt_construction": "v1 prompt bytes + one newline + exact safeguard bytes",
    }


def plan_from_v1(v1_plan_path: Path, output_dir: Path) -> dict[str, Any]:
    _configure_core()
    v1_plan = core._load_verified_plan(v1_plan_path)
    if v1_plan["requests"] != 100 or v1_plan["unique_users"] != 100:
        raise RuntimeError("V1 reference is not the 100-user calibration")
    repository_root = Path(__file__).resolve().parents[1]
    protocol = core.protocol_manifest(repository_root)
    delta = _protocol_delta(v1_plan["protocol"], protocol)

    v1_requests = _load_jsonl(Path(v1_plan["request_file"]["path"]))
    if len(v1_requests) != 100:
        raise RuntimeError("V1 request file is not exactly 100 rows")
    request_by_custom_id = {row["custom_id"]: row for row in v1_requests}
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    request_differences: list[dict[str, Any]] = []
    for v1_record in v1_plan["records"]:
        source = request_by_custom_id[v1_record["custom_id"]]
        body = deepcopy(source["body"])
        inputs = body.get("input") or []
        if len(inputs) != 2 or inputs[0].get("role") != "system" or inputs[1].get("role") != "user":
            raise RuntimeError("Unexpected v1 input construction")
        if inputs[0].get("content") != V1_PROMPT:
            raise RuntimeError("V1 request prompt differs from frozen v1 prompt")
        body["input"][0]["content"] = V2_PROMPT
        cache_key = stable_hash(
            {
                "user_id": v1_record["user_id"],
                "history_hash": v1_record["history_hash"],
                "protocol_fingerprint": protocol["fingerprint"],
            }
        )
        custom_id = f"cal-user-{v1_record['user_id']}-{cache_key[:16]}-a0"
        rows.append(
            {
                "custom_id": custom_id,
                "method": source["method"],
                "url": source["url"],
                "body": body,
            }
        )
        record = deepcopy(v1_record)
        record["custom_id"] = custom_id
        record["cache_key"] = cache_key
        records.append(record)
        v1_comparable = deepcopy(source)
        v2_comparable = deepcopy(rows[-1])
        v1_comparable.pop("custom_id")
        v2_comparable.pop("custom_id")
        v1_comparable["body"]["input"][0]["content"] = "<SYSTEM_PROMPT>"
        v2_comparable["body"]["input"][0]["content"] = "<SYSTEM_PROMPT>"
        if v1_comparable != v2_comparable:
            raise RuntimeError(
                f"Request changed beyond system prompt for user {v1_record['user_id']}"
            )
        request_differences.append(
            {
                "user_id": v1_record["user_id"],
                "v1_custom_id": source["custom_id"],
                "v2_custom_id": custom_id,
                "history_hash": v1_record["history_hash"],
                "only_body_change": "system prompt content",
            }
        )

    output_dir = output_dir.resolve()
    requests_path = output_dir / "requests" / "calibration-100.jsonl"
    count, size = core._write_immutable_jsonl(requests_path, rows)
    estimated_input = sum(
        core.estimate_tokens(
            row["body"]["input"][0]["content"]
            + row["body"]["input"][1]["content"]
        )
        for row in rows
    )
    plan: dict[str, Any] = {
        "scope": "calibration_only",
        "expected_requests": 100,
        "requests": count,
        "unique_users": len({record["user_id"] for record in records}),
        "seed": v1_plan["seed"],
        "protocol": protocol,
        "dataset": deepcopy(v1_plan["dataset"]),
        "sample": deepcopy(v1_plan["sample"]),
        "request_file": {
            "path": str(requests_path),
            "requests": count,
            "bytes": size,
            "sha256": sha256_file(requests_path),
        },
        "records": records,
        "estimated_input_tokens": estimated_input,
        "reserved_output_tokens": v1_plan["reserved_output_tokens"],
        "v1_reference": {
            "plan_path": str(v1_plan_path.resolve()),
            "plan_fingerprint": v1_plan["fingerprint"],
            "request_sha256": v1_plan["request_file"]["sha256"],
            "user_ids_sha256": v1_plan["sample"]["user_ids_sha256"],
        },
    }
    if plan["sample"]["user_ids"] != v1_plan["sample"]["user_ids"]:
        raise RuntimeError("V2 user IDs differ from v1")
    if len(rows) != 100 or len({row["custom_id"] for row in rows}) != 100:
        raise RuntimeError("V2 plan is not exactly 100 unique requests")
    plan["fingerprint"] = stable_hash(plan)
    core._write_immutable_json(output_dir / "request_plan.json", plan)
    core._write_immutable_json(output_dir / "protocol.json", protocol)
    _write_immutable_text(output_dir / "prompt" / "emiliano_system_prompt_v1.txt", V1_PROMPT)
    _write_immutable_text(output_dir / "prompt" / "emiliano_system_prompt_v2.txt", V2_PROMPT)
    core._write_immutable_json(
        output_dir / "prompt" / "protocol_delta.json",
        {
            **delta,
            "v1_protocol_fingerprint": v1_plan["protocol"]["fingerprint"],
            "v2_protocol_fingerprint": protocol["fingerprint"],
            "per_request_verification": request_differences,
        },
    )
    core._write_immutable_json(
        output_dir / "prompt" / "provenance.json",
        {
            "protocol_version": protocol["version"],
            "protocol_fingerprint": protocol["fingerprint"],
            "prompt_sha256": V2_PROMPT_SHA256,
            "v1_prompt_sha256": v1_plan["protocol"]["system_prompt_sha256"],
            "system_message": V2_PROMPT,
            "sole_prompt_addition": SAFEGUARD,
            "source": protocol["source"],
            "user_history_construction": protocol["history"],
            "generation": protocol["generation"],
            "validation": protocol["validation"],
            "v1_reference": plan["v1_reference"],
        },
    )
    core._write_immutable_json(
        output_dir / "calibration_user_ids.json",
        {
            "seed": plan["seed"],
            "count": 100,
            "user_ids": plan["sample"]["user_ids"],
            "user_ids_sha256": plan["sample"]["user_ids_sha256"],
            "exact_match_to_v1": True,
            "v1_reference": plan["v1_reference"],
            "split_counts": plan["sample"]["split_counts"],
            "activity_band_counts": plan["sample"]["activity_band_counts"],
            "stratum_counts": plan["sample"]["stratum_counts"],
        },
    )
    try:
        import openai
        openai_version = openai.__version__
    except (ImportError, AttributeError):
        openai_version = None
    core._write_immutable_json(
        output_dir / "runtime_environment.json",
        {
            "python_version": sys.version,
            "openai_sdk_version": openai_version,
            "api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
            "api_key_value_recorded": False,
            "repository_root": str(repository_root),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tears_training.final_summaries_v2")
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan-from-v1")
    plan.add_argument("--v1-plan", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    submit = sub.add_parser("submit-calibration")
    submit.add_argument("--plan", type=Path, required=True)
    submit.add_argument("--input-price-per-million", type=float, required=True)
    submit.add_argument("--cached-input-price-per-million", type=float, required=True)
    submit.add_argument("--output-price-per-million", type=float, required=True)
    submit.add_argument("--max-calibration-cost-usd", type=float, default=1.0)
    submit.add_argument("--confirm-calibration-spend", action="store_true")
    poll = sub.add_parser("poll-calibration")
    poll.add_argument("--submission", type=Path, required=True)
    validate = sub.add_parser("validate-calibration")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _configure_core()
    if args.action == "plan-from-v1":
        result = plan_from_v1(args.v1_plan, args.output_dir)
    elif args.action == "submit-calibration":
        result = core.submit_calibration(
            args.plan,
            None,
            args.input_price_per_million,
            args.cached_input_price_per_million,
            args.output_price_per_million,
            args.max_calibration_cost_usd,
            args.confirm_calibration_spend,
        )
    elif args.action == "poll-calibration":
        result = core.poll_calibration(args.submission, None)
    else:
        result = core.validate_calibration(
            args.plan, None, write_artifacts=not args.dry_run
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
