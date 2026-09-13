"""Third frozen 100-user calibration with evidence-calibrated negatives.

The request plan is derived from immutable v2 artifacts. This module has no
full-cohort or training operation.
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
from . import final_summaries_v2 as v2_module
from .artifacts import atomic_write_bytes, sha256_file, stable_hash


PROTOCOL_VERSION = (
    "ml32m-support20-emiliano-v3-evidence-calibrated-negative-grounding-"
    "gpt5mini-2025-08-07"
)
V1_PROMPT = v2_module.V1_PROMPT
V2_SAFEGUARD = v2_module.SAFEGUARD
V2_PROMPT = v2_module.V2_PROMPT
V3_POLICY = """Apply the following evidence-calibrated negative-preference policy:
If there is NO reliable negative evidence in the supplied rating history, do not invent any disliked genre, style, theme, or plot preference. Explicitly state that the available history does not support a strong negative preference.
If negative evidence is WEAK or isolated, describe only the narrow negative tendency directly supported by those ratings. Use cautious language such as "may be less interested in" or "shows limited evidence of preferring". Do not generalize one or two low-rated films into broad dislikes of unrelated genres, styles, themes, or plot types.
If negative evidence is STRONG and repeated, describe the supported negative preferences directly, but only at the level justified by the observed pattern. Do not add additional dislikes merely to fill the negative-preference sections.
Never infer a negative preference from absence of positive ratings alone.
Never invent negative preferences in order to make the four-part structure look complete.
An explicit evidence-based abstention is preferable to an unsupported dislike claim.
The model should still describe supported negative preferences when clear evidence exists. Do NOT use a blanket statement that there are no strong dislikes when the history contains repeated, coherent negative evidence.
For grounding guidance only, use these evidence categories: None means no supplied rating at or below 2.5. Weak means one or two supplied ratings at or below 2.5 without meeting the strong condition. Strong means at least three supplied ratings at or below 2.5, or at least two supplied ratings at or below 1.5. Do not mention threshold numbers or rating values in the summary."""
V3_PROMPT = V1_PROMPT + "\n" + V3_POLICY
V3_PROMPT_SHA256 = hashlib.sha256(V3_PROMPT.encode("utf-8")).hexdigest()


def configure_core() -> None:
    core.PROTOCOL_VERSION = PROTOCOL_VERSION
    core.EMILIANO_SYSTEM_PROMPT = V3_PROMPT
    core.EMILIANO_PROMPT_SHA256 = V3_PROMPT_SHA256
    core.EXPECTED_EMILIANO_PROMPT_SHA256 = V3_PROMPT_SHA256


def write_immutable_text(path: Path, text: str) -> None:
    payload = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Refusing to overwrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def plan_from_v2(v2_plan_path: Path, output_dir: Path) -> dict[str, Any]:
    configure_core()
    reference = core._load_verified_plan(v2_plan_path)
    if reference["requests"] != 100 or reference["unique_users"] != 100:
        raise RuntimeError("V2 reference is not exactly 100 users")
    repository_root = Path(__file__).resolve().parents[1]
    protocol = core.protocol_manifest(repository_root)
    allowed = {"version", "system_prompt", "system_prompt_sha256", "fingerprint"}
    changed = sorted(
        key
        for key in set(reference["protocol"]) | set(protocol)
        if reference["protocol"].get(key) != protocol.get(key)
    )
    unexpected = sorted(set(changed) - allowed)
    if unexpected:
        raise RuntimeError(f"Unexpected protocol changes: {unexpected}")
    if reference["protocol"]["system_prompt"] != V2_PROMPT:
        raise RuntimeError("V2 prompt bytes differ from the frozen v2 prompt")
    if protocol["system_prompt"] != V1_PROMPT + "\n" + V3_POLICY:
        raise RuntimeError("V3 prompt construction changed")

    source_requests = load_jsonl(Path(reference["request_file"]["path"]))
    source_by_id = {row["custom_id"]: row for row in source_requests}
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for source_record in reference["records"]:
        source = source_by_id[source_record["custom_id"]]
        body = deepcopy(source["body"])
        inputs = body.get("input") or []
        if len(inputs) != 2 or inputs[0] != {"role": "system", "content": V2_PROMPT}:
            raise RuntimeError("Unexpected v2 input construction")
        body["input"][0]["content"] = V3_PROMPT
        cache_key = stable_hash(
            {
                "user_id": source_record["user_id"],
                "history_hash": source_record["history_hash"],
                "protocol_fingerprint": protocol["fingerprint"],
            }
        )
        custom_id = f"cal-user-{source_record['user_id']}-{cache_key[:16]}-a0"
        target = {
            "custom_id": custom_id,
            "method": source["method"],
            "url": source["url"],
            "body": body,
        }
        left, right = deepcopy(source), deepcopy(target)
        left.pop("custom_id")
        right.pop("custom_id")
        left["body"]["input"][0]["content"] = "<SYSTEM_PROMPT>"
        right["body"]["input"][0]["content"] = "<SYSTEM_PROMPT>"
        if left != right:
            raise RuntimeError(
                f"Request changed beyond system prompt for {source_record['user_id']}"
            )
        row_record = deepcopy(source_record)
        row_record["custom_id"] = custom_id
        row_record["cache_key"] = cache_key
        rows.append(target)
        records.append(row_record)
        checks.append(
            {
                "user_id": source_record["user_id"],
                "history_hash": source_record["history_hash"],
                "v2_custom_id": source_record["custom_id"],
                "v3_custom_id": custom_id,
                "only_body_change": "system prompt content",
            }
        )
    if len(rows) != 100 or len({row["custom_id"] for row in rows}) != 100:
        raise RuntimeError("V3 plan is not exactly 100 unique requests")

    output_dir = output_dir.resolve()
    request_path = output_dir / "requests" / "calibration-100.jsonl"
    count, size = core._write_immutable_jsonl(request_path, rows)
    plan: dict[str, Any] = {
        "scope": "calibration_only",
        "expected_requests": 100,
        "requests": count,
        "unique_users": len({row["user_id"] for row in records}),
        "seed": reference["seed"],
        "protocol": protocol,
        "dataset": deepcopy(reference["dataset"]),
        "sample": deepcopy(reference["sample"]),
        "request_file": {
            "path": str(request_path),
            "requests": count,
            "bytes": size,
            "sha256": sha256_file(request_path),
        },
        "records": records,
        "estimated_input_tokens": sum(
            core.estimate_tokens(
                row["body"]["input"][0]["content"]
                + row["body"]["input"][1]["content"]
            )
            for row in rows
        ),
        "reserved_output_tokens": reference["reserved_output_tokens"],
        "v2_reference": {
            "plan_path": str(v2_plan_path.resolve()),
            "plan_fingerprint": reference["fingerprint"],
            "request_sha256": reference["request_file"]["sha256"],
            "user_ids_sha256": reference["sample"]["user_ids_sha256"],
        },
    }
    plan["fingerprint"] = stable_hash(plan)
    core._write_immutable_json(output_dir / "request_plan.json", plan)
    core._write_immutable_json(output_dir / "protocol.json", protocol)
    write_immutable_text(output_dir / "prompt" / "emiliano_system_prompt_v1.txt", V1_PROMPT)
    write_immutable_text(output_dir / "prompt" / "emiliano_system_prompt_v2.txt", V2_PROMPT)
    write_immutable_text(output_dir / "prompt" / "emiliano_system_prompt_v3.txt", V3_PROMPT)
    core._write_immutable_json(
        output_dir / "prompt" / "protocol_delta.json",
        {
            "actual_changed_top_level_fields": changed,
            "unexpected_changed_top_level_fields": unexpected,
            "v1_prompt_sha256": hashlib.sha256(V1_PROMPT.encode()).hexdigest(),
            "v2_prompt_sha256": reference["protocol"]["system_prompt_sha256"],
            "v3_prompt_sha256": V3_PROMPT_SHA256,
            "removed_v2_policy": V2_SAFEGUARD,
            "added_v3_policy": V3_POLICY,
            "prompt_construction": "v1 prompt bytes + one newline + v3 policy bytes",
            "per_request_verification": checks,
        },
    )
    core._write_immutable_json(
        output_dir / "prompt" / "provenance.json",
        {
            "protocol_version": protocol["version"],
            "protocol_fingerprint": protocol["fingerprint"],
            "prompt_sha256": V3_PROMPT_SHA256,
            "system_message": V3_PROMPT,
            "negative_grounding_policy": V3_POLICY,
            "source": protocol["source"],
            "user_history_construction": protocol["history"],
            "generation": protocol["generation"],
            "validation": protocol["validation"],
            "v2_reference": plan["v2_reference"],
        },
    )
    core._write_immutable_json(
        output_dir / "calibration_user_ids.json",
        {
            "seed": plan["seed"],
            "count": 100,
            "user_ids": plan["sample"]["user_ids"],
            "user_ids_sha256": plan["sample"]["user_ids_sha256"],
            "exact_match_to_v1_and_v2": True,
            "v2_reference": plan["v2_reference"],
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
    parser = argparse.ArgumentParser(prog="python -m tears_training.final_summaries_v3")
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan-from-v2")
    plan.add_argument("--v2-plan", type=Path, required=True)
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
    configure_core()
    if args.action == "plan-from-v2":
        result = plan_from_v2(args.v2_plan, args.output_dir)
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
        result = core.validate_calibration(args.plan, None, write_artifacts=not args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
