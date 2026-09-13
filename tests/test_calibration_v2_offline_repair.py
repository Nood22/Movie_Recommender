from __future__ import annotations

import json
from pathlib import Path

from tears_training.calibration_v2_offline_repair import run


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_frozen_v2_offline_repair(tmp_path: Path) -> None:
    output = tmp_path / "v002_grounding_repair_offline"
    report = run(output)

    rows = _jsonl(output / "per_user_repair_audit.jsonl")
    summaries = _jsonl(output / "repaired_summaries.jsonl")
    assert len(rows) == len(summaries) == 100
    assert sum(row["changed"] for row in rows) == 21
    assert sum(not row["changed"] for row in rows) == 79
    assert all(
        row["repaired_grounding_classification"] == "supported" for row in rows
    )
    assert report["comparison_to_raw_v2"][
        "v2_negative_grounding_failure_rate"
    ] == 0.21
    assert report["comparison_to_raw_v2"][
        "repaired_v2_negative_grounding_failure_rate"
    ] == 0.0
    assert report["comparison_to_raw_v2"]["new_failures_introduced"] == 0
    assert report["repaired_v2"]["supported_content_accidentally_removed"] == 0
    assert report["methodology"]["openai_api_calls"] == 0
    assert report["methodology"]["tmdb_used"] is False
    assert report["methodology"]["tears_training_started"] is False
    assert (output / "manifest.json").is_file()

    changed = {row["user_id"]: row for row in rows if row["changed"]}
    assert set(changed) == set(report["repaired_v2"]["changed_user_ids"])
    assert changed[183589]["changes"][0]["replacement_span"] == (
        "The user does not enjoy Action-Adventure films."
    )
    assert "Mystery" not in changed[183589]["changes"][0]["replacement_span"]
