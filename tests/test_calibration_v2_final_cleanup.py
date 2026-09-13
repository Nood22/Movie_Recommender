from __future__ import annotations

import json
from pathlib import Path

from tears_training.calibration_v2_final_cleanup import run


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_final_cleanup_from_repaired_v2(tmp_path: Path) -> None:
    output = tmp_path / "v002_final_cleanup_offline"
    report = run(output)

    summaries = _jsonl(output / "final_summaries.jsonl")
    audits = _jsonl(output / "per_user_final_cleanup_audit.jsonl")
    abstention_repairs = _jsonl(output / "abstention_repairs.jsonl")
    formatting_repairs = _jsonl(output / "formatting_repairs.jsonl")

    assert len(summaries) == len(audits) == 100
    assert sum(row["changed_from_repaired_v2"] for row in summaries) == 32
    assert len(abstention_repairs) == 7
    assert len(formatting_repairs) == 26
    assert all(row["summary"].startswith("Summary:") for row in summaries)

    final = report["final_candidate"]
    assert final["positive_grounding"]["supported"] == 100
    assert final["negative_grounding"]["supported"] == 100
    assert final["negative_grounding"]["overstated"] == 0
    assert final["negative_grounding"]["unsupported"] == 0
    assert final["abstentions"]["inappropriate"] == 0
    assert final["abstentions"]["appropriate"] == 75
    assert final["supported_negative_evidence_accidentally_omitted"]["count"] == 0
    assert final["formatting"]["failures"] == 0
    assert final["literal_placeholders"]["count"] == 0
    assert final["duplicates"]["exact_group_count"] == 0
    assert final["duplicates"]["near_pair_count"] == 0
    assert report["length"]["materially_compressed_relative_to_raw_v2"][
        "count"
    ] == 15
    assert report["length"]["materially_compressed_relative_to_raw_v2"][
        "all_semantically_complete"
    ]
    assert report["promotion_decision"]["eligible_as_next_offline_profile_candidate"]
    assert report["methodology"]["openai_api_calls"] == 0
    assert report["methodology"]["tmdb_used"] is False
    assert report["methodology"]["tears_training_started"] is False
    assert (output / "manifest.json").is_file()

    for repair in abstention_repairs:
        assert repair["evidence"]["supported_negative_genres"]
        assert repair["unsupported_semantic_claims_added"] == []
