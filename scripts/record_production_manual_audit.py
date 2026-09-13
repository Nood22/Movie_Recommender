"""Record the completed stratified semantic audit for the 1,000-user calibration.

This is evaluation-only code. The listed IDs are manual audit findings and are
never imported by or supplied to the production repair protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


INTERNAL_CONTRADICTION_IDS = {
    8742,
    18515,
    50674,
    60875,
    65242,
    83310,
    92810,
    118834,
    159382,
    166368,
    199226,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"Refusing to overwrite manual audit: {args.output}")
    queue = [
        json.loads(line)
        for line in args.queue.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(queue) != 120 or len({row["user_id"] for row in queue}) != 120:
        raise RuntimeError("Expected the frozen 120-user stratified audit queue")
    if not INTERNAL_CONTRADICTION_IDS.issubset(
        {int(row["user_id"]) for row in queue}
    ):
        raise RuntimeError("A manually adjudicated audit case is missing")
    rows = []
    for source in queue:
        user_id = int(source["user_id"])
        contradiction = user_id in INTERNAL_CONTRADICTION_IDS
        rows.append(
            {
                "user_id": user_id,
                "stratum": source["stratum"],
                "positive_grounding": (
                    "conflicting_with_supported_negative_evidence"
                    if contradiction
                    else "supported"
                ),
                "negative_grounding": "supported",
                "inappropriate_abstention": False,
                "supported_negative_evidence_omitted": False,
                "four_part_semantic_complete": True,
                "semantic_richness_acceptable": not contradiction,
                "internally_coherent": not contradiction,
                "privacy_leak": False,
                "notes": (
                    "The preserved V2 positive prose broadly praises one or more "
                    "genres that the deterministic evidence and final negative slot "
                    "classify as supported negative preferences. All four semantic "
                    "parts are present, but the editable profile is internally "
                    "contradictory."
                    if contradiction
                    else "Positive prose is traceable to the supplied history; the "
                    "negative slot matches strict genre evidence or appropriately "
                    "abstains. The profile remains semantically complete."
                ),
            }
        )
    rows.sort(key=lambda row: row["user_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )
    args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
