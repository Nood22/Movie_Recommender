"""Generate a fixed, small online summary suite without writing study events.

Uses the production prompt, evidence classifier, retry loop and validators.
Writes exact texts and validation attempts, plus input for evaluate_tears_quality.
Requires OPENAI_API_KEY; --repeat controls the explicit generation budget.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pilot_api
from fastapi import HTTPException
from openai import OpenAI


CASES = [
    ("lotr", [(4993, 5)], [], ""),
    ("barbie", [(288513, 5)], [], ""),
    ("horror", [(168250, 5)], [], ""),
    ("romance", [(168492, 5)], [], ""),
    ("animation", [(68954, 5)], [], ""),
    ("soul", [(225173, 5)], [], ""),
    ("explicit_dislike", [(4993, 5)], ["Horror"], ""),
    ("unwind", [(4993, 5)], [], "I want to unwind"),
    ("neutral", [(4993, 3)], [], ""),
    ("weak_negative", [(4993, 5), (168250, 1)], [], ""),
    ("rich_fantasy", [(4993, 5), (5952, 5), (7153, 4)], [], ""),
]
EXPECTED_GENRES = {"lotr": ["Fantasy", "Adventure"], "barbie": ["Comedy"],
                   "horror": ["Horror"], "romance": ["Romance"], "animation": ["Animation"],
                   "soul": ["Animation"], "explicit_dislike": ["Fantasy", "Adventure", "Horror"],
                   "unwind": ["Fantasy", "Adventure"], "neutral": [],
                   "weak_negative": ["Fantasy", "Adventure"], "rich_fantasy": ["Fantasy", "Adventure"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat", default=1, type=int)
    parser.add_argument("--cases", nargs="+", choices=[case[0] for case in CASES])
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    from pilot_recommender import MATRIX_DIR
    catalog = pd.read_csv(MATRIX_DIR / "catalog.csv").set_index("movieId")
    for _, ratings, _, _ in CASES:
        for movie, _ in ratings:
            if movie not in catalog.index:
                raise ValueError(f"Fixture movie {movie} is absent from the serving catalog")
    recommender = SimpleNamespace(config=SimpleNamespace(summaries=SimpleNamespace(min_words=140, max_words=180)))

    def generate(job):
        case, repeat = job
        name, ratings, dislikes, context = case
        movies = [pilot_api.SummaryMovieEvidence(title=catalog.loc[movie].title,
            rating=rating, genres=catalog.loc[movie].genres.split("|")) for movie, rating in ratings]
        payload = pilot_api.SummaryRequest.model_construct(movies=movies, disliked=dislikes, context=context)
        lines, evidence = pilot_api._online_generation_evidence(payload)
        attempts = []
        record = {"id": f"{name}-{repeat}", "evidence": [movie.model_dump() for movie in movies],
                  "selected_ids": [movie for movie, _ in ratings], "disliked": dislikes, "context": context}
        try:
            summary, _ = pilot_api._generate_valid_tears_summary(
                OpenAI(timeout=90, max_retries=0), lines, recommender,
                [movie.title for movie in movies], attempts, evidence)
            record.update({"summary": summary, "valid": True, "word_count": len(summary.split())})
            record["missing_probe_genres"] = sorted(set(EXPECTED_GENRES[name]) - pilot_api.mentioned_online_genres(summary))
        except HTTPException as error:
            record.update({"valid": False, "error": error.detail})
        record["attempts"] = attempts
        print(json.dumps({"id": record["id"], "valid": record["valid"], "attempts": len(attempts)}), flush=True)
        return record

    jobs = [(case, repeat) for case in CASES if not args.cases or case[0] in args.cases for repeat in range(args.repeat)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(generate, jobs))
    report = {"model": pilot_api.SUMMARY_MODEL, "prompt_sha256": pilot_api.ONLINE_TASK1A_PROMPT_SHA256,
              "api_source_sha256": pilot_api.PILOT_API_SOURCE_SHA256,
              "summary_policy": pilot_api.SUMMARY_POLICY, "records": records}
    (args.output / "generation.json").write_text(json.dumps(report, indent=2) + "\n")
    probes = [{key: row[key] for key in ("id", "summary", "selected_ids")}
              for row in records if row["valid"]]
    (args.output / "generated_probes.json").write_text(json.dumps(probes, indent=2) + "\n")
    if not all(row["valid"] and not row["missing_probe_genres"] for row in records):
        raise SystemExit("Some summaries failed validation or probe coverage; see generation.json")


if __name__ == "__main__":
    main()
