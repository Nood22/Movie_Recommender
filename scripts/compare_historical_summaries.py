"""Recover August 25 generation and compare profiles with raw frozen TEARS.

Generation writes resumable experiment artifacts, never modifies live serving.
Ranking deliberately calls the model directly, bypassing the online reranker.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REVISION = "1754778"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def literal(source, name):
    for node in ast.parse(source).body:
        names = [target.id for target in getattr(node, "targets", []) if isinstance(target, ast.Name)]
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
        if name in names:
            return ast.literal_eval(node.value)
    raise ValueError(name)


def prepare(out):
    if (out / "experiment.json").exists():
        return json.loads((out / "experiment.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    recovered = {}
    for name in ("pilot_api.py", "tears_training/summaries.py", "configs/ml32m.toml"):
        source = subprocess.check_output(["git", "show", f"{REVISION}:{name}"], cwd=ROOT, text=True)
        destination = out / "recovered" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)
        recovered[name] = source
    with sqlite3.connect(f"file:{ROOT}/logs/pilot_study.sqlite3?mode=ro", uri=True) as db:
        def event(request_id, kind):
            row = db.execute("SELECT recorded_at,payload_json FROM events WHERE request_id=? AND event_type=?",
                             (request_id, kind)).fetchone()
            return row[0], json.loads(row[1])
        cases = []
        for case_id, request_id in (
            ("Soul", "request-0b7e6a3b-ae7c-42b6-8a5c-dad60f35fa79"),
            ("Barbie", "request-94708077-28f7-4681-86b5-9f6808cd0088"),
        ):
            _, rec = event(request_id, "recommendation_result")
            source_id = rec["effective_model_input"]["summary_source_request_id"]
            date, summary = event(source_id, "summary_result")
            original_input = summary["effective_generation_input"]
            cases.append({"id": case_id, "movies": original_input["movies"],
                "disliked": original_input["disliked"], "context": original_input["context"],
                "excluded_ids": rec["request"]["excluded_movie_ids"],
                "saved": [{"version": "saved_current_backend", "summary": summary["backend_summary"],
                           "recorded_at": date, "request_id": source_id},
                          {"version": "saved_current_display", "summary": summary["representation"],
                           "recorded_at": date, "request_id": source_id}]})
        archive_id = "request-positive-2da7ebc6-13f3-49b7-b10b-aacd98019432"
        date, archived = event(archive_id, "summary_result")
        cases.append({"id": "Soul_mixed_history", "movies": archived["preference_evidence"],
            "disliked": [], "context": archived.get("context", ""),
            "excluded_ids": cases[0]["excluded_ids"],
            "saved": [{"version": "archived_20260826", "summary": archived["representation"],
                       "recorded_at": date, "request_id": archive_id}]})
    experiment = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "historical_commit": subprocess.check_output(["git", "rev-parse", REVISION], cwd=ROOT, text=True).strip(),
        "historical_system_prompt": literal(recovered["tears_training/summaries.py"], "SYSTEM_PROMPT"),
        "historical_model": literal(recovered["pilot_api.py"], "SUMMARY_MODEL"),
        "historical_schema": literal(recovered["tears_training/summaries.py"], "SUMMARY_SCHEMA"),
        "historical_settings": {"min_words": 120, "max_words": 260, "reasoning": "minimal",
                                "verbosity": "low", "max_output_tokens": 450, "validation": "nonempty_only"},
        "current_source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("pilot_api.py", "tears_summary_views.py", "pilot_recommender.py")},
        "cases": cases,
    }
    write_json(out / "experiment.json", experiment)
    return experiment


def generate(out, experiment, repeats):
    import pilot_api
    from openai import OpenAI
    from tears_summary_views import generate_backend
    from tears_training.summaries import render_history_prompt

    def run(case, version, repeat):
        destination = out / "generated" / f"{case['id']}-{version}-{repeat}.json"
        if destination.exists():
            return f"cached {destination.stem}"
        client = OpenAI()
        result = {"case": case["id"], "version": version, "repeat": repeat, "outputs": []}
        try:
            if version == "historical_regenerated":
                lines = [f"- title={m['title']!r}; private_rating={m['rating']:g}/5; genres={'|'.join(m['genres']) or 'Unknown'}"
                         for m in case["movies"]]
                if case["disliked"]:
                    lines.append("- explicitly_disliked=" + "|".join(case["disliked"]))
                if case["context"]:
                    lines.append("- viewing_context=" + case["context"])
                response = client.responses.create(
                    model=experiment["historical_model"],
                    input=[{"role": "system", "content": experiment["historical_system_prompt"]},
                           {"role": "user", "content": render_history_prompt("\n".join(lines), 120, 260)}],
                    reasoning={"effort": "minimal"}, text={"format": {"type": "json_schema",
                        "name": "viewer_profile", "strict": True, "schema": experiment["historical_schema"]},
                        "verbosity": "low"}, max_output_tokens=450, store=False)
                result["raw_response"] = response.output_text
                summary = str(json.loads(response.output_text)["summary"]).strip()
                if not summary:
                    raise ValueError("empty summary")
                result["outputs"].append({"version": version, "summary": summary})
            else:
                payload = pilot_api.SummaryRequest.model_construct(
                    movies=[pilot_api.SummaryMovieEvidence(**movie) for movie in case["movies"]],
                    disliked=case["disliked"], context=case["context"])
                lines, separated = pilot_api._online_generation_evidence(payload)
                recommender = SimpleNamespace(config=SimpleNamespace(summaries=SimpleNamespace(min_words=120, max_words=260)))
                titles = [movie["title"] for movie in case["movies"]]
                attempts = []
                result["backend_attempts"] = attempts
                backend, _ = generate_backend(client, pilot_api.SUMMARY_MODEL,
                    {"classified_evidence": separated["inference_payload"],
                     "explicit_dislikes": case["disliked"], "context": case["context"]},
                    titles, lambda text: [e for e in pilot_api._summary_validation_errors(
                        text, recommender, titles, separated) if e.startswith("grounding_")], attempts=attempts)
                result["outputs"].append({"version": "current_backend", "summary": backend})
                display_attempts = []
                result["display_attempts"] = display_attempts
                display, _ = pilot_api._generate_valid_tears_summary(client, lines, recommender,
                    titles, display_attempts, separated)
                result["outputs"].append({"version": "current_display", "summary": display})
        except Exception as error:
            result["error"] = getattr(error, "detail", str(error))
        write_json(destination, result)
        return f"{destination.stem}: {'ERROR ' + str(result['error']) if 'error' in result else 'generated'}"

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, case, version, repeat) for case in experiment["cases"]
                   for version in ("historical_regenerated", "current") for repeat in range(1, repeats + 1)]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def rank(out, experiment):
    import torch
    from pilot_recommender import PilotHybridRecommender, EXPECTED_CHECKPOINT_SHA256
    from tears_summary_views import NO_NEGATIVE_SENTENCES
    torch.set_num_threads(2)
    recommender = PilotHybridRecommender()
    rows = []
    errors = []
    for case in experiment["cases"]:
        for saved in case["saved"]:
            rows.append({"case": case["id"], "repeat": 0, **saved})
    for path in sorted((out / "generated").glob("*.json")):
        generated = json.loads(path.read_text())
        if generated.get("error"):
            errors.append({"file": path.name, "error": generated["error"]})
        rows.extend({"case": generated["case"], "repeat": generated["repeat"], **output}
                    for output in generated["outputs"])
    suffix = " ".join(NO_NEGATIVE_SENTENCES)
    for row in list(rows):
        if row["version"] in {"current_backend", "saved_current_backend"} and row["summary"].endswith(suffix):
            rows.append({**row, "version": row["version"] + "_without_negative_boilerplate",
                         "summary": row["summary"][:-len(suffix)].strip(),
                         "intervention": "Remove only exact fixed negative abstention suffix; preserve first two sentences verbatim"})
    cases = {case["id"]: case for case in experiment["cases"]}
    for offset in range(0, len(rows), 4):
        batch = rows[offset:offset + 4]
        encoded = recommender.tokenizer([row["summary"] for row in batch], padding="max_length",
            truncation=True, max_length=recommender.config.model.max_text_tokens, return_tensors="pt")
        with torch.inference_mode():
            logits, _ = recommender.tears(encoded.input_ids.to(recommender.device), encoded.attention_mask.to(recommender.device))
        for index, row in enumerate(batch):
            items = recommender._ranked_items(logits[index:index+1], cases[row["case"]]["excluded_ids"], 12, 2015)
            row.update(words=len(row["summary"].split()), recommendations=items,
                       animation_count=sum("Animation" in item["genres"] for item in items),
                       comedy_count=sum("Comedy" in item["genres"] for item in items))
        print(f"raw model scored {min(offset + 4, len(rows))}/{len(rows)} profiles", flush=True)
    write_json(out / "rankings.json", {"scoring": "raw_tears_no_reranking", "min_year": 2015,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256["tears_base"], "rows": rows, "errors": errors})
    report(out, experiment, rows, errors)


def report(out, experiment, rows, errors):
    lines = ["# Historical versus current TEARS summary comparison", "",
        "Recovered source: August 25 commit `" + experiment["historical_commit"] + "`.", "",
        "Historical generation requested 120–260 words and four sentences, using minimal reasoning, low verbosity, 450 output tokens, and only a nonempty-response check. Regenerated outputs are fresh samples using those settings, not recovered historical outputs.", "",
        "All profiles below were scored by the same frozen TEARS checkpoint, with raw logits, the same current onboarding exclusions, and release year >=2015. The genre reranker was bypassed. The saved August 26 profile is an original artifact; its recommendations here are newly rescored, not recovered historical rankings.", "",
        "Counts of Animation/Comedy tags are diagnostic metadata counts, not ground-truth relevance or NDCG measurements.", "",
        "The additional without_negative_boilerplate variants remove only the exact two fixed negative-abstention sentences from the current backend; the preceding preference text is unchanged. These are offline diagnostic variants, not deployed summaries.", "",
        "| Case | Version | Run | Words | Animation /12 | Comedy /12 |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['version']} | {row['repeat']} | {row['words']} | {row['animation_count']} | {row['comedy_count']} |")
    lines.extend(["", "## Summaries and raw recommendations", ""])
    for row in rows:
        lines.extend([f"### {row['case']} — {row['version']} — run {row['repeat']}", "",
                      row["summary"], "", "Recommendations:", ""])
        lines.extend(f"{item['rank']}. {item['title']}" for item in row["recommendations"])
        lines.append("")
    if errors:
        lines.extend(["## Generation failures", "", "```json", json.dumps(errors, indent=2), "```", ""])
    (out / "comparison.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "generate", "rank"])
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/summary_history_comparison_20260906")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    experiment = prepare(args.out)
    if args.phase == "generate":
        generate(args.out, experiment, args.repeats)
    elif args.phase == "rank":
        rank(args.out, experiment)
    else:
        print(json.dumps({"historical_commit": experiment["historical_commit"],
                          "cases": [case["id"] for case in experiment["cases"]]}))


if __name__ == "__main__":
    main()
