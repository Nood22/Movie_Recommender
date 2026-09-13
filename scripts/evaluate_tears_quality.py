"""Read-only TEARS quality harness: frozen probes and validation relevance.

No recommendation API calls, training, test targets, or LLM judging. Genre
coverage is a diagnostic, not relevance. Validation relevance uses held-out
positive ratings, with the same candidate mask for every ranking variant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pilot_recommender import MATRIX_DIR, TEARS_RUN_DIR, PilotHybridRecommender
from tears_preference_ranking import align_scores, explicit_genre_preferences
from scripts.tears_quality_policies import bounded_align_scores, similarity_align_scores
from artifacts.recommendation_quality_fix_20260909.legacy_ranking import explicit_genre_preferences as legacy_preferences


def legacy_align(logits, memberships, preferences):
    positive = torch.zeros_like(logits, dtype=torch.bool)
    negative = torch.zeros_like(logits, dtype=torch.bool)
    for name in preferences["preferred_genres"]:
        if name in memberships:
            positive |= memberships[name]
    for name in preferences["avoided_genres"]:
        if name in memberships:
            negative |= memberships[name]
    finite = logits[torch.isfinite(logits)]
    if not finite.numel():
        return logits.clone()
    return logits + (positive.float() - 2 * negative.float()) * (
        (finite.max() - finite.min()).clamp_min(1) + 1)


POLICIES = {"raw": lambda scores, *_: scores, "legacy_tiers": legacy_align,
            "serving": align_scores, "bounded_coverage": bounded_align_scores, "bounded_similarity": similarity_align_scores}


def relevance_metrics(ranked, targets, k):
    """Binary held-out relevance; undefined when no target is eligible."""
    if not targets:
        return None
    hits = np.asarray([int(item in targets) for item in ranked[:k]])
    discounts = 1 / np.log2(np.arange(2, len(hits) + 2))
    ideal = (1 / np.log2(np.arange(2, min(k, len(targets)) + 2))).sum()
    return {f"ndcg@{k}": float((hits * discounts).sum() / ideal),
            f"recall@{k}": float(hits.sum() / len(targets))}


def score_texts(model, texts, cache_dir):
    key = hashlib.sha256(json.dumps({"texts": texts, "model": model.status()["models"]["tears"],
                                    "max_tokens": model.config.model.max_text_tokens},
                                   sort_keys=True).encode()).hexdigest()
    cache = cache_dir / f"logits-{key}.npz"
    if cache.exists():
        return torch.from_numpy(np.load(cache)["logits"])
    batches = []
    for start in range(0, len(texts), 4):
        encoded = model.tokenizer(texts[start:start + 4], padding="max_length",
            truncation=True, max_length=model.config.model.max_text_tokens, return_tensors="pt")
        with torch.inference_mode():
            logits, _ = model.tears(encoded.input_ids.to(model.device), encoded.attention_mask.to(model.device))
        batches.append(logits.cpu())
        print(f"Encoded {min(start + 4, len(texts))}/{len(texts)} profiles", flush=True)
    result = torch.cat(batches)
    np.savez_compressed(cache, logits=result.numpy())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-users", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--probes", type=Path, default=ROOT / "artifacts/recommendation_quality_fix_20260909/probes.json")
    args = parser.parse_args()
    if args.validation_users < 0:
        parser.error("--validation-users must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (Path(__file__), ROOT / "tears_preference_ranking.py",
                                  ROOT / "artifacts/recommendation_quality_fix_20260909/legacy_ranking.py")}
    torch.set_num_threads(2)
    model = PilotHybridRecommender(device="cpu")
    memberships = {genre: torch.tensor([genre in str(value).split("|") for value in model.catalog.genres])
                   for genre in model.genre_names}
    onboarding = json.loads((ROOT / "movie-recommender-pilot/src/data/pilot_support20_onboarding.json").read_text())
    onboarding_items = [model.movie_to_item[int(movie["movieId"])] for movie in onboarding]
    fixture_path = args.probes
    probes = json.loads(fixture_path.read_text())
    cases = list(probes)
    targets_by_case = {}
    observed_by_case = {}
    if args.validation_users:
        users = pd.read_csv(MATRIX_DIR / "users.csv")
        selected = users[users.split == "validation"].sample(
            n=args.validation_users, random_state=args.seed).sort_values("modelUserId")
        user_ids = set(selected.userId.astype(int))
        manifest = json.loads((TEARS_RUN_DIR / "manifest.json").read_text())
        summaries = {}
        with Path(manifest["arguments"]["summaries"]).open() as handle:
            for line in handle:
                row = json.loads(line)
                if int(row["user_id"]) in user_ids:
                    summaries[int(row["user_id"])] = row["summary"]
        observed = sparse.load_npz(MATRIX_DIR / "validation_observed.npz").tocsr()
        targets = sparse.load_npz(MATRIX_DIR / "validation_target.npz").tocsr()
        for user in selected.itertuples():
            name = f"validation-{user.userId}"
            cases.append({"id": name, "summary": summaries[user.userId], "validation_user": int(user.userId)})
            targets_by_case[name] = set(targets[int(user.modelUserId)].indices.tolist())
            observed_by_case[name] = observed[int(user.modelUserId)].indices.tolist()
    texts = [case["summary"] for case in cases]
    logits = score_texts(model, texts, args.output)
    output = {"model": model.status()["models"]["tears"], "seed": args.seed,
              "source_sha256": source_hashes,
              "matrix_manifest_sha256": hashlib.sha256((MATRIX_DIR / "manifest.json").read_bytes()).hexdigest(),
              "validation_users": args.validation_users,
              "probe_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
              "metric_scope": "Validation targets are held-out ratings. Probe genre metrics are diagnostics only. No test targets used.",
              "cases": [], "aggregate": {}, "edit_pairs": []}
    aggregates = {}
    probe_rankings = {}
    for index, case in enumerate(cases):
        name = case["id"]
        preferences = explicit_genre_preferences(case["summary"])
        for candidates in ("legacy_2015_onboarding", "selected_only_all_years"):
            mask = torch.ones(model.item_count, dtype=torch.bool)
            if candidates == "legacy_2015_onboarding":
                mask &= model.release_years.cpu() >= 2015
                mask[onboarding_items] = False
            excluded = observed_by_case.get(name, [model.movie_to_item[movie] for movie in case.get("selected_ids", [])])
            mask[excluded] = False
            raw = logits[index:index + 1].clone()
            raw[:, ~mask] = -torch.inf
            eligible_targets = targets_by_case.get(name, set()) & set(mask.nonzero().flatten().tolist())
            for policy, scorer in POLICIES.items():
                # Legacy scaling intentionally includes the original full catalog.
                if policy in {"legacy_tiers", "serving"}:
                    scores = scorer(logits[index:index + 1], memberships, legacy_preferences(case["summary"]) if policy == "legacy_tiers" else preferences)
                    scores[:, ~mask] = -torch.inf
                else:
                    scores = scorer(raw, memberships, preferences)
                ranked = scores[0].argsort(descending=True)[:50].tolist()
                ranked = [item for item in ranked if mask[item]]
                record = {"id": name, "candidates": candidates, "ranking": policy}
                if name in targets_by_case:
                    metrics = relevance_metrics(ranked, eligible_targets, 12)
                    if metrics is not None:
                        metrics.update(relevance_metrics(ranked, eligible_targets, 50))
                    record.update({"eligible_targets": len(eligible_targets), "metrics": metrics})
                    key = f"{candidates}/{policy}"
                    aggregates.setdefault(key, []).append(metrics)
                else:
                    items = model.catalog.iloc[ranked[:12]]
                    tags = [set(value.split("|")) for value in items.genres]
                    positive = set(case.get("expected_positive", []))
                    negative = set(case.get("expected_negative", []))
                    record.update({"summary": case["summary"], "extracted_preferences": preferences,
                        "top12": [{"movie_id": int(row.movieId), "title": row.title, "genres": row.genres,
                                   "raw_score": float(logits[index, int(row.modelItemId)])} for row in items.itertuples()],
                        "all_preferred_fraction": sum(positive <= tag for tag in tags) / len(tags) if positive and tags else None,
                        "avoided_fraction": sum(bool(negative & tag) for tag in tags) / len(tags) if negative and tags else None})
                    probe_rankings[(name, candidates, policy)] = ranked[:12]
                output["cases"].append(record)
    for key, values in aggregates.items():
        valid = [value for value in values if value is not None]
        output["aggregate"][key] = {"users_with_eligible_targets": len(valid), "users_without_eligible_targets": len(values) - len(valid),
            **{metric: float(np.mean([value[metric] for value in valid])) for metric in (valid[0] if valid else [])}}
    for case in probes:
        if not case.get("edit_of"):
            continue
        for candidates in ("legacy_2015_onboarding", "selected_only_all_years"):
            for policy in POLICIES:
                before = probe_rankings[(case["edit_of"], candidates, policy)]
                after = probe_rankings[(case["id"], candidates, policy)]
                output["edit_pairs"].append({"before": case["edit_of"], "after": case["id"],
                    "candidates": candidates, "ranking": policy, "top12_overlap": len(set(before) & set(after)) / 12})
    (args.output / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
