"""Compare genre coverage policies using saved, identical TEARS logits.

Development diagnostics only: no training, LLM calls, or reserved-test targets.
"""
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
from pilot_recommender import MATRIX_DIR, TEARS_RUN_DIR
from tears_preference_ranking import align_scores, explicit_genre_preferences
from artifacts.recommendation_reliability_20260910.previous_ranking import align_scores as old_align
from artifacts.recommendation_reliability_20260910.previous_ranking import explicit_genre_preferences as old_preferences
from scripts.evaluate_tears_quality import relevance_metrics
from tears_training.config import load_config


def coverage_scores(raw, memberships, preferences, mode):
    if mode == 'compound':
        return align_scores(raw, memberships, preferences)
    positive = sum((memberships[g].float() for g in preferences['preferred_genres'] if g in memberships), torch.zeros_like(raw))
    negative = sum((memberships[g].float() for g in preferences['avoided_genres'] if g in memberships), torch.zeros_like(raw)) > 0
    finite = raw[torch.isfinite(raw)]
    span = (finite.max() - finite.min()).clamp_min(1) + 1
    count = len(preferences['preferred_genres'])
    if mode == 'coverage_tiers':
        priority = positive - (count + 1) * negative.float()
        return raw + priority * span
    if mode == 'majority_tiers':
        priority = (positive > 0).float() + (positive > count / 2).float() - 3 * negative.float()
        return raw + priority * span
    if mode == 'bounded_within_tiers':
        head = finite.topk(min(100, finite.numel())).values
        return old_align(raw, memberships, preferences) + positive / max(count, 1) * (head[0] - head[-1])
    return old_align(raw, memberships, preferences)


def main():
    torch.set_num_threads(2)
    output = ROOT / 'artifacts/recommendation_reliability_20260910'
    catalog = pd.read_csv(MATRIX_DIR / 'catalog.csv').sort_values('modelItemId')
    movie_to_item = dict(zip(catalog.movieId, catalog.modelItemId))
    tags = [set(g.split('|')) for g in catalog.genres]
    memberships = {g: torch.tensor([g in t for t in tags]) for g in set.union(*tags)}
    onboarding = {m['movieId'] for m in json.loads((ROOT / 'movie-recommender-pilot/src/data/pilot_support20_onboarding.json').read_text())}
    cases = json.loads((ROOT / 'artifacts/recommendation_quality_fix_20260909/probes.json').read_text())
    users = pd.read_csv(MATRIX_DIR / 'users.csv')
    selected = users[users.split == 'validation'].sample(n=128, random_state=20260909).sort_values('modelUserId')
    user_ids = set(selected.userId)
    manifest = json.loads((TEARS_RUN_DIR / 'manifest.json').read_text())
    summaries = {}
    with Path(manifest['arguments']['summaries']).open() as handle:
        for line in handle:
            row = json.loads(line)
            if int(row['user_id']) in user_ids:
                summaries[int(row['user_id'])] = row['summary']
    observed = sparse.load_npz(MATRIX_DIR / 'validation_observed.npz').tocsr()
    targets = sparse.load_npz(MATRIX_DIR / 'validation_target.npz').tocsr()
    for user in selected.itertuples():
        cases.append({'id': f'validation-{user.userId}', 'summary': summaries[user.userId],
                      'observed': observed[user.modelUserId].indices.tolist(),
                      'targets': targets[user.modelUserId].indices.tolist()})
    datasets = [
        (cases, ROOT / 'artifacts/recommendation_quality_fix_20260909/evaluation'),
        (json.loads((output / 'reported_probes.json').read_text()), output / 'reported'),
    ]
    records, metrics = [], {}
    for cases, folder in datasets:
        caches = list(folder.glob('logits-*.npz'))
        assert len(caches) == 1, caches
        source = json.loads((folder / 'results.json').read_text())
        key = hashlib.sha256(json.dumps({
            'texts': [c['summary'] for c in cases], 'model': source['model'],
            'max_tokens': load_config(ROOT / 'configs/ml32m.toml').model.max_text_tokens,
        }, sort_keys=True).encode()).hexdigest()
        assert caches[0].stem == f'logits-{key}', 'Cached logits do not match the evidence/model'
        logits = torch.from_numpy(np.load(caches[0])['logits'])
        assert len(logits) == len(cases)
        for case, row in zip(cases, logits):
            raw = row[None, :].clone()
            raw[:, case.get('observed', [movie_to_item[m] for m in case.get('selected_ids', [])])] = -torch.inf
            preferences = explicit_genre_preferences(case['summary'])
            for policy in ('previous', 'coverage_tiers', 'majority_tiers', 'bounded_within_tiers', 'compound'):
                policy_preferences = old_preferences(case['summary']) if policy == 'previous' else preferences
                scores = coverage_scores(raw, memberships, policy_preferences, policy)
                ranked = scores[0].argsort(descending=True)[:50].tolist()
                if 'targets' in case:
                    value = relevance_metrics(ranked, set(case['targets']), 12)
                    value.update(relevance_metrics(ranked, set(case['targets']), 50))
                    metrics.setdefault(policy, []).append(value)
                else:
                    items = catalog.iloc[ranked[:12]]
                    records.append({'id': case['id'], 'policy': policy, 'preferences': preferences,
                        'top12': items.title.tolist(),
                        'animation_family_count': sum({'Animation', 'Children'} <= tags[i] for i in ranked[:12]),
                        'outside_onboarding_count': sum(int(m) not in onboarding for m in items.movieId)})
    aggregates = {p: {m: float(np.mean([r[m] for r in rows])) for m in rows[0]} for p, rows in metrics.items()}
    rng = np.random.default_rng(20260910)
    bootstrap_rows = rng.integers(0, len(metrics['previous']), size=(2000, len(metrics['previous'])))
    deltas = {}
    for metric in aggregates['previous']:
        paired = np.array([new[metric] - old[metric] for new, old in zip(metrics['compound'], metrics['previous'])])
        deltas[metric] = {'mean': float(paired.mean()),
            'paired_bootstrap_95_percent_interval': np.quantile(paired[bootstrap_rows].mean(1), [.025, .975]).tolist()}
    (output / 'coverage_comparison.json').write_text(json.dumps({
        'aggregate': aggregates, 'paired_changes': deltas, 'probes': records,
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), ROOT / 'tears_preference_ranking.py',
                                    output / 'previous_ranking.py')},
        'validation_users': 128, 'selection_seed': 20260909,
        'scope': 'Development validation, not reserved-test evaluation; genre counts are diagnostics only.',
    }, indent=2) + '\n')
    print(json.dumps(aggregates, indent=2))
    for r in records:
        if r['id'] in ('request-0dfcbc16-7e48-457c-8e3f-236afb0c8f1a', 'request-22a77bba-5da4-4378-b62a-f2d4590e4836'):
            print(json.dumps(r))


if __name__ == '__main__':
    main()
