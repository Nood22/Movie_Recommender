"""Replay a fresh A/B/A sequence on the actual checkpoint, without API writes."""
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pilot_recommender import PilotHybridRecommender


def main():
    torch.set_num_threads(2)
    output = ROOT / 'artifacts/recommendation_reliability_20260910'
    cases = json.loads((output / 'reported_probes.json').read_text())
    case = next(c for c in cases if c['id'] == 'request-22a77bba-5da4-4378-b62a-f2d4590e4836')
    model = PilotHybridRecommender(device='cpu')
    raw = []
    hook = model.tears.register_forward_hook(lambda module, args, result: raw.append(result[0].detach().cpu().clone()))
    before = model.recommend_tears(case['summary'], case['selected_ids'], [])
    model.recommend_tears('Summary: The viewer enjoys crime and horror. They dislike comedy.', [], [])
    after = model.recommend_tears(case['summary'], case['selected_ids'], [])
    hook.remove()
    result = {
        'model': model.status()['models']['tears'],
        'all_modules_in_eval_mode': all(not m.training for m in model.tears.modules()),
        'same_raw_logits': torch.equal(raw[0], raw[2]),
        'max_absolute_logit_difference': float((raw[0] - raw[2]).abs().max()),
        'same_recommendations': before == after,
        'first_request_items': before,
    }
    (output / 'request_independence.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['same_raw_logits'] and result['same_recommendations'] and result['all_modules_in_eval_mode']
    print(json.dumps({k: v for k, v in result.items() if k not in ('model', 'first_request_items')}), flush=True)


if __name__ == '__main__':
    main()
