"""Check deployed ranking and refresh reuse using an explicitly synthetic session."""
import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pilot_api
from scripts.smoke_pilot_candidate_policy import request_json, recommendation_payload
from study_runtime import PROTOCOL_ID, PROTOCOL_VERSION, signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', required=True, help='Backend base URL, without /api')
    args = parser.parse_args()
    output = ROOT / 'artifacts/recommendation_reliability_20260910'
    health = request_json(args.api, '/api/health')
    assert health['tears_ranking_policy'] == pilot_api.TEARS_RANKING_POLICY
    assert health['online_task1a']['source_sha256_at_process_start'] == pilot_api.PILOT_API_SOURCE_SHA256
    (output / 'health_after.json').write_text(json.dumps(health, indent=2) + '\n')
    participant = 'synthetic-refresh-reliability-' + uuid4().hex
    session = 'refresh-' + uuid4().hex

    def metadata(task, snapshot):
        return {'protocol_id': PROTOCOL_ID, 'protocol_version': PROTOCOL_VERSION,
                'participant_id': participant, 'session_id': session, 'system': 'TEARS',
                'task_id': task, 'representation_revision': 1, 'request_id': uuid4().hex,
                'input_signature': signature(snapshot)}

    catalog = {'catalog_fingerprint': pilot_api.EXPECTED_MATRIX_FINGERPRINT,
               'onboarding_fingerprint': pilot_api.EXPECTED_ONBOARDING_FINGERPRINT,
               'items': json.loads(pilot_api.ONBOARDING_PATH.read_text())}
    cases = json.loads((output / 'reported_probes.json').read_text())
    results = {'participant': participant, 'exclude_from_study_analysis': True, 'replays': []}
    for original_id in ('request-0dfcbc16-7e48-457c-8e3f-236afb0c8f1a', 'request-22a77bba-5da4-4378-b62a-f2d4590e4836'):
        case = next(c for c in cases if c['id'] == original_id)
        payload = recommendation_payload('TEARS', case['summary'], catalog)
        payload['liked_movie_ids'] = case['selected_ids']
        payload['preference_evidence'] = [{'movie_id': m, 'rating': 5.0} for m in case['selected_ids']]
        snapshot = {k: v for k, v in payload.items() if k not in ('study', 'summary')}
        payload['study'] = metadata('1b', {'system': 'TEARS', 'representation': case['summary'], **snapshot})
        response = request_json(args.api, '/api/recommend', payload)
        items = response['items']
        assert len(items) == 12
        assert all({'Animation', 'Children'} <= set(item['genres']) for item in items)
        assert not (set(case['selected_ids']) & {item['movie_id'] for item in items})
        assert [item['rank'] for item in items] == list(range(1, 13))
        results['replays'].append({'source_request_id': original_id, 'request': payload, 'response': response})
        print('Exact reported replay passed:', original_id, flush=True)

    evidence = {'movies': [{'title': 'Soul (2020)', 'rating': 5,
                'genres': ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy']}],
                'disliked': [], 'context': ''}
    generated = []
    for _ in range(2):
        payload = {**evidence, 'study': metadata('1a', {'system': 'TEARS', **evidence})}
        generated.append(request_json(args.api, '/api/summarize', payload))
    assert generated[0]['summary'] == generated[1]['summary']
    assert generated[1]['generation']['reused'] is True
    assert generated[1]['generation']['attempt_count'] == 0
    assert generated[0]['summary_source_request_id'] != generated[1]['summary_source_request_id']
    results['refresh'] = generated
    gers_payload = recommendation_payload('GERS', ['Drama', 'Science Fiction', 'Adventure'], catalog)
    gers_payload['study']['participant_id'] = participant
    gers = request_json(args.api, '/api/gers', gers_payload)
    assert len(gers['items']) == 12
    results['gers'] = gers
    (output / 'live_smoke.json').write_text(json.dumps(results, indent=2) + '\n')
    print('Refresh reuse and GERS checks passed.', flush=True)


if __name__ == '__main__':
    main()
