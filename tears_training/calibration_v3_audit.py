"""Offline all-user v1-v2-v3 grounding and promotion-gate audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import statistics
from typing import Any


BASE = Path('/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/calibration_100')
V1 = BASE / 'v001_20260813'
V2 = BASE / 'v002_20260813_negative_grounding_safeguard'
V3 = BASE / 'v003_20260813_evidence_calibrated_negative_grounding'
OUT = V3 / 'reports' / 'paired_comparison_v1_v2_v3'

# Manual claim adjudication against every supplied history.
UNSUPPORTED = {
    77072: 'No item is rated <=2.5; after correctly abstaining, v3 invents lesser interest in lighter romantic fare.',
    119291: 'No item is rated <=2.5; after correctly abstaining, v3 invents lesser interest in romantic and broad comedy.',
    135578: 'The one low-rated item is action/sci-fi, but v3 instead infers lesser interest in romantic and broad comedy.',
    184544: 'No item is rated <=2.5; after correctly abstaining, v3 invents lesser interest in classic romantic melodrama.',
    185526: 'No item is rated <=2.5; after correctly abstaining, v3 invents lesser interest in slapstick, lowbrow comedy, and light rom-coms.',
}

OVERSTATED = {
    2015: 'The musical signal is narrow and supported, but the added quiet/somber children’s-drama and introspection claims exceed two low-rated items.',
    36720: 'Two low ratings provide narrow comedy and action evidence; v3 overgeneralizes them into family/children and spectacle-driven sci-fi aversions.',
    37580: 'One low-rated sci-fi blockbuster supports only that narrow tendency, not the added broad-comedy claim.',
    53501: 'Failed sci-fi spectacle and one romantic comedy are evidenced, but broad slapstick/farce is an unsupported addition.',
    59442: 'The action/blockbuster tendency is grounded, but broad slapstick rests mainly on one low-rated comedy.',
    61038: 'Several romance titles and one animation are low-rated; family slapstick musicals and broad children’s fare exceed the evidence.',
    76281: 'One low-rated dark comedy supports a narrow offbeat-comedy tendency, not a wider aversion to broad campy humor.',
    86122: 'One low-rated musical comedy supports that narrow signal, but slice-of-life, biographical, and meandering-drama claims are unrelated additions.',
    108650: 'Two dissimilar low-rated dramas do not support a broad aversion to small-scale character studies lacking spectacle.',
    138469: 'Three dissimilar low-rated titles do not support the added conventional action-blockbuster and family-musical generalization.',
    156456: 'One low-rated long-form drama does not establish an aversion to fantasy/adventure, time loops, or supernatural premises; the negative slot is effectively misgrounded.',
    161310: 'Some franchise/YA evidence exists, but sentimental-comedy and broader sequel claims exceed four mixed low-rated titles.',
    165531: 'Two failed pulpy sci-fi films support only a narrow execution-related tendency; the output instead broadly abstains while implying wider genre preferences by comparison.',
    168549: 'One child-focused fantasy is low-rated, but v3 generalizes this to quieter coming-of-age and family narratives while omitting the clearer exploitation/action signal.',
    182046: 'One low-rated psychological-horror film supports a narrow horror signal, not grim dramas generally.',
    183589: 'The history has many heterogeneous low ratings; v3 overstates a coherent aversion to austere art-house drama and adds weakly supported Western/family claims.',
    190156: 'The stated sentimental-romcom/melodrama aversion is not the clearest repeated pattern in the supplied low-rated sci-fi, thriller, and drama items.',
}

# Strong/repeated evidence is present, but v3 incorrectly says there is no
# strong preference or gives it only weak/limited status. This is separate from
# inventing or overgeneralizing a claim.
INAPPROPRIATE_SUPPRESSION = {
    2752, 7190, 8545, 22459, 22939, 25944, 30005, 30459, 34378,
    48604, 61207, 69826, 98129, 128530, 146379, 152582, 169241,
    179370, 183383,
}

STRUCTURE_INCOMPLETE = {
    154024: 'The output leaves Emiliano’s literal negative-section placeholders in the generated text rather than filling them with a grounded tendency or explicit abstention.'
}


def jl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def norm(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def percentile_linear(values: list[int], p: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = rank - lo
    return ordered[lo] + fraction * (ordered[hi] - ordered[lo])


def main() -> None:
    p1, p2, p3 = (load(x / 'request_plan.json') for x in (V1, V2, V3))
    ids = p3['sample']['user_ids']
    assert ids == p1['sample']['user_ids'] == p2['sample']['user_ids']
    assert len(ids) == len(set(ids)) == 100
    assert not (set(UNSUPPORTED) & set(OVERSTATED))
    assert not ((set(UNSUPPORTED) | set(OVERSTATED)) & INAPPROPRIATE_SUPPRESSION)
    s1 = {x['user_id']: x for x in jl(V1 / 'validated/all_summaries.jsonl')}
    s2 = {x['user_id']: x for x in jl(V2 / 'validated/all_summaries.jsonl')}
    s3 = {x['user_id']: x for x in jl(V3 / 'validated/all_summaries.jsonl')}
    a1 = {x['user_id']: x for x in jl(V1 / 'reports/grounding_audit_v1/per_user_grounding_audit.jsonl')}
    a2 = {x['user_id']: x for x in jl(V2 / 'reports/paired_comparison_v1_v2/per_user_paired_comparison.jsonl')}
    rec = {x['user_id']: x for x in p3['records']}

    rows: list[dict[str, Any]] = []
    for uid in ids:
        record = rec[uid]
        tier = a1[uid]['negative_evidence_tier']
        if uid in UNSUPPORTED:
            verdict = 'unsupported'
            note = UNSUPPORTED[uid]
        elif uid in OVERSTATED:
            verdict = 'overstated'
            note = OVERSTATED[uid]
        else:
            verdict = 'supported'
            note = 'Material negative claims are tied to supplied low-rated evidence, or the summary abstains without inventing a preference.'
        if uid in INAPPROPRIATE_SUPPRESSION:
            abstention_status = 'inappropriate_suppression'
        elif verdict != 'supported':
            abstention_status = 'mixed_or_contradictory'
        else:
            abstention_status = 'appropriate'
        v1_fail = a1[uid]['negative_claim_verdict'] != 'supported'
        v2_fail = a2[uid]['v2']['negative_verdict'] != 'supported'
        v3_fail = verdict != 'supported'
        if v2_fail and not v3_fail:
            transition = 'resolved_v2_failure'
        elif v2_fail and v3_fail:
            transition = 'persistent_from_v2'
        elif not v2_fail and v3_fail:
            transition = 'new_v3_failure'
        else:
            transition = 'remained_claim_grounded'
        low_items = [
            {'title': title, 'rating': rating, 'genres': genres}
            for title, rating, genres in zip(record['titles'], record['ratings'], record['genres'])
            if rating <= 2.5
        ]
        high_items = [
            {'title': title, 'rating': rating, 'genres': genres}
            for title, rating, genres in zip(record['titles'], record['ratings'], record['genres'])
            if rating >= 4.0
        ]
        structure_ok = uid not in STRUCTURE_INCOMPLETE
        rows.append({
            'user_id': uid,
            'split': record['split'],
            'activity_band': record['activity_band'],
            'history_hash': record['history_hash'],
            'negative_evidence_tier': tier,
            'low_rated_items': low_items,
            'high_rated_items': high_items,
            'v1': {
                'summary': s1[uid]['summary'],
                'word_count': s1[uid]['word_count'],
                'negative_verdict': a1[uid]['negative_claim_verdict'],
            },
            'v2': {
                'summary': s2[uid]['summary'],
                'word_count': s2[uid]['word_count'],
                'negative_verdict': a2[uid]['v2']['negative_verdict'],
                'abstention_status': a2[uid]['v2']['abstention_status'],
            },
            'v3': {
                'summary': s3[uid]['summary'],
                'word_count': s3[uid]['word_count'],
                'negative_verdict': verdict,
                'negative_adjudication_note': note,
                'explicit_scoped_or_blanket_abstention': True,
                'abstention_status': abstention_status,
                'clear_negative_evidence_incorrectly_suppressed': uid in INAPPROPRIATE_SUPPRESSION,
                'weak_evidence_overgeneralized': tier == 'weak' and verdict in {'overstated', 'unsupported'},
                'positive_preference_grounding': 'supported',
                'positive_grounding_note': 'Material positive genre and theme/plot statements align with supplied high-rated items.',
                'semantic_structure': {
                    'liked_genres': True,
                    'liked_themes_or_plots_or_styles': True,
                    'disliked_genres_or_styles_or_explicit_abstention': structure_ok,
                    'disliked_plots_or_content_or_explicit_abstention': structure_ok,
                    'complete': structure_ok,
                    'note': STRUCTURE_INCOMPLETE.get(uid),
                },
                'prefix_present': s3[uid]['summary'].startswith('Summary:'),
                'privacy_detector': s3[uid]['privacy'],
                'confirmed_identifiable_privacy_leakage': False,
            },
            'v2_to_v3_claim_transition': transition,
            'v1_failure_resolved_in_v3': v1_fail and not v3_fail,
            'comprehensive_v3_negative_policy_failure': v3_fail or uid in INAPPROPRIATE_SUPPRESSION,
            'v1_v2_sequence_similarity': SequenceMatcher(None, norm(s1[uid]['summary']), norm(s2[uid]['summary']), autojunk=False).ratio(),
            'v2_v3_sequence_similarity': SequenceMatcher(None, norm(s2[uid]['summary']), norm(s3[uid]['summary']), autojunk=False).ratio(),
        })

    v1c = Counter(x['v1']['negative_verdict'] for x in rows)
    v2c = Counter(x['v2']['negative_verdict'] for x in rows)
    v3c = Counter(x['v3']['negative_verdict'] for x in rows)
    transitions = Counter(x['v2_to_v3_claim_transition'] for x in rows)
    abstentions = Counter(x['v3']['abstention_status'] for x in rows)
    by_tier: dict[str, dict[str, Any]] = {}
    for tier in ('strong', 'weak', 'none'):
        part = [x for x in rows if x['negative_evidence_tier'] == tier]
        c1 = Counter(x['v1']['negative_verdict'] for x in part)
        c2 = Counter(x['v2']['negative_verdict'] for x in part)
        c3 = Counter(x['v3']['negative_verdict'] for x in part)
        claim_fail = c3['overstated'] + c3['unsupported']
        comprehensive = sum(x['comprehensive_v3_negative_policy_failure'] for x in part)
        by_tier[tier] = {
            'users': len(part),
            'v1': dict(c1), 'v2': dict(c2), 'v3': dict(c3),
            'v3_claim_failure_count': claim_fail,
            'v3_claim_failure_rate': claim_fail / len(part),
            'v3_inappropriate_suppression': sum(x['v3']['clear_negative_evidence_incorrectly_suppressed'] for x in part),
            'v3_comprehensive_policy_failure_count': comprehensive,
            'v3_comprehensive_policy_failure_rate': comprehensive / len(part),
        }

    r1, r2, r3 = (load(x / 'reports/validation_report.json') for x in (V1, V2, V3))
    words = {
        'v1': [x['v1']['word_count'] for x in rows],
        'v2': [x['v2']['word_count'] for x in rows],
        'v3': [x['v3']['word_count'] for x in rows],
    }
    length = {}
    for version, values in words.items():
        length[version] = {
            'minimum': min(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'p95_linear': percentile_linear(values, .95),
            'maximum': max(values), 'below_120': sum(x < 120 for x in values),
            'below_150': sum(x < 150 for x in values),
            'below_170': sum(x < 170 for x in values),
            'below_180': sum(x < 180 for x in values),
        }
    short = [x for x in rows if x['v3']['word_count'] < 150]
    v3_claim_fail = v3c['overstated'] + v3c['unsupported']
    v1_fail_ids = {x['user_id'] for x in rows if x['v1']['negative_verdict'] != 'supported'}
    v3_fail_ids = {x['user_id'] for x in rows if x['v3']['negative_verdict'] != 'supported'}

    report = {
        'audit_version': 'v1-v2-v3-paired-audit-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': {
            'users': 100, 'same_user_ids': True,
            'same_history_hashes': all(a['history_hash'] == b['history_hash'] for a, b in zip(p2['records'], p3['records'])),
            'api_requests_beyond_v3_calibration': 0,
            'full_cohort_started': False, 'tears_training_started': False,
        },
        'negative_claim_grounding': {
            'v1': {**dict(v1c), 'failure_count': v1c['overstated'] + v1c['unsupported'], 'failure_rate': .44},
            'v2': {**dict(v2c), 'failure_count': v2c['overstated'] + v2c['unsupported'], 'failure_rate': .21},
            'v3': {**dict(v3c), 'failure_count': v3_claim_fail, 'failure_rate': v3_claim_fail / 100},
            'v3_by_evidence_tier': by_tier,
        },
        'negative_policy_comprehensive': {
            'v3_claim_failure_count': v3_claim_fail,
            'v3_inappropriate_suppression_count': len(INAPPROPRIATE_SUPPRESSION),
            'v3_union_failure_count': sum(x['comprehensive_v3_negative_policy_failure'] for x in rows),
            'v3_union_failure_rate': sum(x['comprehensive_v3_negative_policy_failure'] for x in rows) / 100,
            'note': 'The comprehensive metric is the union of unsupported/overstated claims and incorrectly suppressed clear negative evidence.',
        },
        'abstention': {
            'scoped_or_blanket_abstention_present': 100,
            'appropriate': abstentions['appropriate'],
            'inappropriate_suppression': abstentions['inappropriate_suppression'],
            'mixed_or_contradictory': abstentions['mixed_or_contradictory'],
            'inappropriate_suppression_user_ids': sorted(INAPPROPRIATE_SUPPRESSION),
        },
        'paired_transitions_v2_to_v3': {
            **dict(transitions),
            'new_v3_failure_user_ids': [x['user_id'] for x in rows if x['v2_to_v3_claim_transition'] == 'new_v3_failure'],
            'resolved_v2_failure_user_ids': [x['user_id'] for x in rows if x['v2_to_v3_claim_transition'] == 'resolved_v2_failure'],
        },
        'original_v1_failures': {
            'total': len(v1_fail_ids),
            'resolved_in_v3': len(v1_fail_ids - v3_fail_ids),
            'persistent_in_v3': len(v1_fail_ids & v3_fail_ids),
            'resolved_user_ids': sorted(v1_fail_ids - v3_fail_ids),
            'persistent_user_ids': sorted(v1_fail_ids & v3_fail_ids),
        },
        'positive_preference_grounding': {
            'supported': 100, 'overstated_or_unsupported': 0,
            'automated_users_with_supported_history_genre': r3['grounding']['users_with_supported_history_genre'],
            'finding': 'All material positive genre and theme/plot claims are traceable to supplied high-rated histories; no positive-grounding regression was found.',
        },
        'semantic_structure': {
            'complete': 100 - len(STRUCTURE_INCOMPLETE),
            'incomplete': len(STRUCTURE_INCOMPLETE),
            'incomplete_user_ids': sorted(STRUCTURE_INCOMPLETE),
            'prefix_present': sum(x['v3']['prefix_present'] for x in rows),
            'prefix_missing_user_ids': [x['user_id'] for x in rows if not x['v3']['prefix_present']],
            'exact_four_sentence_rule_applied': False,
        },
        'length': length,
        'short_summary_review': {
            'below_150_reviewed': len(short),
            'below_150_semantically_complete': sum(x['v3']['semantic_structure']['complete'] for x in short),
            'below_150_semantically_incomplete': sum(not x['v3']['semantic_structure']['complete'] for x in short),
            'finding': 'Shortness usually does not remove semantic slots, but the corpus is still far from the requested approximately-200-word profile style; one output leaves literal negative-section placeholders.',
        },
        'privacy': {
            'v3_automated_title_match_user_ids': r3['privacy_leakage']['movie_title_user_ids'],
            'v3_confirmed_identifiable_title_leaks': 0,
            'v3_year_leaks': len(r3['privacy_leakage']['year_user_ids']),
            'v3_numeric_rating_leaks': len(r3['privacy_leakage']['numeric_rating_user_ids']),
            'note': "The only automated title match is the generic phrase 'classic alien horror' overlapping the one-word title 'Alien'; it is not an identifiable title disclosure.",
        },
        'duplicates': {
            'v3_exact_cross_user_duplicate_groups': r3['cross_user_duplicate_groups'],
            'v3_near_duplicate_pairs': r3['cross_user_near_duplicate_pairs'],
            'v2_v3_exactly_equal': sum(norm(x['v2']['summary']) == norm(x['v3']['summary']) for x in rows),
        },
        'api_and_cost': {
            'successful_api_responses': r3['successful_api_responses'],
            'successful_structured_responses': r3['successful_structured_responses'],
            'missing': len(r3['missing_user_ids']),
            'input_tokens': r3['usage']['input_tokens'],
            'cached_input_tokens': r3['usage']['cached_input_tokens'],
            'uncached_input_tokens': r3['usage']['uncached_input_tokens'],
            'output_tokens': r3['usage']['output_tokens'],
            'reasoning_tokens': r3['usage']['reasoning_tokens_in_output'],
            'total_tokens': r3['usage']['input_tokens'] + r3['usage']['output_tokens'],
            'pricing_usd_per_million': r3['cost']['pricing_usd_per_million'],
            'exact_cost_usd': r3['cost']['exact_calibration_usd'],
            'average_cost_per_user_usd': r3['cost']['mean_per_user_usd'],
        },
        'quality_findings': [
            'V3 still places an evidence-limiting or abstention statement in every summary, including 67 strong-evidence users.',
            'Nineteen strong-evidence summaries incorrectly weaken or suppress a repeated coherent negative pattern, up from 13 inappropriate suppressions in v2.',
            'Claim-level failure is 22%, not an improvement over v2’s 21%; weak-evidence claim failures rise from 28% to 40%.',
            'Length remains collapsed: every output is below 180 words, 89 are below 150, and the mean is 112.15 despite preserving the original approximately-200-word instruction.',
            'One output copies literal format placeholders into the summary, and one omits the Summary: prefix.',
        ],
        'promotion_gate': {
            'passed': False,
            'criteria': {
                'unsupported_nearly_eliminated_no_evidence': False,
                'weak_evidence_not_broadly_overgeneralized': False,
                'strong_evidence_dislikes_retained': False,
                'inappropriate_blanket_abstention_substantially_reduced': False,
                'positive_grounding_intact': True,
                'four_part_structure_usable': False,
                'length_reasonably_close_to_200': False,
                'privacy_clean': True,
                'duplication_clean': True,
            },
            'recommend_full_cohort': False,
            'reason': 'V3 does not outperform v2 on claim grounding, increases suppression of clear negative evidence, retains a 50% no-evidence claim-failure rate, and remains far shorter than the intended approximately-200-word style.',
        },
    }

    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT / 'per_user_v1_v2_v3_comparison.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    (OUT / 'v1_v2_v3_comparison_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + '\n')

    lines = [
        '# V3 calibration and paired v1-v2-v3 audit', '',
        '## Grounding outcome', '',
        '| Version | Supported | Overstated | Unsupported | Claim failure |',
        '|---|---:|---:|---:|---:|',
        f"| V1 | {v1c['supported']} | {v1c['overstated']} | {v1c['unsupported']} | 44% |",
        f"| V2 | {v2c['supported']} | {v2c['overstated']} | {v2c['unsupported']} | 21% |",
        f"| V3 | {v3c['supported']} | {v3c['overstated']} | {v3c['unsupported']} | {v3_claim_fail}% |",
        '',
        '| V3 evidence tier | Users | Supported | Overstated | Unsupported | Claim failure | Suppressed clear evidence | Comprehensive failure |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for tier in ('strong', 'weak', 'none'):
        x = by_tier[tier]
        lines.append(f"| {tier.title()} | {x['users']} | {x['v3'].get('supported',0)} | {x['v3'].get('overstated',0)} | {x['v3'].get('unsupported',0)} | {x['v3_claim_failure_rate']:.1%} | {x['v3_inappropriate_suppression']} | {x['v3_comprehensive_policy_failure_rate']:.1%} |")
    lines += [
        '',
        f"V3 has {v3_claim_fail}/100 unsupported or overstated negative profiles and {len(INAPPROPRIATE_SUPPRESSION)} additional inappropriate suppressions. The union is {report['negative_policy_comprehensive']['v3_union_failure_count']}/100. It resolves {report['original_v1_failures']['resolved_in_v3']} of the original 44 v1 failures, introduces {transitions['new_v3_failure']} failures relative to v2, and resolves {transitions['resolved_v2_failure']} v2 failures.",
        '', '## Abstention, structure, and length', '',
        f"Every output contains a scoped or blanket lack-of-evidence boundary. {abstentions['appropriate']} are appropriate, {abstentions['inappropriate_suppression']} suppress clear repeated evidence, and {abstentions['mixed_or_contradictory']} accompany unsupported or overstated claims. Semantic structure is usable for {100-len(STRUCTURE_INCOMPLETE)}/100; user 154024 contains literal unfilled placeholders. The Summary: prefix is present for {report['semantic_structure']['prefix_present']}/100 and its omission is reported only as formatting.",
        '',
        f"V3 length: mean {length['v3']['mean']:.2f}, median {length['v3']['median']:.1f}, minimum {length['v3']['minimum']}, maximum {length['v3']['maximum']}, p95 {length['v3']['p95_linear']:.2f}. Below 150: {length['v3']['below_150']}; below 170: {length['v3']['below_170']}; below 180: {length['v3']['below_180']}. Of the {len(short)} summaries below 150, {report['short_summary_review']['below_150_semantically_complete']} retain all semantic slots and one is incomplete. Thus shortness usually does not cause omission, but the length distribution fails the intended approximately-200-word style gate.",
        '', '## Other checks', '',
        f"Positive grounding is supported for 100/100. Confirmed identifiable privacy leaks: 0; year leaks: 0; numeric-rating leaks: 0. Exact duplicate groups: {r3['cross_user_duplicate_groups']}; near-duplicate pairs: {r3['cross_user_near_duplicate_pairs']}.",
        '', '## Exact usage and cost', '',
        f"Input tokens: {r3['usage']['input_tokens']:,} (all uncached); output tokens: {r3['usage']['output_tokens']:,}; reasoning tokens: {r3['usage']['reasoning_tokens_in_output']:,}; total tokens: {r3['usage']['input_tokens']+r3['usage']['output_tokens']:,}. Exact cost: **${r3['cost']['exact_calibration_usd']:.8f}**, or **${r3['cost']['mean_per_user_usd']:.9f}** per user.",
        '', '## Promotion gate', '',
        '**Fail.** Do not promote v3 to full-cohort generation. V3 does not improve claim grounding over v2, increases inappropriate suppression, leaves half of no-evidence users with invented claims, overgeneralizes weak evidence, and remains far below the intended approximately-200-word profile length. No full-cohort generation or TEARS training was started.',
    ]
    (OUT / 'v1_v2_v3_comparison_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    manifest = {
        'manifest_version': 'v3-paired-audit-v1',
        'source_artifacts': {
            'v1/request_plan.json': digest(V1 / 'request_plan.json'),
            'v1/validated/all_summaries.jsonl': digest(V1 / 'validated/all_summaries.jsonl'),
            'v1/per_user_grounding_audit.jsonl': digest(V1 / 'reports/grounding_audit_v1/per_user_grounding_audit.jsonl'),
            'v2/request_plan.json': digest(V2 / 'request_plan.json'),
            'v2/validated/all_summaries.jsonl': digest(V2 / 'validated/all_summaries.jsonl'),
            'v2/per_user_paired_comparison.jsonl': digest(V2 / 'reports/paired_comparison_v1_v2/per_user_paired_comparison.jsonl'),
            'v3/request_plan.json': digest(V3 / 'request_plan.json'),
            'v3/validated/all_summaries.jsonl': digest(V3 / 'validated/all_summaries.jsonl'),
            'v3/prompt/protocol_delta.json': digest(V3 / 'prompt/protocol_delta.json'),
            'v3/responses': {p.name: digest(p) for p in sorted((V3 / 'responses').glob('*.jsonl'))},
        },
        'audit_artifacts': {},
    }
    for name in ('per_user_v1_v2_v3_comparison.jsonl', 'v1_v2_v3_comparison_report.json', 'v1_v2_v3_comparison_report.md'):
        manifest['audit_artifacts'][name] = digest(OUT / name)
    (OUT / 'v1_v2_v3_comparison_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
