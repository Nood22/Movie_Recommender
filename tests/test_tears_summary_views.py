import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import pilot_api
from tears_preference_ranking import explicit_genre_preferences
from tears_summary_views import NO_NEGATIVE_SENTENCES, _backend_request, apply_patches, generate_backend, synchronize_edit


def test_sparse_fallback_preserves_animation_family_scope_after_rejected_drafts():
    payload = pilot_api.SummaryRequest.model_construct(movies=[pilot_api.SummaryMovieEvidence(
        title='Soul (2020)', rating=5, genres=['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy'])],
        disliked=[], context='')
    lines, evidence = pilot_api._online_generation_evidence(payload)
    recommender = SimpleNamespace(config=SimpleNamespace(summaries=SimpleNamespace(min_words=140, max_words=180)))
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        output_text=json.dumps({'summary': 'Invalid draft'}))))
    attempts = []
    summary, _ = pilot_api._generate_valid_tears_summary(client, lines, recommender,
        ['Soul (2020)'], attempts, evidence)
    assert attempts[-1]['kind'] == 'validated_sparse_fallback'
    assert explicit_genre_preferences(summary)['preferred_genre_combinations'] == [['Animation', 'Children']]
    assert pilot_api._summary_validation_errors(summary, recommender, ['Soul (2020)'], evidence) == []


def test_deleting_preference_preserves_unrelated_hidden_details():
    original = "Summary: They enjoy comedy and romance. They favor friendships. They dislike horror."
    result = apply_patches(original, [{"old": " and romance", "new": ""}])
    assert result == "Summary: They enjoy comedy. They favor friendships. They dislike horror."


def test_reversal_replaces_claim_without_losing_hidden_dislike():
    original = "Summary: They enjoy comedy. They dislike horror."
    assert apply_patches(original, [{"old": "enjoy comedy", "new": "dislike comedy"}]) == (
        "Summary: They dislike comedy. They dislike horror."
    )


@pytest.mark.parametrize("patches", [
    [{"old": "missing", "new": "new"}],
    [{"old": "comedy", "new": "drama"}, {"old": "enjoy comedy", "new": "prefer romance"}],
])
def test_invalid_patches_fail_closed(patches):
    with pytest.raises(ValueError):
        apply_patches("Summary: They enjoy comedy.", patches)


def test_unchanged_display_uses_full_backend_without_model_call():
    assert synchronize_edit(None, "unused", "full backend", "short display", "short display") == (
        "full backend", []
    )


def test_invalid_mapping_is_repaired_automatically():
    outputs = iter([
        {"patches": [{"old": "nonexistent", "new": "drama"}]},
        {"patches": [{"old": "comedy", "new": "drama"}]},
    ])
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(next(outputs)))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result, _ = synchronize_edit(client, "test", "Summary: They enjoy comedy and friendship.",
                                 "They enjoy comedy.", "They enjoy drama.")
    assert result == "Summary: They enjoy drama and friendship."
    assert len(calls) == 2
    assert "repair_error" in calls[1]["input"][-1]["content"]


def test_source_summary_is_scoped_to_session():
    store = SimpleNamespace(event_by_request=lambda *_: {
        "participant_id": "other", "session_id": "session", "system": "TEARS",
    })
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(study_store=store)))
    payload = SimpleNamespace(summary_source_request_id="source", study=SimpleNamespace(
        participant_id="viewer", session_id="session", system="TEARS"))
    with pytest.raises(HTTPException) as error:
        pilot_api._recommendation_summary(payload, request)
    assert error.value.status_code == 409


@pytest.mark.parametrize("edited", [
    "Summary: They enjoy fantasy and adventure.",
    "Summary: They dislike fantasy and enjoy comedy.",
    "Summary: They prefer stories about friendship.",
])
def test_shared_profile_edits_reach_model_verbatim_without_synchronization(edited):
    original = "Summary: They enjoy fantasy and adventure."
    source = {"participant_id": "viewer", "session_id": "session", "system": "TEARS",
              "payload": {"backend_summary": original, "representation": original,
                          "summary_policy": pilot_api.SUMMARY_POLICY}}
    # No synchronization cache or LLM exists in this stub: new profiles need neither.
    store = SimpleNamespace(event_by_request=lambda *_: source)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(study_store=store)))
    payload = SimpleNamespace(summary=edited, summary_source_request_id="source",
        study=SimpleNamespace(participant_id="viewer", session_id="session", system="TEARS"))
    assert pilot_api._recommendation_summary(payload, request) == (edited, [])


def test_explicit_dislike_cannot_be_silently_dropped_or_reversed():
    payload = pilot_api.SummaryRequest.model_construct(
        movies=[pilot_api.SummaryMovieEvidence(title="Private Film", rating=5, genres=["Comedy"])],
        disliked=["Horror"], context="")
    lines, evidence = pilot_api._online_generation_evidence(payload)
    assert "positive-only preference profile" not in " ".join(lines)
    recommender = SimpleNamespace(config=SimpleNamespace(summaries=SimpleNamespace(min_words=140, max_words=180)))
    validate = lambda text: pilot_api._summary_validation_errors(text, recommender, [], evidence)
    assert not validate("Summary: The viewer may enjoy comedy. They dislike horror.")
    assert "grounding_missing_negative:Horror" in validate("Summary: The viewer may enjoy comedy.")
    assert "grounding_missing_negative:Horror" in validate("Summary: The viewer may enjoy comedy and horror.")


def test_neutral_history_needs_no_generation_or_invented_preference():
    payload = pilot_api.SummaryRequest.model_construct(
        movies=[pilot_api.SummaryMovieEvidence(title="Private Film", rating=3, genres=["Comedy"])],
        disliked=[], context="")
    lines, evidence = pilot_api._online_generation_evidence(payload)
    recommender = SimpleNamespace(config=SimpleNamespace(summaries=SimpleNamespace(min_words=140, max_words=180)))
    summary, attempts = pilot_api._generate_valid_tears_summary(None, lines, recommender, [], [], evidence)
    assert summary == "Summary: The viewer is still exploring their movie preferences."
    assert attempts[-1]["kind"] == "validated_neutral_profile"


def test_positive_history_has_a_validated_fallback_after_failed_drafts():
    payload = pilot_api.SummaryRequest.model_construct(
        movies=[pilot_api.SummaryMovieEvidence(title=f"Private Film {index}", rating=5,
                                              genres=["Adventure", "Fantasy"]) for index in range(3)],
        disliked=[], context="")
    lines, evidence = pilot_api._online_generation_evidence(payload)
    recommender = SimpleNamespace(config=SimpleNamespace(summaries=SimpleNamespace(min_words=140, max_words=180)))
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: SimpleNamespace(
        output_text=json.dumps({"summary": "Summary: The viewer dislikes horror."}))))
    summary, attempts = pilot_api._generate_valid_tears_summary(client, lines, recommender, [], [], evidence)
    assert summary == "Summary: The viewer enjoys adventure, fantasy films."
    assert attempts[-1]["kind"] == "validated_positive_genre_fallback"
    assert not pilot_api._summary_validation_errors(summary, recommender, [], evidence)


def test_backend_enforces_length_and_structure():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps({"summary": "Summary: They like comedy."}))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    with pytest.raises(HTTPException) as error:
        generate_backend(client, "test", {}, [], lambda _: [])
    assert "word_count" in error.value.detail
    assert "format_parts" in error.value.detail
    assert len(calls) == 3


def test_no_negative_evidence_uses_validated_fixed_abstentions():
    from tears_training.evidence_gated_summary_harness_v11 import build_separated_evidence, validate_summary_contract
    evidence = build_separated_evidence({"user_id": 0, "history_hash": "test",
        "history_items": 1, "movie_ids": [1], "titles": ["Barbie (2023)"],
        "ratings": [5], "genres": ["Comedy"]})
    sentences = [
        "Summary: The viewer may enjoy comedy with a playful tone, although this narrow tendency does not establish a general preference across every form of comedy or imply the same response to all stories within that broad genre.",
        "No strong preference is supported for specific plot points or themes, leaving the viewer's taste in narrative details unspecified beyond that limited tendency without identifying particular character relationships, story developments, or recurring subjects as established personal preferences.",
    ]
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps({"positive_sentences": sentences}))
    result, _ = generate_backend(SimpleNamespace(responses=SimpleNamespace(create=create)), "test",
        {"classified_evidence": evidence["inference_payload"]}, ["Barbie (2023)"],
        lambda text: ["grounding_none_fabricated_dislike"] if validate_summary_contract(text, evidence).none_fabricated_dislike else [])
    assert result.endswith(" ".join(NO_NEGATIVE_SENTENCES))
    assert 140 <= len(result.split()) <= 180
    assert result.count(".") == 4
    assert "positive_sentences" in calls[0]["text"]["format"]["schema"]["properties"]


def test_sparse_backend_survives_three_rating_leaking_drafts():
    from tears_training.evidence_gated_summary_harness_v11 import build_separated_evidence, validate_summary_contract
    evidence = build_separated_evidence({"user_id": 0, "history_hash": "test",
        "history_items": 1, "movie_ids": [1], "titles": ["Barbie (2023)"],
        "ratings": [5], "genres": ["Comedy"]})
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
        output_text=json.dumps({"positive_sentences": ["Summary: They rated comedy five stars.", "They dislike horror."]}))))
    result, attempts = generate_backend(client, "test", {"classified_evidence": evidence["inference_payload"]},
        ["Barbie (2023)"], lambda text: ["grounding_none_fabricated_dislike"]
        if validate_summary_contract(text, evidence).none_fabricated_dislike else [])
    assert len(attempts) == 4
    assert attempts[-1]["kind"] == "validated_sparse_fallback"
    assert "rated" not in result and "horror" not in result
    assert 140 <= len(result.split()) <= 180


def test_sparse_display_survives_three_audit_language_drafts():
    from tears_training.evidence_gated_summary_harness_v11 import build_separated_evidence
    evidence = build_separated_evidence({"user_id": 0, "history_hash": "test",
        "history_items": 1, "movie_ids": [1], "titles": ["Barbie (2023)"],
        "ratings": [5], "genres": ["Comedy"]})
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
        output_text=json.dumps({"summary": "Summary: The viewer's evidence is insufficient."}))))
    recommender = SimpleNamespace(config=SimpleNamespace(summaries=SimpleNamespace(min_words=140, max_words=180)))
    result, attempts = pilot_api._generate_valid_tears_summary(client, [], recommender,
        ["Barbie (2023)"], [], evidence)
    assert "may enjoy comedy" in result
    assert attempts[-1]["kind"] == "validated_sparse_fallback"


@pytest.mark.parametrize("evidence", [
    {"classified_evidence": {"negative_evidence_status": "STRONG"}},
    {"classified_evidence": {"negative_evidence_status": "WEAK"}},
    {"classified_evidence": {"negative_evidence_status": "NONE"}, "explicit_dislikes": ["Horror"]},
])
def test_supported_dislikes_are_not_replaced_with_abstentions(evidence):
    assert _backend_request(evidence)[2] is False
