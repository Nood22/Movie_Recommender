import pytest
import torch

from tears_preference_ranking import align_scores, explicit_genre_preferences
from tears_summary_views import NO_NEGATIVE_SENTENCES


@pytest.mark.parametrize("text,positive,negative", [
    ("The viewer shows a preference for lighthearted comedic fare.", ["Comedy"], []),
    ("They may enjoy comedy. " + " ".join(NO_NEGATIVE_SENTENCES), ["Comedy"], []),
    ("They enjoy comedy and dislike horror.", ["Comedy"], ["Horror"]),
    ("They dislike horror and prefer romantic comedy.", ["Comedy", "Romance"], ["Horror"]),
    ("They do not enjoy horror but enjoy science fiction.", ["Sci-Fi"], ["Horror"]),
    ("They enjoy comedy and not horror.", ["Comedy"], ["Horror"]),
    ("They have mixed preferences for comedy.", [], []),
    ("No strong preference is supported for comedy or horror.", [], []),
    ("They dislike horror. They enjoy horror.", [], []),
    ("They do not dislike horror.", [], []),
    ("They prefer stories about friendship.", [], []),
    ("The viewer seems to respond positively to high-fantasy adventure.", ["Adventure", "Fantasy"], []),
    ("They are interested in science fiction.", ["Sci-Fi"], []),
    ("They are not interested in horror.", [], []),
    ("They no longer enjoy comedy.", [], []),
    ("They may respond to animated stories and seem open to comedy.", ["Animation", "Comedy"], []),
])
def test_genre_directions_and_abstentions(text, positive, negative):
    actual = explicit_genre_preferences(text)
    assert actual["preferred_genres"] == positive
    assert actual["avoided_genres"] == negative


def test_preferred_candidates_outrank_generic_popularity_without_changing_within_group_order():
    raw = torch.tensor([[10., 8., 3., 4., 9.]])
    genres = {"Comedy": torch.tensor([False, False, True, True, True]),
              "Horror": torch.tensor([False, True, False, False, True])}
    scores = align_scores(raw, genres, explicit_genre_preferences("They enjoy comedy and dislike horror."))
    assert scores.argsort(descending=True).tolist() == [[3, 2, 0, 4, 1]]
    assert torch.equal(raw, torch.tensor([[10., 8., 3., 4., 9.]]))


def test_no_genre_direction_preserves_raw_ranking_and_masks():
    raw = torch.tensor([[float("-inf"), 4., 2.]])
    assert torch.equal(align_scores(raw, {}, explicit_genre_preferences("They prefer friendship.")), raw)


def test_reversing_preference_reverses_candidate_groups():
    raw = torch.tensor([[5., 1.]])
    genres = {"Comedy": torch.tensor([True, False]), "Horror": torch.tensor([False, True])}
    scores = align_scores(raw, genres, explicit_genre_preferences("They dislike comedy and enjoy horror."))
    assert scores.argmax().item() == 1


@pytest.mark.parametrize("summary", [
    "The viewer may enjoy films that use animation and family-friendly storytelling—particularly lighthearted, comedic adventures with whimsical or fantastical elements. They seem open to imaginative, visually driven stories aimed at younger or broad audiences, where playful tone and accessible humor are central.",
    "The viewer may be drawn to animated, family-oriented films that blend gentle comedy with fantastical, adventurous storytelling. They may appreciate warm, imaginative narratives that balance playful humor and whimsical visuals suited to children and broad family audiences.",
])
def test_reported_paraphrases_prioritize_family_animation_over_generic_adventure(summary):
    preferences = explicit_genre_preferences(summary)
    assert preferences['preferred_genre_combinations'] == [['Animation', 'Children']]
    raw = torch.tensor([[100., 3., 4., 99.]])
    membership = {'Adventure': torch.tensor([True, True, False, True]),
                  'Animation': torch.tensor([False, True, True, False]),
                  'Children': torch.tensor([False, True, True, True])}
    scores = align_scores(raw, membership, preferences)
    assert scores.argsort(descending=True).tolist() == [[2, 1, 0, 3]]


@pytest.mark.parametrize('summary', [
    'They enjoy animated family-friendly films. They also enjoy crime and horror.',
    'They enjoy animation or family-friendly films.',
    'They no longer enjoy animated family-friendly films.',
    'They have mixed preferences for animated family-friendly films.',
    'They enjoy animated family-friendly films. They dislike animation.',
    'They enjoy friendship and adventure.',
])
def test_alternatives_withdrawals_and_conflicts_do_not_create_compound_constraints(summary):
    assert explicit_genre_preferences(summary)['preferred_genre_combinations'] == []


def test_compound_never_overrides_dislike_or_selected_mask():
    raw = torch.tensor([[100., 2., 4., -torch.inf]])
    memberships = {'Romance': torch.tensor([True, True, False, True]),
                   'Comedy': torch.tensor([True, True, False, True]),
                   'Horror': torch.tensor([True, False, False, False])}
    scores = align_scores(raw, memberships, explicit_genre_preferences(
        'They enjoy romantic comedy. They dislike horror.'))
    assert scores.argsort(descending=True).tolist() == [[1, 2, 0, 3]]
