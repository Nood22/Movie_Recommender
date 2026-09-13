import pytest
import torch

from scripts.evaluate_tears_quality import relevance_metrics
from tears_preference_ranking import explicit_genre_preferences
from scripts.tears_quality_policies import bounded_align_scores


def test_relevance_uses_target_count_and_rank_discounts():
    assert relevance_metrics([1, 3], {1, 2}, 2) == pytest.approx({
        "ndcg@2": 1 / (1 + 1 / __import__("math").log2(3)), "recall@2": 0.5})
    assert relevance_metrics([1, 3], set(), 2) is None


def test_bounded_adjustment_rewards_joint_genres_without_overriding_arbitrary_model_gaps():
    raw = torch.tensor([[10., 9.9, 9.9] + [9.] * 97 + [-100.]])
    genres = {"Adventure": torch.tensor([True, True, True] + [False] * 97 + [True]),
              "Fantasy": torch.tensor([False, False, True] + [False] * 97 + [True])}
    scores = bounded_align_scores(raw, genres, explicit_genre_preferences("They enjoy adventure and fantasy."))
    assert scores[0, 2] > scores[0, 0]
    assert scores[0, -1] < scores[0, 0]
    assert torch.all((scores - raw).abs() <= 1.00001)
    assert raw[0, 0] == 10


def test_bounded_adjustment_is_per_profile_and_preserves_exclusions():
    raw = torch.tensor([[float("-inf"), 4., 3.], [float("-inf"), 40., 10.]])
    genres = {"Comedy": torch.tensor([True, False, True])}
    prefs = explicit_genre_preferences("They enjoy comedy.")
    result = bounded_align_scores(raw, genres, prefs)
    assert torch.equal(result, torch.cat([bounded_align_scores(row[None], genres, prefs) for row in raw]))
    assert torch.isneginf(result[:, 0]).all()


@pytest.mark.parametrize("raw", [torch.tensor([[float("-inf"), float("-inf")]]),
                                  torch.tensor([[float("-inf"), 4.]])])
def test_bounded_adjustment_handles_empty_or_single_candidate(raw):
    genres = {"Comedy": torch.tensor([True, True])}
    assert torch.equal(bounded_align_scores(raw, genres, explicit_genre_preferences("They enjoy comedy.")), raw)
