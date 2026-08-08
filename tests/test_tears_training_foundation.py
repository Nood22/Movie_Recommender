from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tears_training.artifacts import atomic_write_json, stable_hash
from tears_training.config import ExperimentConfig
from tears_training.data import _chronological_eval_rows, stratified_sample
from tears_training.models import GaussianLatent, HybridVAE, RecVAE, diagonal_wasserstein


def test_default_paths_are_derived_from_scratch(monkeypatch, tmp_path):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    config = ExperimentConfig.defaults()
    assert config.raw_data == tmp_path / "datasets" / "ml-32m"
    assert config.output_root == tmp_path / "FullTrainingTEARS"
    assert config.tracking.entity == "niita-mila"


def test_atomic_json_and_stable_hash(tmp_path):
    path = tmp_path / "nested" / "manifest.json"
    atomic_write_json(path, {"b": 2, "a": 1})
    assert path.read_text().startswith('{\n  "a"')
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_stratified_sample_is_deterministic_and_exact():
    users = pd.DataFrame(
        {
            "userId": np.arange(100),
            "interaction_count": [30] * 50 + [600] * 50,
            "activity_band": ["20-49"] * 50 + ["500+"] * 50,
        }
    )
    first = stratified_sample(users, 20, 2024)
    second = stratified_sample(users, 20, 2024)
    assert np.array_equal(first, second)
    assert len(set(first)) == 20
    selected = users[users.userId.isin(first)]
    assert selected.activity_band.value_counts().to_dict() == {"20-49": 10, "500+": 10}


def test_chronological_split_uses_early_observed_and_positive_late_targets():
    frame = pd.DataFrame(
        {
            "userId": [1] * 10,
            "movieId": np.arange(10),
            "rating": [1, 2, 3, 4, 5, 1, 2, 4, 3, 5],
            "timestamp": np.arange(10),
            "modelUserId": [0] * 10,
            "modelItemId": np.arange(10),
        }
    )
    observed, target = _chronological_eval_rows(frame, 0.7, 4.0)
    assert observed.movieId.tolist() == list(range(7))
    assert target.movieId.tolist() == [7, 9]


class FixedEditable(torch.nn.Module):
    def __init__(self, item_count: int):
        super().__init__()
        self.decoder = torch.nn.Linear(2, item_count, bias=False)

    def encode(self, value: torch.Tensor) -> GaussianLatent:
        return GaussianLatent(value, torch.full_like(value, -20.0))


def test_hybrid_uses_paper_alpha_direction():
    recvae = RecVAE(3, latent_dim=2)
    editable = FixedEditable(3)
    hybrid = HybridVAE(recvae, editable).eval()
    with torch.no_grad():
        hybrid.recvae.decoder.weight.copy_(torch.tensor([[1.0, 0], [0, 1.0], [1.0, 1.0]]))
        hybrid.recvae.mean.weight.zero_()
        hybrid.recvae.mean.bias.copy_(torch.tensor([2.0, 0.0]))
        hybrid.recvae.log_variance.weight.zero_()
        hybrid.recvae.log_variance.bias.fill_(-20)
    ratings = torch.ones(1, 3)
    text = torch.tensor([[0.0, 3.0]])
    text_only = hybrid(ratings, text, alpha=1.0)["merged_logits"]
    rec_only = hybrid(ratings, text, alpha=0.0)["merged_logits"]
    assert torch.allclose(text_only, hybrid.editable.decoder(text), atol=1e-3)
    assert not torch.allclose(text_only, rec_only)


def test_diagonal_wasserstein_is_zero_for_identical_latents():
    latent = GaussianLatent(torch.randn(4, 3), torch.randn(4, 3))
    assert torch.allclose(diagonal_wasserstein(latent, latent), torch.tensor(0.0))
