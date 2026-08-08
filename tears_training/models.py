from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GaussianLatent:
    mean: torch.Tensor
    log_variance: torch.Tensor
    sampled: torch.Tensor | None = None

    def sample(self, training: bool = True) -> torch.Tensor:
        if not training:
            return self.mean
        if self.sampled is None:
            self.sampled = self.mean + torch.randn_like(self.mean) * torch.exp(
                0.5 * self.log_variance
            )
        return self.sampled


class LatentEncoder(Protocol):
    def encode(self, *args: torch.Tensor) -> GaussianLatent: ...


def _swish(value: torch.Tensor) -> torch.Tensor:
    return value * torch.sigmoid(value)


class RecVAEEncoder(nn.Module):
    """Residual encoder used by the original RecVAE architecture."""

    def __init__(self, item_count: int, latent_dim: int, hidden_dim: int = 600):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(item_count, hidden_dim),
                *[nn.Linear(hidden_dim, hidden_dim) for _ in range(4)],
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim, eps=0.1) for _ in range(5)])
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_variance = nn.Linear(hidden_dim, latent_dim)

    def forward(self, ratings: torch.Tensor, dropout: float) -> GaussianLatent:
        value = F.normalize(ratings, dim=-1)
        value = F.dropout(value, dropout, self.training)
        residuals: list[torch.Tensor] = []
        for layer, norm in zip(self.layers, self.norms):
            projected = layer(value)
            if residuals:
                projected = projected + torch.stack(residuals).sum(0)
            value = norm(_swish(projected))
            residuals.append(value)
        return GaussianLatent(self.mean(value), self.log_variance(value).clamp(-12, 12))


def _log_normal(value: torch.Tensor, latent: GaussianLatent) -> torch.Tensor:
    return -0.5 * (
        latent.log_variance
        + torch.log(torch.tensor(2.0 * torch.pi, device=value.device))
        + (value - latent.mean).square() / latent.log_variance.exp()
    )


class CompositePrior(nn.Module):
    """RecVAE's standard/old-posterior/broad Gaussian mixture prior."""

    def __init__(self, item_count: int, latent_dim: int, hidden_dim: int = 600):
        super().__init__()
        self.old_encoder = RecVAEEncoder(item_count, latent_dim, hidden_dim)
        self.old_encoder.requires_grad_(False)
        self.register_buffer("zero", torch.zeros(1, latent_dim))
        self.register_buffer("broad_log_variance", torch.full((1, latent_dim), 10.0))
        self.weights = (3 / 20, 3 / 4, 1 / 10)

    def forward(self, ratings: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        old = self.old_encoder(ratings, 0.0)
        standard = GaussianLatent(self.zero, self.zero)
        broad = GaussianLatent(self.zero, self.broad_log_variance)
        components = [standard, old, broad]
        weighted = [
            _log_normal(sample, latent) + torch.log(sample.new_tensor(weight))
            for latent, weight in zip(components, self.weights)
        ]
        return torch.logsumexp(torch.stack(weighted, dim=-1), dim=-1)


class RecVAE(nn.Module):
    """RecVAE backbone with its composite prior and editable shared decoder."""

    def __init__(
        self,
        item_count: int,
        latent_dim: int = 400,
        dropout: float = 0.1,
        gamma: float = 0.0035,
    ):
        super().__init__()
        self.item_count = item_count
        self.dropout = dropout
        self.gamma = gamma
        self.encoder = RecVAEEncoder(item_count, latent_dim)
        self.prior = CompositePrior(item_count, latent_dim)
        self.decoder = nn.Linear(latent_dim, item_count)

    @property
    def mean(self) -> nn.Linear:
        return self.encoder.mean

    @property
    def log_variance(self) -> nn.Linear:
        return self.encoder.log_variance

    def encode(self, ratings: torch.Tensor) -> GaussianLatent:
        return self.encoder(ratings, self.dropout)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, ratings: torch.Tensor) -> tuple[torch.Tensor, GaussianLatent]:
        latent = self.encode(ratings)
        return self.decode(latent.sample(self.training)), latent

    def loss(self, logits: torch.Tensor, ratings: torch.Tensor, latent: GaussianLatent) -> torch.Tensor:
        sample = latent.sample(self.training)
        likelihood = (F.log_softmax(logits, dim=-1) * ratings).sum(-1)
        kl = (_log_normal(sample, latent) - self.prior(ratings, sample)).sum(-1)
        weight = self.gamma * ratings.sum(-1)
        return -(likelihood - weight * kl).mean()

    @torch.no_grad()
    def update_prior(self) -> None:
        self.prior.old_encoder.load_state_dict(deepcopy(self.encoder.state_dict()))


class T5SummaryEncoder(nn.Module):
    def __init__(
        self,
        item_count: int,
        latent_dim: int = 400,
        backbone: str = "google-t5/t5-base",
        lora_rank: int = 64,
        lora_alpha: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import T5EncoderModel

        encoder = T5EncoderModel.from_pretrained(backbone)
        lora = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=dropout,
            target_modules=["q", "k", "v"],
        )
        self.encoder = get_peft_model(encoder, lora)
        dimension = encoder.config.d_model
        self.projection = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(dimension, latent_dim * 2)
        )
        self.decoder = nn.Linear(latent_dim, item_count)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> GaussianLatent:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        mean, log_variance = self.projection(pooled).chunk(2, dim=-1)
        return GaussianLatent(mean, log_variance.clamp(-12, 12))


class GenreEncoder(nn.Module):
    def __init__(self, genre_count: int, item_count: int, latent_dim: int = 400):
        super().__init__()
        self.projection = nn.Linear(genre_count, latent_dim * 2, bias=False)
        self.decoder = nn.Linear(latent_dim, item_count)

    def encode(self, genre_vector: torch.Tensor) -> GaussianLatent:
        mean, log_variance = self.projection(genre_vector).chunk(2, dim=-1)
        return GaussianLatent(mean, log_variance.clamp(-12, 12))


class EditableBase(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.editable = encoder

    def forward(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, GaussianLatent]:
        latent = self.editable.encode(*inputs)
        return self.editable.decoder(latent.sample(self.training)), latent


class HybridVAE(nn.Module):
    """Paper-convention hybrid: alpha=1 editable, alpha=0 RecVAE."""

    def __init__(self, recvae: RecVAE, editable: nn.Module):
        super().__init__()
        self.recvae = recvae
        self.editable = editable
        self.editable.decoder = self.recvae.decoder
        self.recvae.encoder.requires_grad_(False)
        self.recvae.prior.requires_grad_(False)

    def forward(
        self, ratings: torch.Tensor, *editable_inputs: torch.Tensor, alpha: float = 0.5
    ) -> dict[str, torch.Tensor | GaussianLatent]:
        rec = self.recvae.encode(ratings)
        edit = self.editable.encode(*editable_inputs)
        rec_z = rec.sample(self.training)
        edit_z = edit.sample(self.training)
        merged = alpha * edit_z + (1.0 - alpha) * rec_z
        return {
            "merged_logits": self.recvae.decode(merged),
            "rec_logits": self.recvae.decode(rec_z),
            "editable_logits": self.recvae.decode(edit_z),
            "rec_latent": rec,
            "editable_latent": edit,
            "rec_prior_kl": (
                _log_normal(rec_z, rec) - self.recvae.prior(ratings, rec_z)
            ).sum(-1).mean(),
        }


def multinomial_reconstruction(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(F.log_softmax(logits, dim=-1) * targets).sum(-1).mean()


def gaussian_kl(latent: GaussianLatent) -> torch.Tensor:
    return -0.5 * (1 + latent.log_variance - latent.mean.square() - latent.log_variance.exp()).sum(-1).mean()


def diagonal_wasserstein(first: GaussianLatent, second: GaussianLatent) -> torch.Tensor:
    mean = (first.mean - second.mean).square().sum(-1)
    std = (torch.exp(0.5 * first.log_variance) - torch.exp(0.5 * second.log_variance)).square().sum(-1)
    return (mean + std).mean()


def hybrid_loss(
    outputs: dict[str, torch.Tensor | GaussianLatent],
    targets: torch.Tensor,
    ot_weight: float,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    merged = outputs["merged_logits"]
    rec_logits = outputs["rec_logits"]
    editable_logits = outputs["editable_logits"]
    rec = outputs["rec_latent"]
    editable = outputs["editable_latent"]
    assert isinstance(merged, torch.Tensor)
    assert isinstance(rec_logits, torch.Tensor)
    assert isinstance(editable_logits, torch.Tensor)
    assert isinstance(rec, GaussianLatent)
    assert isinstance(editable, GaussianLatent)
    reconstruction = sum(
        multinomial_reconstruction(logits, targets)
        for logits in (merged, rec_logits, editable_logits)
    ) / 3.0
    ot = diagonal_wasserstein(editable, rec)
    editable_kl = gaussian_kl(editable)
    rec_kl = outputs["rec_prior_kl"]
    assert isinstance(rec_kl, torch.Tensor)
    kl = 0.5 * (editable_kl + rec_kl)
    total = reconstruction + ot_weight * ot + kl_weight * kl
    return total, {
        "reconstruction": float(reconstruction.detach()),
        "ot": float(ot.detach()),
        "kl": float(kl.detach()),
    }
