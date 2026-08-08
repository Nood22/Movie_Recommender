from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]
from typing import Any


MODEL_NAMES = (
    "recvae",
    "tears_base",
    "gers_base",
    "tears_recvae",
    "gers_recvae",
)


def default_scratch() -> Path:
    value = os.environ.get("SCRATCH")
    if not value:
        raise RuntimeError("SCRATCH must point to the user's cluster scratch directory")
    return Path(value).expanduser().resolve()


@dataclass(frozen=True)
class DataConfig:
    split_seed: int = 2024
    train_users: int = 180_948
    validation_users: int = 10_000
    test_users: int = 10_000
    observed_fraction: float = 0.70
    positive_rating: float = 4.0
    item_support_candidates: tuple[int, ...] = (20, 50, 200)
    smoke_users: int = 10_000
    smoke_items: int = 4_000
    pilot_train_users: int = 9_000
    pilot_validation_users: int = 500
    pilot_test_users: int = 500
    activity_bands: tuple[int, ...] = (20, 50, 100, 200, 500)


@dataclass(frozen=True)
class SummaryConfig:
    model: str = "gpt-5-mini-2025-08-07"
    prompt_version: str = "tears-emiliano-strict-v2"
    max_history_items: int = 50
    min_words: int = 120
    max_words: int = 260
    max_output_tokens: int = 450
    retry_limit: int = 2
    spend_cap_usd: float = 250.0
    reserve_fraction: float = 0.15
    max_batch_requests: int = 50_000
    max_batch_bytes: int = 200_000_000


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "google-t5/t5-base"
    latent_dim: int = 400
    lora_rank: int = 64
    lora_alpha: int = 16
    train_alpha: float = 0.5
    kl_anneal_cap: float = 0.5
    max_text_tokens: int = 512


@dataclass(frozen=True)
class TrackingConfig:
    entity: str = "niita-mila"
    project: str = "tears-ml32m"
    mode: str = "online"


@dataclass(frozen=True)
class ExperimentConfig:
    scratch_root: Path
    raw_data: Path
    output_root: Path
    paid_backup_root: Path
    data: DataConfig = field(default_factory=DataConfig)
    summaries: SummaryConfig = field(default_factory=SummaryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    seeds: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024)

    def __post_init__(self) -> None:
        if not 0 < self.data.observed_fraction < 1:
            raise ValueError("data.observed_fraction must be strictly between 0 and 1")
        if self.data.positive_rating <= 0:
            raise ValueError("data.positive_rating must be positive")
        if sorted(self.data.activity_bands) != list(self.data.activity_bands):
            raise ValueError("data.activity_bands must be sorted")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not 0 <= self.summaries.reserve_fraction < 1:
            raise ValueError("summaries.reserve_fraction must be in [0, 1)")
        if self.summaries.spend_cap_usd <= 0:
            raise ValueError("summaries.spend_cap_usd must be positive")
        if self.summaries.min_words > self.summaries.max_words:
            raise ValueError("summary word-count bounds are reversed")
        if not 0 <= self.model.train_alpha <= 1:
            raise ValueError("model.train_alpha must be in [0, 1]")

    @classmethod
    def defaults(cls) -> "ExperimentConfig":
        scratch = default_scratch()
        return cls(
            scratch_root=scratch,
            raw_data=scratch / "datasets" / "ml-32m",
            output_root=scratch / "FullTrainingTEARS",
            paid_backup_root=Path.home() / "FullTrainingTEARS_paid_artifacts",
        )

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))


def _merge_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    values = asdict(instance)
    values.update(updates)
    for key, current in asdict(instance).items():
        if isinstance(current, tuple) and key in values:
            values[key] = tuple(values[key])
    return type(instance)(**values)


def load_config(path: Path | None = None) -> ExperimentConfig:
    config = ExperimentConfig.defaults()
    if path is None:
        return config
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    allowed = {"paths", "data", "summaries", "model", "tracking", "seeds"}
    unexpected = set(raw) - allowed
    if unexpected:
        raise ValueError(f"Unknown configuration sections: {sorted(unexpected)}")
    paths = raw.get("paths", {})
    scratch = Path(paths.get("scratch_root", config.scratch_root)).expanduser().resolve()
    return ExperimentConfig(
        scratch_root=scratch,
        raw_data=Path(paths.get("raw_data", scratch / "datasets" / "ml-32m")),
        output_root=Path(paths.get("output_root", scratch / "FullTrainingTEARS")),
        paid_backup_root=Path(
            paths.get("paid_backup_root", config.paid_backup_root)
        ).expanduser(),
        data=_merge_dataclass(config.data, raw.get("data", {})),
        summaries=_merge_dataclass(config.summaries, raw.get("summaries", {})),
        model=_merge_dataclass(config.model, raw.get("model", {})),
        tracking=_merge_dataclass(config.tracking, raw.get("tracking", {})),
        seeds=tuple(raw.get("seeds", config.seeds)),
    )
