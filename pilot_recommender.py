"""Inference for the promoted full-corpus TEARS and GERS models."""

from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from tears_training.artifacts import sha256_file
from tears_training.config import load_config
from tears_training.models import EditableBase, HybridVAE
from tears_training.train import make_model


PROJECT_ROOT = Path(__file__).resolve().parent
ML32M_LINKS = PROJECT_ROOT / "ml-32m" / "links.csv"
MATRIX_DIR = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/"
    "datasets/catalog_selection/support_20/matrix"
)
PILOT_MATRIX_DIR = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/"
    "datasets/pilot/support_20/matrix"
)
RECVAE_CHECKPOINT = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/checkpoints/full/"
    "recvae/seed-2022/e7a7e560dc42/best.pt"
)
TEARS_RUN_DIR = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/checkpoints/full/"
    "tears_base/seed-2022/10d017146b22"
)
GERS_RUN_DIR = Path(
    "/network/scratch/a/adls/FullTrainingTEARS/checkpoints/full/"
    "gers_recvae/seed-2022/d6335f29527f"
)

EXPECTED_MATRIX_FINGERPRINT = (
    "27596ef71a4ac0f46e81ca97a40c4a9475bb1c2cc62c9db13819fe3cbeb714ce"
)
EXPECTED_PILOT_MATRIX_FINGERPRINT = (
    "a2cd89f043c8729ac84ebefbd4d38b8f98655d2bc77c598a19c7f51eeb8fd3b4"
)
EXPECTED_CATALOG_SHA256 = (
    "ac8b369749f8305bd712ec011150ff57b21459fe48be052f515ce5e291e00961"
)
EXPECTED_LINKS_SHA256 = (
    "ef17da7710be76f7d510d5768d1b61826e3af4bf57812b9ca377e4c912123b22"
)
EXPECTED_CHECKPOINT_SHA256 = {
    "recvae": "dcddf9fb6198841b0e30668007f2505a526265f669448a244acb16ac0f1fdc8a",
    "tears_base": "08f2609c6dae073a1d6c2a8ba14a3121c5ff4ec807201c7f97c7d4aa3c598e2b",
    "gers_recvae": "f4755f87b133b95ee6e9919c498011efd2096ee24da3e13895f48e3d30669dea",
}
EXPECTED_RUN_FINGERPRINTS = {
    "tears_base": "969f4b6ddb710f372739a6203724716e3f2a5e77ec9c3cd62b9e5058158ac3bc",
    "gers_recvae": "d2c828c1be5af3cf37716e4efb9bf4c4eeaab7d1f91bf2ac34e0c6aa52b8171f",
}

GENRE_ALIASES = {
    "family": "Children",
    "history": "Drama",
    "music": "Musical",
    "science fiction": "Sci-Fi",
    "sci fi": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "tv movie": "Drama",
}


def _checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError(f"Invalid pilot checkpoint: {path}")
    return payload


class PilotHybridRecommender:
    """Load the promoted full 200,948-profile TEARS and GERS runs."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        verify_hashes: bool = True,
    ) -> None:
        self.device = torch.device(
            device
            if device is not None
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for pilot inference but is unavailable")

        self._lock = RLock()
        self.config = load_config(PROJECT_ROOT / "configs" / "ml32m.toml")
        self._validate_artifacts(verify_hashes=verify_hashes)

        self.catalog = pd.read_csv(MATRIX_DIR / "catalog.csv").sort_values(
            "modelItemId"
        )
        links = pd.read_csv(
            ML32M_LINKS,
            dtype={"movieId": "int64", "imdbId": "string", "tmdbId": "Int64"},
        )
        self.catalog = self.catalog.merge(
            links,
            on="movieId",
            how="left",
            validate="one_to_one",
        )
        if self.catalog.imdbId.isna().any():
            raise RuntimeError("Pilot catalog contains IDs absent from ML-32M links.csv")
        expected_indices = np.arange(len(self.catalog), dtype=np.int64)
        actual_indices = self.catalog.modelItemId.to_numpy(np.int64)
        if not np.array_equal(actual_indices, expected_indices):
            raise RuntimeError("Pilot catalog modelItemId values are not contiguous")
        if not self.catalog.movieId.is_unique:
            raise RuntimeError("Pilot catalog MovieLens IDs are not unique")

        release_years = pd.to_numeric(
            self.catalog.title.str.extract(r"\((\d{4})\)\s*$", expand=False),
            errors="coerce",
        ).fillna(0)
        self.release_years = torch.tensor(
            release_years.to_numpy(dtype=np.int64),
            dtype=torch.int64,
            device=self.device,
        )

        self.movie_to_item = {
            int(row.movieId): int(row.modelItemId)
            for row in self.catalog.itertuples(index=False)
        }
        self.item_count = len(self.catalog)
        self.genre_names = sorted(
            {
                genre
                for value in self.catalog.genres.fillna("Unknown")
                for genre in str(value).split("|")
            }
        )
        self.genre_to_index = {
            genre: index for index, genre in enumerate(self.genre_names)
        }

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.backbone)
        self.tears = self._load_tears_model()
        self.gers = self._load_hybrid_model("gers_recvae", GERS_RUN_DIR)

    def _validate_artifacts(self, verify_hashes: bool) -> None:
        required = [
            MATRIX_DIR / "manifest.json",
            MATRIX_DIR / "catalog.csv",
            MATRIX_DIR / "users.csv",
            PILOT_MATRIX_DIR / "manifest.json",
            ML32M_LINKS,
            RECVAE_CHECKPOINT,
            TEARS_RUN_DIR / "manifest.json",
            TEARS_RUN_DIR / "best.pt",
            GERS_RUN_DIR / "manifest.json",
            GERS_RUN_DIR / "best.pt",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("Missing serving artifacts:\n" + "\n".join(missing))

        matrix_manifest = json.loads(
            (MATRIX_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        if matrix_manifest.get("fingerprint") != EXPECTED_MATRIX_FINGERPRINT:
            raise RuntimeError("Full matrix fingerprint differs from the promoted run")
        if matrix_manifest.get("items") != 22_343:
            raise RuntimeError("Full matrix does not contain the expected 22,343 items")
        if matrix_manifest.get("users") != 200_948:
            raise RuntimeError("Full matrix does not contain the expected 200,948 users")
        if matrix_manifest.get("catalog_sha256") != EXPECTED_CATALOG_SHA256:
            raise RuntimeError("Full matrix catalog hash differs from the promoted run")

        split_counts = (
            pd.read_csv(MATRIX_DIR / "users.csv", usecols=["split"])["split"]
            .value_counts()
            .to_dict()
        )
        if split_counts != {"train": 180_948, "validation": 10_000, "test": 10_000}:
            raise RuntimeError(f"Full matrix split counts differ: {split_counts}")

        pilot_matrix_manifest = json.loads(
            (PILOT_MATRIX_DIR / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            pilot_matrix_manifest.get("fingerprint")
            != EXPECTED_PILOT_MATRIX_FINGERPRINT
            or pilot_matrix_manifest.get("catalog_sha256")
            != EXPECTED_CATALOG_SHA256
        ):
            raise RuntimeError("Retained GERS pilot matrix differs from promotion")

        for name, run_dir, matrix_dir, recvae_checkpoint in (
            ("tears_base", TEARS_RUN_DIR, MATRIX_DIR, None),
            ("gers_recvae", GERS_RUN_DIR, MATRIX_DIR, RECVAE_CHECKPOINT),
        ):
            run_manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            if run_manifest.get("fingerprint") != EXPECTED_RUN_FINGERPRINTS[name]:
                raise RuntimeError(f"{name} run fingerprint differs from promotion")
            arguments = run_manifest.get("arguments", {})
            if arguments.get("matrix_dir") != str(matrix_dir):
                raise RuntimeError(f"{name} references a different matrix")
            if arguments.get("recvae_checkpoint") != (
                str(recvae_checkpoint) if recvae_checkpoint is not None else None
            ):
                raise RuntimeError(f"{name} references a different RecVAE checkpoint")
            if arguments.get("epochs") != 200:
                raise RuntimeError(f"{name} is not the frozen 200-epoch run")

        if verify_hashes:
            if sha256_file(ML32M_LINKS) != EXPECTED_LINKS_SHA256:
                raise RuntimeError("ML-32M links.csv SHA-256 mismatch")
            paths = {
                "recvae": RECVAE_CHECKPOINT,
                "tears_base": TEARS_RUN_DIR / "best.pt",
                "gers_recvae": GERS_RUN_DIR / "best.pt",
            }
            for name, path in paths.items():
                actual = sha256_file(path)
                if actual != EXPECTED_CHECKPOINT_SHA256[name]:
                    raise RuntimeError(f"{name} checkpoint SHA-256 mismatch")

    def _load_tears_model(self) -> EditableBase:
        model = make_model(
            "tears_base",
            self.item_count,
            len(self.genre_names),
            self.config,
            None,
            dropout=0.1,
            gamma=0.0035,
        )
        if not isinstance(model, EditableBase):
            raise AssertionError("Expected TEARS Base editable model")
        payload = _checkpoint_payload(TEARS_RUN_DIR / "best.pt")
        if payload.get("fingerprint") != EXPECTED_RUN_FINGERPRINTS["tears_base"]:
            raise RuntimeError("tears_base checkpoint fingerprint mismatch")
        model.load_state_dict(payload["model"], strict=True)
        return model.to(self.device).eval()

    def _load_hybrid_model(self, name: str, run_dir: Path) -> HybridVAE:
        model = make_model(
            name,
            self.item_count,
            len(self.genre_names),
            self.config,
            RECVAE_CHECKPOINT,
            dropout=0.1,
            gamma=0.0035,
        )
        if not isinstance(model, HybridVAE):
            raise AssertionError(f"Expected a hybrid model for {name}")
        payload = _checkpoint_payload(run_dir / "best.pt")
        if payload.get("fingerprint") != EXPECTED_RUN_FINGERPRINTS[name]:
            raise RuntimeError(f"{name} checkpoint fingerprint mismatch")
        model.load_state_dict(payload["model"], strict=True)
        return model.to(self.device).eval()

    def _ratings(self, movie_ids: Iterable[int]) -> torch.Tensor:
        ratings = torch.zeros((1, self.item_count), dtype=torch.float32)
        for raw_movie_id in movie_ids:
            try:
                item = self.movie_to_item.get(int(raw_movie_id))
            except (TypeError, ValueError, OverflowError):
                item = None
            if item is not None:
                ratings[0, item] = 5.0
        return ratings.to(self.device)

    def validate_movie_ids(
        self,
        movie_ids: Iterable[int],
        field_name: str,
    ) -> list[int]:
        normalized: list[int] = []
        invalid: list[Any] = []
        for raw_movie_id in movie_ids:
            try:
                movie_id = int(raw_movie_id)
            except (TypeError, ValueError, OverflowError):
                invalid.append(raw_movie_id)
                continue
            if movie_id not in self.movie_to_item:
                invalid.append(raw_movie_id)
                continue
            normalized.append(movie_id)
        if invalid:
            raise ValueError(
                f"{field_name} contains IDs outside the frozen ML-32M pilot catalog: "
                + ", ".join(map(str, invalid[:10]))
            )
        return normalized

    def _genre_vector(self, genres: Iterable[str]) -> torch.Tensor:
        vector = torch.zeros((1, len(self.genre_names)), dtype=torch.float32)
        for raw_genre in genres:
            value = str(raw_genre).strip()
            canonical = GENRE_ALIASES.get(value.casefold(), value)
            index = self.genre_to_index.get(canonical)
            if index is not None:
                vector[0, index] += 1.0
        total = vector.sum()
        if total <= 0:
            raise ValueError("None of the supplied genres exist in the pilot catalog")
        return (vector / total).to(self.device)

    def _ranked_items(
        self,
        logits: torch.Tensor,
        excluded_movie_ids: Iterable[int],
        top_k: int,
        min_release_year: int | None = None,
    ) -> list[dict[str, Any]]:
        scores = logits[0].clone()
        release_years = getattr(self, "release_years", None)
        if release_years is None:
            parsed_years = pd.to_numeric(
                self.catalog.title.str.extract(
                    r"\((\d{4})\)\s*$", expand=False
                ),
                errors="coerce",
            ).fillna(0)
            release_years = torch.tensor(
                parsed_years.to_numpy(dtype=np.int64),
                dtype=torch.int64,
                device=scores.device,
            )
        else:
            release_years = release_years.to(scores.device)
        for raw_movie_id in excluded_movie_ids:
            try:
                item = self.movie_to_item.get(int(raw_movie_id))
            except (TypeError, ValueError, OverflowError):
                item = None
            if item is not None:
                scores[item] = -torch.inf

        if min_release_year is not None:
            scores[release_years < int(min_release_year)] = -torch.inf

        width = min(max(int(top_k), 1), self.item_count)
        values, indices = torch.topk(scores, width)
        items: list[dict[str, Any]] = []
        for rank, (score, item_id) in enumerate(
            zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()),
            start=1,
        ):
            if not np.isfinite(score):
                continue
            row = self.catalog.iloc[int(item_id)]
            tmdb_id = None if pd.isna(row.tmdbId) else int(row.tmdbId)
            items.append(
                {
                    "movie_id": int(row.movieId),
                    "model_item_id": int(row.modelItemId),
                    "imdb_id": str(row.imdbId),
                    "tmdb_id": tmdb_id,
                    "title": str(row.title),
                    "release_year": int(release_years[int(item_id)].item()),
                    "genres": str(row.genres).split("|"),
                    "score": float(score),
                    "rank": rank,
                    "rank_label": f"#{rank}",
                }
            )
        return items

    def recommend_tears(
        self,
        summary: str,
        liked_movie_ids: Iterable[int],
        excluded_movie_ids: Iterable[int],
        alpha: float = 0.5,
        top_k: int = 12,
        min_release_year: int | None = None,
    ) -> list[dict[str, Any]]:
        if not summary.strip():
            raise ValueError("summary must not be empty")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        liked = self.validate_movie_ids(liked_movie_ids, "liked_movie_ids")
        excluded = self.validate_movie_ids(
            excluded_movie_ids, "excluded_movie_ids"
        )
        encoded = self.tokenizer(
            [summary.strip()],
            padding="max_length",
            truncation=True,
            max_length=self.config.model.max_text_tokens,
            return_tensors="pt",
        )
        with self._lock, torch.inference_mode():
            logits, _ = self.tears(
                encoded.input_ids.to(self.device),
                encoded.attention_mask.to(self.device),
            )
        return self._ranked_items(
            logits,
            [*liked, *excluded],
            top_k,
            min_release_year,
        )

    def recommend_gers(
        self,
        genres: Iterable[str],
        liked_movie_ids: Iterable[int],
        excluded_movie_ids: Iterable[int],
        alpha: float = 0.5,
        top_k: int = 12,
        min_release_year: int | None = None,
    ) -> list[dict[str, Any]]:
        liked = self.validate_movie_ids(liked_movie_ids, "liked_movie_ids")
        excluded = self.validate_movie_ids(
            excluded_movie_ids, "excluded_movie_ids"
        )
        ratings = self._ratings(liked)
        genre_vector = self._genre_vector(genres)
        with self._lock, torch.inference_mode():
            outputs = self.gers(ratings, genre_vector, alpha=alpha)
            logits = outputs["merged_logits"]
            assert isinstance(logits, torch.Tensor)
        return self._ranked_items(
            logits,
            [*liked, *excluded],
            top_k,
            min_release_year,
        )

    def status(self) -> dict[str, Any]:
        return {
            "status": "running",
            "deployment": "full-tears-and-gers-200948",
            "device": str(self.device),
            "users": 200_948,
            "splits": {"train": 180_948, "validation": 10_000, "test": 10_000},
            "items": self.item_count,
            "catalog_support": 20,
            "matrix_fingerprint": EXPECTED_MATRIX_FINGERPRINT,
            "catalog": {
                "path": str(MATRIX_DIR / "catalog.csv"),
                "sha256": EXPECTED_CATALOG_SHA256,
                "links_path": str(ML32M_LINKS),
                "links_sha256": EXPECTED_LINKS_SHA256,
            },
            "models": {
                "recvae": {
                    "path": str(RECVAE_CHECKPOINT),
                    "sha256": EXPECTED_CHECKPOINT_SHA256["recvae"],
                },
                "tears": {
                    "model": "tears_base",
                    "seed": 2022,
                    "epochs": 200,
                    "path": str(TEARS_RUN_DIR / "best.pt"),
                    "sha256": EXPECTED_CHECKPOINT_SHA256["tears_base"],
                    "run_fingerprint": EXPECTED_RUN_FINGERPRINTS["tears_base"],
                },
                "gers": {
                    "model": "gers_recvae",
                    "scope": "full-cohort-180948-train",
                    "seed": 2022,
                    "epochs": 200,
                    "best_epoch": 22,
                    "path": str(GERS_RUN_DIR / "best.pt"),
                    "sha256": EXPECTED_CHECKPOINT_SHA256["gers_recvae"],
                    "run_fingerprint": EXPECTED_RUN_FINGERPRINTS["gers_recvae"],
                },
            },
        }
