"""Checkpoint-backed inference adapter for the Code4Neda TEARS model."""

from __future__ import annotations

from argparse import Namespace
import gc
import math
from pathlib import Path
import pickle
import sys
from threading import Lock
from typing import Any, Iterable

import torch


class TEARSInferenceAdapter:
    """Load the ML-1M OTRecVAE checkpoint and expose JSON-ready inference."""

    NUM_MOVIES = 2745
    LATENT_DIM = 400
    BACKBONE = "google-t5/t5-base"

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.project_root = Path(__file__).resolve().parent
        self.tears_project_root = self.project_root / "TEARS_Project"
        self.code4neda_root = self.tears_project_root / "Code4Neda"

        self.show_to_index_path = (
            self.code4neda_root
            / "data_preprocessed"
            / "ml-1m"
            / "show2id.pkl"
        )
        self.movies_path = (
            self.code4neda_root / "data" / "ml-1m" / "movies.dat"
        )
        self.recvae_checkpoint_path = (
            self.tears_project_root
            / "saved_model"
            / "ml-1m"
            / (
                "t5_classification_fixed_data_ml-1m_embedding_module_"
                "RecVAE_2024-09-27_12-37-38_2024.csv.pt"
            )
        )
        self.final_checkpoint_path = (
            self.tears_project_root
            / "saved_model"
            / "ml-1m"
            / (
                "ot_train_vae_ml-1m_embedding_module_"
                "OTRecVAE_2024-09-24_13-37-29_2022.csv.pt"
            )
        )

        self._validate_artifacts()
        self.device = self._resolve_device(device)
        self._inference_lock = Lock()

        self.show_to_index = self._load_show_to_index()
        self.index_to_movie_id = {
            model_index: movie_id
            for movie_id, model_index in self.show_to_index.items()
        }
        self.movie_metadata = self._load_movie_metadata()
        self._validate_catalog()

        self.args = Namespace(
            data_name="ml-1m",
            embedding_module="OTRecVAE",
            bert=False,
            dropout=0.1,
            gamma=0.0035,
            epsilon=1.0,
            concat=False,
            lora_r=64,
            lora_alpha=16,
        )
        self.tokenizer, self.model = self._load_model()
        self._base_model = self.model.get_base_model()

    def _validate_artifacts(self) -> None:
        artifacts = {
            "MovieLens model-index mapping": self.show_to_index_path,
            "MovieLens metadata": self.movies_path,
            "RecVAE checkpoint": self.recvae_checkpoint_path,
            "final OTRecVAE checkpoint": self.final_checkpoint_path,
        }
        missing = [
            f"{label}: {path}"
            for label, path in artifacts.items()
            if not path.is_file()
        ]
        if missing:
            raise RuntimeError(
                "TEARS startup failed because required artifacts are missing:\n"
                + "\n".join(missing)
            )

    @staticmethod
    def _resolve_device(
        requested: str | torch.device | None,
    ) -> torch.device:
        if requested is None:
            return torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )

        try:
            device = torch.device(requested)
        except (TypeError, RuntimeError) as error:
            raise ValueError(f"Invalid TEARS device {requested!r}") from error

        if device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "TEARS supports only CPU and CUDA devices; "
                f"received {device.type!r}"
            )
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device {device} was requested, but CUDA is unavailable"
                )
            index = 0 if device.index is None else device.index
            if index < 0 or index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device index {index} is unavailable; "
                    f"detected {torch.cuda.device_count()} CUDA device(s)"
                )
            return torch.device("cuda", index)
        return torch.device("cpu")

    def _load_show_to_index(self) -> dict[int, int]:
        try:
            with self.show_to_index_path.open("rb") as file:
                raw_mapping = pickle.load(file)
        except Exception as error:
            raise RuntimeError(
                "Failed to load MovieLens model-index mapping from "
                f"{self.show_to_index_path}: {error}"
            ) from error

        if not isinstance(raw_mapping, dict):
            raise RuntimeError(
                f"{self.show_to_index_path} must contain a dictionary"
            )

        mapping: dict[int, int] = {}
        for raw_movie_id, raw_index in raw_mapping.items():
            if (
                isinstance(raw_movie_id, bool)
                or not isinstance(raw_movie_id, int)
                or isinstance(raw_index, bool)
                or not isinstance(raw_index, int)
            ):
                raise RuntimeError(
                    "show2id.pkl must map integer MovieLens IDs to integer "
                    "model indices"
                )
            mapping[raw_movie_id] = raw_index

        if len(mapping) != self.NUM_MOVIES:
            raise RuntimeError(
                f"show2id.pkl contains {len(mapping)} entries; "
                f"the checkpoint requires exactly {self.NUM_MOVIES}"
            )
        expected_indices = set(range(self.NUM_MOVIES))
        actual_indices = set(mapping.values())
        if actual_indices != expected_indices:
            missing = sorted(expected_indices - actual_indices)[:10]
            unexpected = sorted(actual_indices - expected_indices)[:10]
            raise RuntimeError(
                "show2id.pkl indices are not the contiguous checkpoint range "
                f"0..{self.NUM_MOVIES - 1}; missing={missing}, "
                f"unexpected={unexpected}"
            )
        return mapping

    def _load_movie_metadata(self) -> dict[int, dict[str, Any]]:
        metadata: dict[int, dict[str, Any]] = {}
        try:
            with self.movies_path.open("r", encoding="latin-1") as file:
                for line_number, line in enumerate(file, start=1):
                    parts = line.rstrip("\n").split("::")
                    if len(parts) != 3:
                        raise RuntimeError(
                            f"Malformed movies.dat line {line_number}: "
                            "expected movie_id::title::genres"
                        )
                    movie_id_text, title, genre_text = parts
                    movie_id = int(movie_id_text)
                    metadata[movie_id] = {
                        "title": title,
                        "genres": genre_text.split("|") if genre_text else [],
                    }
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(
                f"Failed to load MovieLens metadata from {self.movies_path}: "
                f"{error}"
            ) from error
        return metadata

    def _validate_catalog(self) -> None:
        missing_metadata = sorted(
            set(self.show_to_index) - set(self.movie_metadata)
        )
        if missing_metadata:
            raise RuntimeError(
                "MovieLens metadata is missing checkpoint movie IDs: "
                f"{missing_metadata[:20]}"
            )

    def _load_model(self) -> tuple[Any, torch.nn.Module]:
        code4neda_path = str(self.code4neda_root)
        tears_project_path = str(self.tears_project_root)
        for import_path in (tears_project_path, code4neda_path):
            if import_path not in sys.path:
                sys.path.insert(0, import_path)

        try:
            from model.MF import RecVAE, T5Vae, get_tokenizer
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as error:
            raise RuntimeError(
                "Failed to import Code4Neda model dependencies. Use the "
                "project environment containing torch, transformers, peft, "
                f"sentence-transformers, and POT: {error}"
            ) from error

        try:
            tokenizer = get_tokenizer(self.args)
        except Exception as error:
            raise RuntimeError(
                f"Failed to initialize the {self.BACKBONE} tokenizer: {error}"
            ) from error

        prior = RecVAE(
            [self.LATENT_DIM, self.NUM_MOVIES],
            dropout=self.args.dropout,
            gamma=self.args.gamma,
        )
        try:
            prior_state = torch.load(
                self.recvae_checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            prior.load_state_dict(prior_state, strict=True, assign=True)
            del prior_state
            gc.collect()
        except Exception as error:
            raise RuntimeError(
                "Failed strict loading of the RecVAE checkpoint "
                f"{self.recvae_checkpoint_path}: {error}"
            ) from error

        try:
            model = T5Vae.from_pretrained(
                self.BACKBONE,
                num_labels=self.NUM_MOVIES,
                classifier_dropout=self.args.dropout,
                prior=None,
                epsilon=self.args.epsilon,
                concat=self.args.concat,
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to reconstruct T5Vae from {self.BACKBONE}: {error}"
            ) from error

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.dropout,
            target_modules=["q", "v", "k"],
            modules_to_save=[
                "classification_head",
                "mlp",
                "RecVAE",
                "Encoder",
                "CompositePrior",
                "concat_mlp",
            ],
        )
        try:
            model = get_peft_model(model, lora_config)
            model.set_vae(prior)
            final_state = torch.load(
                self.final_checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            model.load_state_dict(final_state, strict=True, assign=True)
            del final_state
            gc.collect()
        except Exception as error:
            raise RuntimeError(
                "Failed strict loading of the final OTRecVAE/T5Vae "
                f"checkpoint {self.final_checkpoint_path}: {error}"
            ) from error

        decoder = model.get_base_model().vae.decoder
        if decoder.out_features != self.NUM_MOVIES:
            raise RuntimeError(
                f"Loaded decoder has {decoder.out_features} outputs; "
                f"expected {self.NUM_MOVIES}"
            )

        try:
            model.eval()
            model.to(self.device)
        except Exception as error:
            raise RuntimeError(
                f"Failed to move TEARS model to {self.device}: {error}"
            ) from error
        return tokenizer, model

    def _validate_request(
        self,
        summary: str,
        liked_movie_ids: Iterable[int],
        alpha: float,
        top_k: int,
    ) -> tuple[str, list[int], float, int]:
        if not isinstance(summary, str):
            raise ValueError("summary must be a string")
        summary = summary.strip()
        if not summary:
            raise ValueError("summary must contain non-whitespace text")
        if len(summary) > 4000:
            raise ValueError("summary must not exceed 4000 characters")

        if isinstance(liked_movie_ids, (str, bytes)) or not isinstance(
            liked_movie_ids, Iterable
        ):
            raise ValueError("liked_movie_ids must be an iterable of integers")
        unique_ids: list[int] = []
        seen: set[int] = set()
        for movie_id in liked_movie_ids:
            if isinstance(movie_id, bool) or not isinstance(movie_id, int):
                raise ValueError(
                    "liked_movie_ids must contain only integer MovieLens IDs"
                )
            if movie_id <= 0:
                raise ValueError("liked_movie_ids must contain positive IDs")
            if movie_id not in seen:
                seen.add(movie_id)
                unique_ids.append(movie_id)
        if len(unique_ids) > 500:
            raise ValueError("liked_movie_ids must contain at most 500 IDs")
        unknown_ids = [
            movie_id
            for movie_id in unique_ids
            if movie_id not in self.show_to_index
        ]
        if unknown_ids:
            raise ValueError(
                "liked_movie_ids contains IDs outside the checkpoint catalog: "
                f"{unknown_ids}"
            )

        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError("alpha must be a finite number from 0.0 to 1.0")
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite number from 0.0 to 1.0")

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        available = self.NUM_MOVIES - len(unique_ids)
        if top_k > available:
            raise ValueError(
                f"top_k={top_k} exceeds the {available} non-liked movies"
            )
        return summary, unique_ids, alpha, top_k

    def _tokenize(self, summary: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [summary],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _text_only_logits(
        self, tokenized: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        sentence_representation = self._base_model.llm_forward(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            return_dict=True,
        )
        distribution = self._base_model.mlp(sentence_representation)
        text_mu = distribution[:, : self.LATENT_DIM]
        return self._base_model.vae.decode(text_mu)

    def recommend(
        self,
        summary: str,
        liked_movie_ids: Iterable[int],
        alpha: float = 0.5,
        top_k: int = 12,
    ) -> list[dict[str, Any]]:
        """Return ranked, JSON-ready MovieLens recommendations."""

        summary, liked_ids, alpha, top_k = self._validate_request(
            summary, liked_movie_ids, alpha, top_k
        )
        liked_indices = [
            self.show_to_index[movie_id] for movie_id in liked_ids
        ]

        with self._inference_lock, torch.inference_mode():
            tokenized = self._tokenize(summary)
            if liked_indices:
                interactions = torch.zeros(
                    (1, self.NUM_MOVIES),
                    dtype=torch.float32,
                    device=self.device,
                )
                interactions[0, liked_indices] = 1.0
                logits = self.model(
                    data_tensor=interactions,
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    alpha=alpha,
                )[0]
            else:
                # RecVAE normalizes by the interaction-vector norm. Bypass it
                # for an empty history instead of evaluating 0 / 0.
                logits = self._text_only_logits(tokenized)

            if logits.shape != (1, self.NUM_MOVIES):
                raise RuntimeError(
                    f"TEARS returned logits with shape {tuple(logits.shape)}; "
                    f"expected (1, {self.NUM_MOVIES})"
                )
            if not torch.isfinite(logits).all():
                nonfinite = int((~torch.isfinite(logits)).sum().item())
                raise RuntimeError(
                    f"TEARS returned {nonfinite} non-finite logit(s)"
                )

            masked_logits = logits.clone()
            if liked_indices:
                masked_logits[0, liked_indices] = -torch.inf
            scores, indices = torch.topk(masked_logits[0], k=top_k)

        items: list[dict[str, Any]] = []
        for rank, (score, model_index) in enumerate(
            zip(scores.tolist(), indices.tolist()), start=1
        ):
            movie_id = self.index_to_movie_id[model_index]
            metadata = self.movie_metadata[movie_id]
            items.append(
                {
                    "movie_id": movie_id,
                    "title": metadata["title"],
                    "genres": list(metadata["genres"]),
                    "score": float(score),
                    "rank": rank,
                    "rank_label": f"#{rank}",
                }
            )
        return items
