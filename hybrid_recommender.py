# ---------------------------------------------------------
#   hybrid_recommender.py — NOVA FIXED VERSION (2025-11)
# ---------------------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np


def clean_title(t):
    if not t:
        return ""
    t = t.lower().split("(")[0].strip()
    t = t.replace(", the", "").replace(", a", "").replace(", an", "")
    return t.strip()


class HybridRecommender:

    def __init__(self, summary_encoder, movie_embeddings,
                 movie_titles, movie_genres, device="cuda"):

        self.device = device if torch.cuda.is_available() else "cpu"
        self.encoder = summary_encoder
        self.movie_titles = movie_titles
        self.movie_genres = movie_genres

        # normalize titles
        self.normalized_title_map = {
            clean_title(t): idx for idx, t in enumerate(movie_titles)
        }

        # embeddings
        if isinstance(movie_embeddings, np.ndarray):
            movie_embeddings = torch.tensor(movie_embeddings, dtype=torch.float32)

        movie_embeddings = torch.nan_to_num(movie_embeddings)
        movie_embeddings = F.normalize(movie_embeddings, dim=-1)

        self.movie_embeddings = movie_embeddings.to(self.device)

        self.projection = None

    # ---------------------------------------------------------
    # Main Recommend
    # ---------------------------------------------------------
    def recommend(self,
                  summary_text,
                  context_text="",
                  liked_titles=None,
                  disliked_genres=None,
                  top_k=12,
                  alpha=1.0):

        liked_titles = liked_titles or []
        disliked_genres = disliked_genres or []

        # encode
        s_vec = self.encoder.encode_summary(summary_text)
        c_vec = self.encoder.encode_context(context_text)
        fused = self.encoder.fuse_summary_and_context(s_vec, c_vec, alpha=alpha)

        fused = fused.float().to(self.device)

        # match dim
        if fused.shape[1] != self.movie_embeddings.shape[1]:
            if self.projection is None:
                print(f"⚙️ Creating projection layer {fused.shape[1]} → {self.movie_embeddings.shape[1]}")
                self.projection = torch.nn.Linear(
                    fused.shape[1],
                    self.movie_embeddings.shape[1]
                ).to(self.device)

            fused = self.projection(fused)

        fused = torch.nan_to_num(fused)
        fused = F.normalize(fused, dim=-1)

        # similarity
        sims = torch.matmul(self.movie_embeddings, fused.squeeze(0))
        scores = sims.tolist()

        # disliked filter (NOVA FIX)
        def is_disliked(i):
            genres = [g.strip().lower()
                      for g in str(self.movie_genres[i]).split(",")]
            return any(d.lower() in genres for d in disliked_genres)

        items = []
        for i, score in enumerate(scores):
            if not is_disliked(i):
                items.append({
                    "title": self.movie_titles[i],
                    "score": float(score)
                })

        # liked boost (NOVA FIX)
        for t in liked_titles:
            key = clean_title(t)
            if key in self.normalized_title_map:
                idx = self.normalized_title_map[key]
                for it in items:
                    if it["title"] == self.movie_titles[idx]:
                        it["score"] += 0.15

        # sort
        items = sorted(items, key=lambda x: x["score"], reverse=True)

        # rank
        for idx, it in enumerate(items):
            it["rank"] = idx + 1
            it["rank_label"] = f"Rank #{idx+1}"

        return items[:top_k]
