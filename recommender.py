"""
recommender.py
Final intelligent version — uses T5 encoder to compute semantic similarity between user summary and movie titles.
"""

import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm


def get_text_embedding(model, tokenizer, text, device):
    """Encode text into a dense embedding using T5 encoder."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        hidden = model.encoder(inputs.input_ids)[0]
        emb = hidden.mean(dim=1)  # mean pooling
    return emb


def recommend_from_summary(model, tokenizer, item_title_dict, item_genre_dict, summary, top_k=10):
    """
    Recommend real movie titles by comparing T5-encoded embeddings of user summary vs movie titles.
    """
    try:
        model.eval()
        device = next(model.parameters()).device

        # 1️⃣ Embed user summary
        user_emb = get_text_embedding(model, tokenizer, summary, device)

        # 2️⃣ Embed all movie titles (you can cache this later)
        sims = []
        for mid, title in tqdm(item_title_dict.items(), desc="Scoring movies", leave=False):
            movie_emb = get_text_embedding(model, tokenizer, title, device)
            score = cosine_similarity(user_emb, movie_emb).item()
            sims.append((title, score))

        # 3️⃣ Rank by similarity
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        recs = [title for title, _ in sims[:top_k]]

        return recs

    except Exception as e:
        print(f"⚠️ Error in recommend_from_summary: {e}")
        return list(item_title_dict.values())[:top_k]


def recommend_from_genre(model, tokenizer, item_title_dict, item_genre_dict, genre, top_k=10):
    """Simple genre-based fallback recommender."""
    matches = [
        title for mid, title in item_title_dict.items()
        if genre.lower() in item_genre_dict.get(mid, "").lower()
    ]
    return matches[:top_k] if matches else list(item_title_dict.values())[:top_k]
