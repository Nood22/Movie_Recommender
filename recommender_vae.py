"""
recommender.py
Real inference functions for OTRecVAE / RecVAE models.
"""

import torch
from torch.nn.functional import cosine_similarity

def recommend_from_summary(model, tokenizer, item_title_dict, item_genre_dict, summary, top_k=10):
    """
    Generate movie recommendations using a T5-based text generation model (TEARS).
    This version uses model.generate() to create textual recommendations directly from summary input.
    """

    try:
        model.eval()
        device = next(model.parameters()).device

        # 1️⃣ Prepare the input prompt
        input_text = f"Recommend {top_k} movies for a user who described their preferences as: {summary}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

        # 2️⃣ Generate text output
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                early_stopping=True
            )

        # 3️⃣ Decode the generated text
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # 4️⃣ Basic cleaning and splitting
        recs = [r.strip() for r in generated_text.replace("\n", ",").split(",") if len(r.strip()) > 1]

        # 5️⃣ Take top_k items only
        return recs[:top_k]

    except Exception as e:
        raise RuntimeError(f"Error in T5 recommend_from_summary: {e}")

def recommend_from_genre(model, tokenizer, item_title_dict, item_genre_dict, genre, top_k=10):
    """
    Simple genre-based fallback recommender.
    """
    matches = [title for mid, title in item_title_dict.items()
               if genre.lower() in item_genre_dict.get(mid, "").lower()]

    if not matches:
        return list(item_title_dict.values())[:top_k]

    return matches[:top_k]
