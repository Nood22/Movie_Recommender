# ---------------------------------------------------------
#   summary_encoder.py — FINAL VERSION (2025-11)
# ---------------------------------------------------------

import torch
from transformers import T5Tokenizer, T5EncoderModel
import torch.nn.functional as F


class SummaryEncoder:
    """
    Encodes:
      • summary (text)
      • context (text)
    into dense T5 embeddings.
    Provides summary/context fusion for Task 1a/1b.
    """

    def __init__(self, model_name="t5-small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🧠 Loading T5 model ({model_name}) on {self.device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        print("✅ T5 Encoder loaded successfully.")

    # -----------------------------------------------------
    # Internal method for encoding text → 1×D vector
    # -----------------------------------------------------
    def _encode(self, text):
        text = text.strip()
        if not text:
            return None

        self.model.eval()

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**tokens)
            emb = out.last_hidden_state.mean(dim=1)

        emb = emb.view(1, -1)
        emb = F.normalize(emb, dim=-1)
        return emb

    # -----------------------------------------------------
    def encode_summary(self, text):
        return self._encode(text)

    def encode_context(self, text):
        return self._encode(text)

    # -----------------------------------------------------
    # Fusion (Task 1b controllability insight)
    # -----------------------------------------------------
    def fuse_summary_and_context(self, s_vec, c_vec, alpha=0.75):
        """
        alpha = summary weight
        (1-alpha) = context weight
        """
        if s_vec is None and c_vec is None:
            return None
        if s_vec is None:
            return c_vec
        if c_vec is None:
            return s_vec

        fused = alpha * s_vec + (1 - alpha) * c_vec
        return F.normalize(fused, dim=-1)
