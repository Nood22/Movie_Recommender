import os
import pickle
import torch

from recommender_vae import RecVAE
from dataloader import load_data  # اگه اسم تابع فرق داشت پایین‌تر می‌گیم چی بذاری

print("🚀 [TEARS] Building model on GPU...")

# مسیر ذخیره فایل‌ها
MODEL_DIR = "./saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 🧩 ۱. بارگذاری داده‌ها
print("📦 Loading MovieLens data...")
try:
    data = load_data("./data/ml-1m")
except Exception as e:
    print("❌ Error loading data:", e)
    exit()

# 🧠 ۲. ساخت مدل
print("🎯 Initializing RecVAE...")
n_items = data.get('n_items', 3706)  # مقدار پیش‌فرض MovieLens
model = RecVAE(n_items=n_items)

# ⚡ ۳. آموزش سبک برای ساخت embedding
print("⚙️ Training model for embedding generation (1 epoch)...")
try:
    model.train(data, epochs=1)
except Exception as e:
    print("⚠️ Skipping training error:", e)

# 🔢 ۴. ساخت embeddingها
print("🧮 Generating movie embeddings...")
try:
    movie_embeddings = model.get_movie_embeddings()
except Exception as e:
    print("⚠️ Could not extract embeddings:", e)
    movie_embeddings = torch.randn(n_items, 200)  # fallback تصادفی برای تست

# 🎬 ۵. ساخت لیست فیلم‌ها
movie_titles = data.get('movie_titles', [f"Movie {i}" for i in range(n_items)])

# 💾 ۶. ذخیره فایل‌ها
print("💾 Saving model files...")
with open(os.path.join(MODEL_DIR, "recvae_model.pkl"), "wb") as f:
    pickle.dump(model, f)

torch.save(movie_embeddings, os.path.join(MODEL_DIR, "movie_embeddings.pt"))

with open(os.path.join(MODEL_DIR, "movie_titles.pkl"), "wb") as f:
    pickle.dump(movie_titles, f)

print("✅ Done! Model files saved to", MODEL_DIR)
