from flask import Flask, request, jsonify
import torch
from model.MF import get_model, get_tokenizer
from helper.dataloader import map_id_to_title
from collections import defaultdict

# راه‌اندازی Flask app
app = Flask(__name__)

# ===== پارامترهای مدل =====
rank = 0
world_size = 1

class Args:
    pass

args = Args()
args.data_name = "ml-1m"
args.embedding_module = "OTRecVAE"
args.scratch = "/home/mila/a/adls/saved_model/ml-1m"
args.bs = 64
args.lora_r = 64
args.kfac = 2
args.concat = False

# بارگذاری توکنایزر و عنوان فیلم‌ها
tokenizer = get_tokenizer(args)
item_title_dict = map_id_to_title(args.data_name)
num_movies = len(item_title_dict)

# بارگذاری مدل و وزن‌ها
model_path = f"{args.scratch}/saved_model/{args.data_name}/ot_train_vae_ml-1m_embedding_module_OTRecVAE_2024-09-24_13-37-29_2022.csv.pt"
model, lora_config = get_model(args, tokenizer, num_movies, rank, world_size)
model.to(rank)
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)

# ===== endpoint اصلی =====
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    summary = data.get("summary", "")
    if not summary:
        return jsonify({"error": "No summary provided"}), 400
    
    # آماده‌سازی tensor خالی برای ورودی مدل
    data_tensor = torch.zeros((num_movies)).to(rank) - 1e20

    # گرفتن پیشنهادها
    ranked_logits, _ = model.generate_recommendations(summary, tokenizer, data_tensor, topk=10, alpha=0)

    # تبدیل آی‌دی‌ها به نام فیلم
    movie_titles = [item_title_dict[i] for i in ranked_logits]
    return jsonify({"recommendations": movie_titles})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
