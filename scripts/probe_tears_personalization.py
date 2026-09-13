"""Read-only comparison of raw and neutral-prior-adjusted TEARS rankings."""
import json
from pathlib import Path
import sqlite3
import sys
import argparse
from urllib.request import Request, urlopen
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from pilot_recommender import PilotHybridRecommender
from tears_preference_ranking import align_scores, explicit_genre_preferences
from study_runtime import signature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", help="Replay profiles through the live API using a synthetic session")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    with sqlite3.connect(f"file:{root}/logs/pilot_study.sqlite3?mode=ro", uri=True) as db:
        row = db.execute("SELECT payload_json FROM events WHERE request_id=?",
                         ("request-94708077-28f7-4681-86b5-9f6808cd0088",)).fetchone()
    event = json.loads(row[0])
    profiles = {
        "neutral": "Summary: No strong preference is supported for liked genres. No strong preference is supported for liked themes or content. No strong preference is supported for disliked genres or styles. No strong preference is supported for disliked plots or content.",
        "barbie_backend": event["effective_model_input"]["summary"],
        "barbie_display": event["representation"],
        "comedy": "Summary: The viewer enjoys comedy. They enjoy playful satire, whimsical humor, bright visuals and lighthearted storytelling. They dislike grim violence and bleak drama. They avoid frightening and graphic content.",
        "horror": "Summary: The viewer enjoys horror. They enjoy frightening supernatural stories, sustained dread and terrifying monsters. They dislike lighthearted comedy. They avoid cheerful whimsical stories and playful jokes.",
        "romance": "Summary: The viewer enjoys romance. They enjoy heartfelt love stories, romantic relationships and warm humor. They dislike violent action. They avoid superhero battles and space warfare.",
        "scifi": "Summary: The viewer enjoys science fiction and action. They enjoy space adventures, interstellar battles and heroic quests. They dislike romantic comedy. They avoid lighthearted romantic plots.",
    }
    if args.api:
        session_id = "ranking-probe-" + uuid4().hex
        for name in ("barbie_backend", "horror", "scifi"):
            payload = {k: v for k, v in event["request"].items()
                       if k not in {"system", "representation", "summary_source_request_id"}}
            payload["summary"] = profiles[name]
            snapshot = {"system": "TEARS", **payload}
            snapshot["representation"] = snapshot.pop("summary")
            payload["study"] = {
                "protocol_id": "tears-gers-human-study-tasks", "protocol_version": "1.0.0",
                "participant_id": "synthetic-ranking-probe", "session_id": session_id,
                "system": "TEARS", "task_id": "1b", "representation_revision": 1,
                "request_id": uuid4().hex,
                "input_signature": signature(snapshot),
            }
            request = Request(args.api.rstrip("/") + "/recommend", data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=120) as response:
                result = json.load(response)
            print(json.dumps({"profile": name, "request_id": payload["study"]["request_id"],
                "items": [{"title": item["title"], "genres": item["genres"],
                           "raw_model_score": item.get("raw_model_score")} for item in result["items"]]}), flush=True)
        return
    recommender = PilotHybridRecommender()
    encoded = recommender.tokenizer(list(profiles.values()), padding=True, truncation=True,
        max_length=512, return_tensors="pt").to(recommender.device)
    with torch.inference_mode():
        logits, _ = recommender.tears(encoded.input_ids, encoded.attention_mask)
    excluded = event["request"]["excluded_movie_ids"]
    memberships = {genre: torch.tensor([genre in str(value).split("|") for value in recommender.catalog.genres], device=recommender.device)
                   for genre in recommender.genre_names}
    print(json.dumps({"device": str(recommender.device), "profiles": list(profiles)}), flush=True)
    for index, name in enumerate(profiles):
        for correction in [0, "explicit_genre_alignment"]:
            preferences = explicit_genre_preferences(profiles[name])
            scores = (align_scores(logits[index:index+1], memberships, preferences)
                      if correction else logits[index:index+1])
            items = recommender._ranked_items(scores, excluded, 12, 2015)
            print(json.dumps({"profile": name, "prior_correction": correction,
                "preferences": preferences,
                "items": [i["title"] for i in items]}), flush=True)


if __name__ == "__main__":
    main()
