import os

import gradio as gr
import requests


RECOMMENDER_API_URL = os.getenv(
    "TEARS_RECOMMENDER_API_URL",
    "http://127.0.0.1:8001/recommend",
)


def parse_comma_separated(value: str) -> list[str]:
    return [entry.strip() for entry in value.split(",") if entry.strip()]


def get_recommendations(
    description: str,
    liked_movies: str,
    disliked_genres: str,
    top_k: int,
) -> str:
    payload = {
        "description": description.strip(),
        "liked": parse_comma_separated(liked_movies),
        "disliked_genres": parse_comma_separated(disliked_genres),
        "top_k": int(top_k),
    }

    if not payload["description"] and not payload["liked"]:
        return "Enter a taste description or at least one liked movie."

    try:
        response = requests.post(
            RECOMMENDER_API_URL,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        items = response.json()["items"]
    except requests.RequestException as error:
        detail = ""
        if error.response is not None:
            try:
                detail = error.response.json().get("detail", error.response.text)
            except ValueError:
                detail = error.response.text
        return f"Recommendation service error: {detail or error}"
    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected response from recommendation service: {error}"

    if not items:
        return "No recommendations matched the selected preferences."

    return "\n".join(
        f'{item["rank"]}. {item["title"]} — {", ".join(item["genres"])}'
        for item in items
    )


with gr.Blocks() as demo:
    gr.Markdown("## Movie Recommender (TEARS + EASE)")
    gr.Markdown(
        "Describe your taste and add movies you already like. "
        "Liked movies are used as recommendation evidence and will not be returned."
    )

    description = gr.Textbox(
        lines=5,
        label="Movie taste description",
        placeholder="Atmospheric, intelligent science fiction and action",
    )
    liked_movies = gr.Textbox(
        label="Movies you liked (comma-separated)",
        placeholder="Alien (1979), Blade Runner (1982)",
    )
    disliked_genres = gr.Textbox(
        label="Genres to exclude (comma-separated)",
        placeholder="Romance, Comedy",
    )
    top_k = gr.Slider(
        minimum=1,
        maximum=25,
        value=10,
        step=1,
        label="Number of recommendations",
    )
    output = gr.Textbox(lines=12, label="Recommended movies")
    submit = gr.Button("Recommend", variant="primary")

    submit.click(
        fn=get_recommendations,
        inputs=[description, liked_movies, disliked_genres, top_k],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch()
