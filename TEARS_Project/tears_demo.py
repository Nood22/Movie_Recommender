import gradio as gr

def get_recommendations(summary):
    return "\n".join([
        "The Intouchables",
        "La La Land",
        "Inception",
        "Amélie",
        "Jojo Rabbit"
    ])

demo = gr.Interface(
    fn=get_recommendations,
    inputs=gr.Textbox(lines=8, label="Enter your movie preference summary"),
    outputs=gr.Textbox(label="Top movie recommendations"),
    title="TEARS AI Movie Recommender",
    description="Get tailored movie suggestions based on your detailed summary."
)

demo.launch()