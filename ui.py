from pipeline import generate_video, STYLE_MODELS
import gradio as gr



styles = list(STYLE_MODELS.keys())

with gr.Blocks() as demo:
    gr.Markdown("# T2V Studio : Text Powered Video Generator")
    gr.Markdown(
        "Enter a text prompt, choose a style (1 Style per 1 Execution), and generate a short AI animation."
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Text Prompt",
                placeholder="e.g. a man walking forward in a windy environment",
                lines=3,
            )
            style = gr.Dropdown(
                label="Style",
                choices=styles,
                value="Realistic",
            )
            num_frames = gr.Slider(
                label="Frames",
                minimum=4,
                maximum=24,
                value=16,
                step=1,
            )
            num_steps = gr.Slider(
                label="Inference steps",
                minimum=10,
                maximum=32,
                value=24,
                step=2,
            )
            guidance_scale = gr.Slider(
                label="Guidance scale",
                minimum=3.0,
                maximum=12.0,
                value=7.5,
                step=0.5,
            )
            seed = gr.Number(
                label="Seed (for reproducibility)",
                value=42,
                precision=0,
            )
            generate_btn = gr.Button("Generate Video", variant="primary")

        with gr.Column(scale=1):
            output_video = gr.Video(label="Output Video (mp4/webm)")

    generate_btn.click(
        fn=generate_video,
        inputs=[prompt, style, num_frames, num_steps, guidance_scale, seed],
        outputs=output_video,
    )
