import torch, gc
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_video
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# 1. Motion adapter (core AnimateDiff module)
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5-2"

# 2. Style → base SD1.5 model mapping
STYLE_MODELS = {
    "Realistic": "SG161222/Realistic_Vision_V5.1_noVAE",
    "Anime": "xyn-ai/anything-v4.0",
    "3D / Dreamy": "Lykon/dreamshaper-8",
    "Cartoon": "Lykon/AAM_AnyLora_AnimeMix",
    "Cinematic": "SG161222/Realistic_Vision_V5.1_noVAE",  # reuse but different prompt tuning
}

# Cache so we don't reload weights every click
pipe_cache = {}
adapter = None

def get_motion_adapter():
    global adapter
    if adapter is None:
        adapter = MotionAdapter.from_pretrained(
            MOTION_ADAPTER_ID,
            torch_dtype=torch.float16
        )
    return adapter

def get_pipe(style: str) -> AnimateDiffPipeline:
    """
    Lazy-load a pipeline per style + reuse motion adapter.
    """
    if style in pipe_cache:
        return pipe_cache[style]

    model_id = STYLE_MODELS[style]
    motion_adapter = get_motion_adapter()

    pipe = AnimateDiffPipeline.from_pretrained(
        model_id,
        motion_adapter=motion_adapter,
        torch_dtype=torch.float16,
    )

    # Recommended DDIM scheduler config for AnimateDiff :contentReference[oaicite:3]{index=3}
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.scheduler = scheduler

    # Memory optimizations for Colab :contentReference[oaicite:4]{index=4}
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    pipe_cache[style] = pipe
    return pipe


def generate_video(prompt, style, num_frames, num_steps, guidance_scale, seed):
    if not prompt or prompt.strip() == "":
        return None

    pipe = get_pipe(style)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    # Slightly tuned negative prompt
    negative_prompt = "low quality, worst quality, artifacts, blurry, distorted"

    # Hard clamp to protect GPU
    num_frames = int(min(max(num_frames, 4), 24))
    num_steps = int(min(max(num_steps, 10), 32))

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        guidance_scale=float(guidance_scale),
        generator=generator,
    )

    frames = out.frames[0]
    video_path = export_to_video(frames, fps=8)

    # free some memory
    torch.cuda.empty_cache()
    gc.collect()

    return video_path
