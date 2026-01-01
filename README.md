# T2V-Studio

Text-powered video generation system using diffusion models and AnimateDiff, optimized for Google Colab (T4 GPU) with an interactive Gradio interface. Built for Final Year CSE Major Project.

---

## 📌 Overview

T2V-Studio is a Generative AI project that converts natural language text prompts into short animated videos.  
The system leverages diffusion-based image generation models combined with a motion adapter (AnimateDiff) to synthesize temporally consistent video frames, which are then exported as MP4 animations.

This project focuses on **architecture, optimization, and deployment** of a text-to-video pipeline rather than training models from scratch.

---

## ✨ Features

- Text-to-video generation using diffusion models  
- Multiple visual styles (Realistic, Anime, Cinematic, Dreamy, Cartoon)  
- AnimateDiff-based motion synthesis  
- GPU-optimized execution for Google Colab (T4)  
- Adjustable parameters (frames, inference steps, guidance scale, seed)  
- Interactive Gradio web interface  
- Automatic MP4 video output  

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Frameworks & Libraries:** PyTorch, Diffusers, AnimateDiff  
- **Models:** Stable Diffusion (pretrained)  
- **UI:** Gradio  
- **Deployment:** Google Colab (T4 GPU)  

---

## 🧠 System Architecture

The project follows a modular architecture:

- `main.py`  
  Entry point of the application. Initializes and launches the Gradio interface.

- `pipeline.py`  
  Contains the core text-to-video generation logic.  
  Handles model loading, style-based pipeline selection, GPU optimization, frame generation, and MP4 export.

- `ui.py`  
  Defines the Gradio Blocks-based user interface, including prompt input, style selection, sliders, and video output.

This separation improves readability, scalability, and debugging.

---

## 🚀 How It Works

1. User enters a text prompt and selects a visual style.
2. The pipeline loads the corresponding diffusion model and AnimateDiff motion adapter.
3. Frames are generated based on the prompt and motion conditioning.
4. Frames are stitched into an MP4 video.
5. The video is displayed and made available for download.

---

## ▶️ How to Run (Google Colab)

1. Open the project in **Google Colab**
2. Enable GPU:  
   `Runtime → Change runtime type → GPU (T4)`
3. Run all setup and dependency cells
4. Execute `main.py`
5. Open the Gradio shareable link
6. Enter a prompt and generate a video

---

## 🧪 Example Prompt

A man walking through a glowing forest at night, cinematic lighting 


Recommended settings:
- Frames: 16  
- Inference Steps: 24  
- Guidance Scale: 7.5  
- Seed: 42  

---

## 📚 Learning Outcomes

- Practical understanding of diffusion-based generative models  
- Hands-on experience with AnimateDiff and motion-conditioned generation  
- GPU memory optimization techniques (VAE slicing, CPU offload, caching)  
- Modular AI system design and integration  
- Building and deploying interactive AI applications using Gradio  

---

## ⚠️ Limitations

- Generates short video clips due to GPU constraints  
- Dependent on pretrained models (no custom training)  
- Output quality varies based on prompt clarity and style  

---

## 🔮 Future Enhancements

- Support for longer video sequences  
- Custom-trained motion adapters  
- Batch video generation  
- Deployment on cloud platforms beyond Colab  
