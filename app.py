import os
import random
import uuid
import json
import time
import asyncio
from threading import Thread

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForImageTextToText,
    AutoProcessor,
    TextIteratorStreamer,
)
from transformers.image_utils import load_image

# Constants for text generation
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load VIREX-062225-exp
MODEL_ID_M = "prithivMLmods/VIREX-062225-exp"
processor_m = AutoProcessor.from_pretrained(MODEL_ID_M, trust_remote_code=True)
model_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load DREX-062225-exp
MODEL_ID_X = "prithivMLmods/DREX-062225-exp"
processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True)
model_x = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_X,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load typhoon-ocr-3b
MODEL_ID_T = "scb10x/typhoon-ocr-3b"
processor_t = AutoProcessor.from_pretrained(MODEL_ID_T, trust_remote_code=True)
model_t = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_T,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load olmOCR-7B-0225-preview
MODEL_ID_O = "allenai/olmOCR-7B-0225-preview"
processor_o = AutoProcessor.from_pretrained(MODEL_ID_O, trust_remote_code=True)
model_o = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID_O,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

def downsample_video(video_path):
    """
    Downsamples the video to evenly spaced frames.
    Each frame is returned as a PIL image along with its timestamp.
    """
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames

@spaces.GPU
def generate_image(model_name: str, text: str, image: Image.Image,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generates responses using the selected model for image input.
    """
    if model_name == "VIREX-062225-7B-exp":
        processor = processor_m
        model = model_m
    elif model_name == "DREX-062225-7B-exp":
        processor = processor_x
        model = model_x
    elif model_name == "olmOCR-7B-0225-preview":
        processor = processor_o
        model = model_o
    elif model_name == "Typhoon-OCR-3B":
        processor = processor_t
        model = model_t
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]
    }]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=False,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "streamer": streamer, "max_new_tokens": max_new_tokens}
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        time.sleep(0.01)
        yield buffer, buffer

@spaces.GPU
def generate_video(model_name: str, text: str, video_path: str,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generates responses using the selected model for video input.
    """
    if model_name == "VIREX-062225-7B-exp":
        processor = processor_m
        model = model_m
    elif model_name == "DREX-062225-7B-exp":
        processor = processor_x
        model = model_x
    elif model_name == "olmOCR-7B-0225-preview":
        processor = processor_o
        model = model_o
    elif model_name == "Typhoon-OCR-3B":
        processor = processor_t
        model = model_t
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if video_path is None:
        yield "Please upload a video.", "Please upload a video."
        return

    frames = downsample_video(video_path)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "text": text}]}
    ]
    for frame in frames:
        image, timestamp = frame
        messages[1]["content"].append({"type": "text", "text": f"Frame {timestamp}:"})
        messages[1]["content"].append({"type": "image", "image": image})
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
        max_length=MAX_INPUT_TOKEN_LENGTH
    ).to(device)
    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer

# Define examples for image and video inference
image_examples = [
    ["Convert this page to doc [text] precisely.", "images/3.png"],
    ["Convert this page to doc [text] precisely.", "images/4.png"],
    ["Convert this page to doc [text] precisely.", "images/1.png"],
    ["Convert chart to OTSL.", "images/2.png"]
]

video_examples = [
    ["Explain the video in detail.", "videos/2.mp4"],
    ["Explain the ad in detail.", "videos/1.mp4"]
]

# Added CSS to style the output area as a "Canvas"
css = """
.submit-btn {
    background-color: #2980b9 !important;
    color: white !important;
}
.submit-btn:hover {
    background-color: #3498db !important;
}
.canvas-output {
    border: 2px solid #4682B4;
    border-radius: 10px;
    padding: 20px;
}
"""

# Create the Gradio Interface
with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown("# **[Doc VLMs OCR](https://huggingface.co/collections/prithivMLmods/multimodal-implementations-67c9982ea04b39f0608badb0)**")
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    image_upload = gr.Image(type="pil", label="Image")
                    image_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(
                        examples=image_examples,
                        inputs=[image_query, image_upload]
                    )
                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
                    video_upload = gr.Video(label="Video")
                    video_submit = gr.Button("Submit", elem_classes="submit-btn")
                    gr.Examples(
                        examples=video_examples,
                        inputs=[video_query, video_upload]
                    )
            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.2)

        with gr.Column():
            with gr.Column(elem_classes="canvas-output"):
                gr.Markdown("## Result Canvas")
                output = gr.Textbox(label="Raw Output Stream", interactive=False, lines=2)
                markdown_output = gr.Markdown(label="Formatted Result (Result.Md)")

            model_choice = gr.Radio(
                choices=["DREX-062225-7B-exp", "olmOCR-7B-0225-preview", "VIREX-062225-7B-exp", "Typhoon-OCR-3B"],
                label="Select Model",
                value="DREX-062225-7B-exp"
            )

            gr.Markdown("**Model Info 💻** | [Report Bug](https://huggingface.co/spaces/prithivMLmods/Doc-VLMs/discussions)")          
            gr.Markdown("> [DREX-062225-7B-exp](https://huggingface.co/prithivMLmods/DREX-062225-exp): the drex-062225-exp (document retrieval and extraction expert) model is a specialized fine-tuned version of docscopeocr-7b-050425-exp, optimized for document retrieval, content extraction, and analysis recognition. built on top of the qwen2.5-vl architecture.")
            gr.Markdown("> [VIREX-062225-7B-exp](https://huggingface.co/prithivMLmods/VIREX-062225-exp): the virex-062225-exp (video information retrieval and extraction expert - experimental) model is a fine-tuned version of qwen2.5-vl-7b-instruct, specifically optimized for advanced video understanding, image comprehension, sense of reasoning, and natural language decision-making through cot reasoning.")
            gr.Markdown("> [Typhoon-OCR-3B](https://huggingface.co/scb10x/typhoon-ocr-3b): a bilingual document parsing model built specifically for real-world documents in thai and english, inspired by models like olmocr, based on qwen2.5-vl-instruction. this model is intended to be used with a specific prompt only.")
            gr.Markdown("> [olmOCR-7B-0225](https://huggingface.co/allenai/olmOCR-7B-0225-preview): the olmocr-7b-0225-preview model is based on qwen2-vl-7b, optimized for document-level optical character recognition (ocr), long-context vision-language understanding, and accurate image-to-text conversion with mathematical latex formatting. designed with a focus on high-fidelity visual-textual comprehension.")
            gr.Markdown(">⚠️note: all the models in space are not guaranteed to perform well in video inference use cases.")  
    
    image_submit.click(
        fn=generate_image,
        inputs=[model_choice, image_query, image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=[output, markdown_output]
    )
    video_submit.click(
        fn=generate_video,
        inputs=[model_choice, video_query, video_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=[output, markdown_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=30).launch(share=True, mcp_server=True, ssr_mode=False, show_error=True)
