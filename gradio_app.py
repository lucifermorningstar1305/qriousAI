from typing import Any, Optional, Dict, Tuple, List, Callable

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as alb
import yaml
import cv2
import os

from trainer import LitMobileCLiP
from models.grad_cam_models import ImageEncoder
from transformers import CLIPTokenizerFast

from PIL import Image


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Function to preprocess an image for the MobileCLIP model"""

    transformations = alb.Compose(
        [
            alb.Resize(224, 224, always_apply=True),
            alb.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255,
                always_apply=True,
            ),
        ]
    )

    img = np.array(img)
    img_obj = transformations(image=img)
    img = img_obj["image"]
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)

    img_tensor = torch.tensor(img, dtype=torch.float)
    return img_tensor


def preprocess_text(prompt: str, tokenizer: Callable, max_length: int) -> torch.Tensor:
    """Function to preprocess the text for the MobileCLIP model"""

    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "left"

    prompt = prompt.lower() + "."

    tokenized_prompt = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return tokenized_prompt


def get_text_tensors(
    txt: str, tokenizer: Callable, max_length: int, model: nn.Module
) -> torch.Tensor:
    """Function to calculate the get the tensors for each label"""

    prompt = f"A photo of a {txt}"

    tokenized_txt = preprocess_text(
        prompt=prompt, tokenizer=tokenizer, max_length=max_length
    )

    txt_out = model.encode_text(
        tokenized_txt["input_ids"].to("cuda:0"),
        tokenized_txt["attention_mask"].float().to("cuda:0"),
    )

    return txt_out


def calc_grad_cam(
    img_tensor: torch.Tensor,
    txt_tensor: torch.Tensor,
    img_model: nn.Module,
    img_proj_model: nn.Module,
):
    """Function to calculate the Grad-CAM"""

    img_out = img_model(img_tensor.unsqueeze(0).to("cuda:0"))
    img_encoded = img_proj_model(img_out)

    txt_encoded = txt_tensor

    sims = (
        F.normalize(img_encoded, p=2, dim=-1)
        @ F.normalize(txt_encoded, p=2, dim=-1).t()
    ).softmax(dim=-1)

    pred = torch.argmax(sims, dim=1)

    # Grad-CAM Calculation
    sims[:, pred].backward()
    gradients = img_model.get_activations_gradient()  # (BZ, NCHANNELS, HEIGHT, WIDTH)
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])  # (NCHANNELS)

    activations = img_model.get_activations(
        img_tensor.unsqueeze(0).to("cuda:0")
    ).detach()  # (BZ, NCHANNELS, HEIGHT, WIDTH)

    for i in range(gradients.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.maximum(heatmap, torch.tensor(0.0))

    heatmap /= torch.max(heatmap)

    return heatmap.cpu().numpy(), sims.detach().cpu().numpy().flatten()


def superimpose_gradcam(img: Image.Image, heatmap: np.ndarray):
    """Function to superimpose the grad-CAM on the original image"""
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    super_imposed_img = heatmap * 0.4 + img

    return super_imposed_img


def evaluate(img: Image.Image, txt: str):
    """Function to evaluate the clip model"""

    txts = txt.split(",")
    txts = list(map(lambda x: x.strip(), txts))

    cfg = None
    with open("./configs/config.yaml", "r") as fp:
        try:
            cfg = yaml.safe_load(fp)
        except yaml.YAMLError as err:
            raise Exception(err)

    lit_model = LitMobileCLiP.load_from_checkpoint(
        "./checkpoints/mobilenetv3_large_litetransformers_mscoco.ckpt", config=cfg
    )
    lit_model.eval()

    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    img_model = ImageEncoder(lit_model.clip_model.img_model)
    img_proj_model = lit_model.clip_model.img_projection

    img_tensor = preprocess_image(img)
    txt_tensors = [
        get_text_tensors(
            txt=label,
            tokenizer=tokenizer,
            max_length=cfg["text_model"]["max_seq_length"],
            model=lit_model,
        )
        for label in txts
    ]

    txt_tensors = torch.cat(txt_tensors, dim=0)
    heatmap, res = calc_grad_cam(
        img_tensor=img_tensor,
        txt_tensor=txt_tensors,
        img_model=img_model,
        img_proj_model=img_proj_model,
    )

    grad_cam_img = superimpose_gradcam(img, heatmap)
    cv2.imwrite("./media/grad_cam_output.jpg", grad_cam_img)

    pil_grad_cam_img = Image.open("./media/grad_cam_output.jpg")
    ret_labels = {label: float(round(sim, 2)) for label, sim in zip(txts, res)}
    print(ret_labels)
    print(res)

    return pil_grad_cam_img, ret_labels


with gr.Blocks() as demo:
    gr.Markdown(
        """ 
    # Mobile CLIP: Lean Visual-Language model for mobile Zero-shot classification
                
    This is a part of my dissertation project. The objective of this project was to see whether mobile models are capable of zero-shot classification or not.
    The model was trained on a $10\%$ subset of the MS-COCO Captions dataset. The results shows that there are signs of zero-shot capability even for mobile models.
    """
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            img = gr.Image(type="pil", label="Image", interactive=True)
            txt = gr.Textbox(
                value="",
                placeholder="cat, dog, bird, chair",
                max_lines=100,
                label="Labels",
                info="Prompt: A photo of a object",
            )

        with gr.Column(scale=1, min_width=400):
            out_img = gr.Image(
                type="pil", label="Output", height=320, width=640, interactive=False
            )
            out_labels = gr.Label(num_top_classes=3)

    eval_btn = gr.Button("Evaluate")
    eval_btn.click(
        fn=evaluate,
        inputs=[img, txt],
        outputs=[out_img, out_labels],
        api_name="zero_shot",
    )


if __name__ == "__main__":
    demo.launch()
