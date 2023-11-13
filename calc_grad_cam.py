from typing import Any, Dict, Optional, List, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import albumentations as alb
import cv2
import yaml
import argparse
import os
import sys

from PIL import Image
from transformers import CLIPTokenizerFast
from trainer import LitMobileCLiP
from models.grad_cam_models import ImageEncoder


def preprocess_image(img_path: str) -> torch.Tensor:
    """Function to preprocess an image for the MobileCLIP model"""

    img = Image.open(img_path).convert("RGB")

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


def preprocess_text(prompt: str, max_length: int) -> torch.Tensor:
    """Function to preprocess the text for the MobileCLIP model"""

    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
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


def get_grad_cam(
    img_tensor: torch.Tensor,
    txt_tensor: Dict,
    model: torch.nn.Module,
    img_model: torch.nn.Module,
    img_projection: nn.Module,
):
    """Function to calculate the grad cam"""

    # img_encoded = model.encode_image(img_tensor.unsqueeze(0).to("cuda:0"))

    img_out = img_model(img_tensor.unsqueeze(0).to("cuda:0"))
    img_encoded = img_projection(img_out)

    txt_encoded = model.encode_text(
        txt_tensor["input_ids"].to("cuda:0"),
        txt_tensor["attention_mask"].float().to("cuda:0"),
    )

    res = (
        F.normalize(img_encoded, p=2, dim=-1)
        @ F.normalize(txt_encoded, p=2, dim=-1).t()
    )
    # print(res.shape, res)

    res.backward()
    gradients = img_model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    activations = img_model.get_activations(
        img_tensor.unsqueeze(0).to("cuda:0")
    ).detach()

    for i in range(gradients.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]

    # print(activations.shape)
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.maximum(heatmap, torch.tensor(0.0))

    heatmap /= torch.max(heatmap)
    # print(heatmap.shape)
    # plt.matshow(heatmap.cpu().numpy())
    # plt.show()

    return heatmap.cpu().numpy()


def show_grad_cam(img_path: str, heatmap: np.ndarray):
    """Function to show the grad cam"""

    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    supimposed_img = heatmap * 0.4 + img

    cv2.imwrite("grad_img.jpg", supimposed_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_path",
        "-i",
        required=True,
        type=str,
        help="the path where the image is stored",
    )
    parser.add_argument(
        "--prompt", "-p", required=True, type=str, help="the prompt for the image."
    )
    parser.add_argument(
        "--config_path", "-c", required=True, type=str, help="the config path"
    )

    parser.add_argument(
        "--checkpoint_path",
        "-C",
        required=True,
        type=str,
        help="the model checkpoint path",
    )

    args = parser.parse_args()

    image_path = args.image_path
    prompt = args.prompt
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path

    # Load the config
    cfg = None
    with open(config_path, "r") as fp:
        try:
            cfg = yaml.safe_load(fp)

        except yaml.YAMLError as err:
            raise Exception(err)

    lit_model = LitMobileCLiP.load_from_checkpoint(checkpoint_path, config=cfg)
    lit_model.eval()

    img_model = ImageEncoder(lit_model.clip_model.img_model)
    img_projection = lit_model.clip_model.img_projection

    img_tensor = preprocess_image(img_path=image_path)
    tokenized_prompts = preprocess_text(
        prompt=prompt, max_length=cfg["text_model"]["max_seq_length"]
    )

    grad_cam = get_grad_cam(
        img_tensor=img_tensor,
        txt_tensor=tokenized_prompts,
        model=lit_model,
        img_model=img_model,
        img_projection=img_projection,
    )

    show_grad_cam(img_path=image_path, heatmap=grad_cam)
