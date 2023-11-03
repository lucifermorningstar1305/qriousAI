"""
@author: Adityam Ghosh
Date: 10-30-2023

"""

from typing import Callable, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import pytorch_lightning as pl
import torchvision
import os
import sys
import argparse
import yaml

from sklearn.metrics import accuracy_score, top_k_accuracy_score
from trainer import LitMobileCLiP
from transformers import CLIPTokenizerFast
from utility.datasets import ZeroShotTextVisualDataset
from PIL import Image
from utility.transform_data import (
    NormalizeCaption,
    IMAGENET_COLOR_MEAN,
    IMAGENET_COLOR_STD,
)
import albumentations as alb

from rich.progress import (
    Progress,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
    BarColumn,
)
from pprint import pprint
from transformers import AutoProcessor, CLIPModel

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def text_process(txt: str, tokenizer: Callable, max_length: int) -> torch.Tensor:
    """
    Function to obtain the text captions as a torch Tensor
    """
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "left"
    txt = txt.lower()
    txt += "."

    # text_transform = alb.Compose([NormalizeCaption(max_caption_length=max_length)])
    # txt_obj = text_transform(caption=txt)
    # txt = txt_obj["caption"]

    tok_outputs = tokenizer(
        txt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return tok_outputs


def get_text_tensors(text_captions: List, model: Callable) -> torch.Tensor:
    """Function to obtain the text tensors for all the captions"""

    prog_bar = Progress(
        TextColumn("[progress.percentage] {task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    text_tensors = []

    with prog_bar as p:
        n_captions = len(text_captions)

        for i in p.track(range(n_captions), description="Getting text tensors"):
            txt = text_captions[i]["input_ids"].to("cuda:0")
            attn_mask = text_captions[i]["attention_mask"].float().to("cuda:0")
            text_tensors.append(
                F.normalize(model.encode_text(txt, attn_mask), p=2, dim=-1)
                .detach()
                .cpu()
            )

    concat_text_tensor = torch.cat(text_tensors, dim=0)
    return concat_text_tensor


def evaluate(df: Any, model: Callable, prompts: List, processor: Callable) -> float:
    """Function to evaluate the model"""

    prog_bar = Progress(
        TextColumn("[progress.percentage] {task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    true_labels = list()
    pred_labels = list()
    preds = list()
    with torch.no_grad():
        with prog_bar as p:
            model.eval()
            for i in p.track(range(df.shape[0]), description="Evaluating Model"):
                rec = df.iloc[i]
                img = Image.open(rec["image_path"])
                label = rec["label"]

                inputs = processor(
                    text=prompts, images=img, return_tensors="pt", padding=True
                )

                similarities = model(**inputs).logits_per_image
                pred_label = torch.argmax(similarities.softmax(dim=1), dim=1)

                true_labels.append(label)
                preds.append(similarities.detach().cpu().numpy())
                pred_labels.append(pred_label.detach().cpu().numpy())

    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    preds = np.concatenate(preds, axis=0)
    # preds = np.squeeze(preds, axis=1)
    print(preds.shape)

    unique_preds, counts = np.unique(pred_labels, return_counts=True)
    unique_preds = unique_preds.reshape(-1, 1)
    counts = counts.reshape(-1, 1)
    print(f"frequency of each predicted classes by the model")
    print(np.hstack((unique_preds, counts)))

    return {
        "top_1_accuracy": top_k_accuracy_score(true_labels, preds, k=1),
        "top_5_accuracy": top_k_accuracy_score(true_labels, preds, k=5),
    }


if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(
        prog="evaluation script",
        description="Evaluation script for the Mobile CLiP model",
    )

    parser.add_argument(
        "--csv_path",
        "-C",
        required=True,
        type=str,
        help="the csv file path for the data",
    )

    parser.add_argument(
        "--model_checkpoint",
        "-c",
        required=True,
        type=str,
        help="the model checkpoint location",
    )

    parser.add_argument(
        "--config_path",
        "-p",
        required=True,
        type=str,
        help="the config path for the models",
    )

    args = parser.parse_args()

    csv_path = args.csv_path
    model_checkpoint = args.model_checkpoint
    config_path = args.config_path

    cfg = None
    with open(config_path, "r") as fp:
        try:
            cfg = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)

    df = pd.read_csv(csv_path)
    # print(df.loc[df["label"].isna()])
    # print(torch.load(model_checkpoint)["state_dict"].keys())
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # model.freeze()

    prompts = [
        "a picture of an airplane.",
        "a picture of an automobile.",
        "a picture of a bird.",
        "a picture of a cat.",
        "a picture of a deer.",
        "a picture of a dog.",
        "a picture of a frog.",
        "a picture of a horse.",
        "a picture of a ship.",
        "a picture of a truck.",
    ]

    integer_label_map = {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9,
    }
    pprint(integer_label_map)
    print(df["label"].unique())
    df["label"] = df["label"].str.lower().map(integer_label_map)
    print(df.head())

    transformations = alb.Compose(
        [
            alb.Resize(224, 224, always_apply=True),
            # alb.Normalize(
            #     mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD, always_apply=True
            # ),
        ]
    )

    test_ds = ZeroShotTextVisualDataset(
        data=df,
        transformations=transformations,
        config=cfg,
    )

    test_dl = td.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)

    acc = evaluate(df, model, prompts, processor)

    print(acc)
