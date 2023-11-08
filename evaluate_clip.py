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
import pickle
import json

from sklearn.metrics import accuracy_score, top_k_accuracy_score
from trainer_clip import LitCLIP
from transformers import CLIPTokenizerFast, CLIPImageProcessor, CLIPModel, CLIPConfig
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

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def text_process(txt: str, tokenizer: Callable, max_length: int) -> torch.Tensor:
    """
    Function to obtain the text captions as a torch Tensor
    """

    txt = txt.lower()
    txt += "."

    # text_transform = alb.Compose([NormalizeCaption(max_caption_length=max_length)])
    # txt_obj = text_transform(caption=txt)
    # txt = txt_obj["caption"]

    tok_outputs = tokenizer(
        txt,
        max_length=77,
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


def evaluate(dataloader: Any, model: Callable, text_tensors: torch.Tensor) -> float:
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

    img_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inv_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToPILImage()]
    )

    with prog_bar as p:
        model.eval()
        for batch in p.track(dataloader, description="Evaluating Model"):
            img = batch[0]
            label = batch[1]
            img = inv_transform(img.squeeze())  # Returns a PIL image
            img = img_processor(img, return_tensors="pt")

            img_encoding = model.encode_image(img["pixel_values"].to("cuda:0"))
            img_encoding = F.normalize(img_encoding, p=2, dim=-1)
            text_tensors = text_tensors.cuda()
            similarities = (100.0 * img_encoding @ text_tensors.t()).softmax(dim=-1)
            pred_label = torch.argmax(similarities, dim=1)

            true_labels.append(label.detach().numpy())
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


def unpickle(file_path: str) -> Dict:
    """Function to unpickle the meta data of the CIFAR dataset"""

    with open(file_path, "rb") as fp:
        data = pickle.load(fp)

    return data


def fgvc_aircraft_res(
    root_dir: str,
    annotation_level: str,
    transformations: Callable,
    text_tokenizer: Callable,
    meta_path: str,
    prompt_template: str,
    model: Callable,
    cfg: Dict,
):
    """Function to obtain results for the FGVC Aircraft dataset"""

    fgcv_dataset = torchvision.datasets.FGVCAircraft(
        root=root_dir,
        split="test",
        annotation_level=annotation_level,
        transform=transformations,
        download=True,
    )

    with open(meta_path, "r") as fp:
        meta_data = fp.readlines()

    label_map = {k.replace("\n", ""): idx for idx, k in enumerate(meta_data)}
    pprint(label_map)

    prompts = list(
        map(
            lambda x: prompt_template + " " + x.replace("\n", ""),
            meta_data,
        )
    )
    tokenized_txts = [
        text_process(txt, text_tokenizer, cfg["text_model"]["max_seq_length"])
        for txt in prompts
    ]

    txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model)
    test_dl = td.DataLoader(fgcv_dataset, batch_size=1, shuffle=False, num_workers=1)
    acc = evaluate(test_dl, model, txt_tensors)

    print(acc)


if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(
        prog="evaluation script",
        description="Evaluation script for the Mobile CLiP model",
    )

    parser.add_argument(
        "--root_dir",
        "-r",
        required=False,
        type=str,
        default="./eval_datasets",
        help="the location where to download the datasets if not present",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        type=str,
        help="the name of the dataset for evaluation",
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

    parser.add_argument(
        "--prompt",
        "-P",
        required=False,
        type=str,
        default="a photo of a",
        help="the prompt template to use",
    )

    args = parser.parse_args()

    root_dir = args.root_dir
    dataset_name = args.dataset
    model_checkpoint = args.model_checkpoint
    config_path = args.config_path
    prompt_template = args.prompt

    cfg = None
    with open(config_path, "r") as fp:
        try:
            cfg = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)

    # print(df.loc[df["label"].isna()])
    # print(torch.load(model_checkpoint)["state_dict"].keys())
    model = LitCLIP.load_from_checkpoint(
        model_checkpoint, model=CLIPModel(CLIPConfig())
    )
    model.freeze()
    # clip_model = model.clip_model

    text_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    transformations = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ]
    )
    if dataset_name == "cifar10":
        cifar10_dataset = torchvision.datasets.CIFAR10(
            root=root_dir, train=False, download=True, transform=transformations
        )

        meta_path = os.path.join(root_dir, "cifar-10-batches-py/batches.meta")
        meta_data = unpickle(meta_path)
        prompts = list(
            map(lambda x: prompt_template + " " + x, meta_data["label_names"])
        )
        label_map = {k: idx for idx, k in enumerate(meta_data["label_names"])}

        tokenized_txts = [
            text_process(txt, text_tokenizer, cfg["text_model"]["max_seq_length"])
            for txt in prompts
        ]

        pprint(label_map)

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model)

        test_dl = td.DataLoader(
            cifar10_dataset, batch_size=1, shuffle=False, num_workers=1
        )

        acc = evaluate(test_dl, model, txt_tensors)
        print(acc)

    elif dataset_name == "cifar100":
        cifar100_dataset = torchvision.datasets.CIFAR100(
            root=root_dir, train=False, download=True, transform=transformations
        )

        meta_path = os.path.join(root_dir, "cifar-100-python/meta")
        meta_data = unpickle(meta_path)
        prompts = list(
            map(lambda x: prompt_template + " " + x, meta_data["fine_label_names"])
        )
        label_map = {k: idx for idx, k in enumerate(meta_data["fine_label_names"])}
        tokenized_txts = [
            text_process(txt, text_tokenizer, cfg["text_model"]["max_seq_length"])
            for txt in prompts
        ]

        pprint(label_map)

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model)

        test_dl = td.DataLoader(
            cifar100_dataset, batch_size=1, shuffle=False, num_workers=1
        )

        acc = evaluate(test_dl, model, txt_tensors)
        print(acc)

    elif dataset_name == "food101":
        food101_dataset = torchvision.datasets.Food101(
            root=root_dir, split="test", transform=transformations, download=True
        )

        meta_path = os.path.join(root_dir, "food-101/meta/labels.txt")
        with open(meta_path, "r") as fp:
            meta_data = fp.readlines()

        label_map = {k.replace("\n", ""): idx for idx, k in enumerate(meta_data)}
        prompts = list(
            map(lambda x: prompt_template + " " + x.replace("\n", ""), meta_data)
        )
        tokenized_txts = [
            text_process(txt, text_tokenizer, cfg["text_model"]["max_seq_length"])
            for txt in prompts
        ]

        pprint(label_map)

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model)
        test_dl = td.DataLoader(
            food101_dataset, batch_size=1, shuffle=False, num_workers=1
        )

        acc = evaluate(test_dl, model, txt_tensors)
        print(acc)

    elif dataset_name == "fgcv_aircraft":
        meta_path_families = os.path.join(
            root_dir, "fgvc-aircraft-2013b/data/families.txt"
        )
        meta_path_variant = os.path.join(
            root_dir, "fgvc-aircraft-2013b/data/variants.txt"
        )
        meta_path_manufactures = os.path.join(
            root_dir, "fgvc-aircraft-2013b/data/manufacturers.txt"
        )

        print("########## Annotation Level : Family ############")
        fgvc_aircraft_res(
            root_dir=root_dir,
            annotation_level="family",
            transformations=transformations,
            text_tokenizer=text_tokenizer,
            meta_path=meta_path_families,
            prompt_template=prompt_template,
            model=model,
            cfg=cfg,
        )

        print("########## Annotation Level : Variant ############")
        fgvc_aircraft_res(
            root_dir=root_dir,
            annotation_level="variant",
            transformations=transformations,
            text_tokenizer=text_tokenizer,
            meta_path=meta_path_variant,
            prompt_template=prompt_template,
            model=model,
            cfg=cfg,
        )

        print("########## Annotation Level : Manufacturer ############")
        fgvc_aircraft_res(
            root_dir=root_dir,
            annotation_level="manufacturer",
            transformations=transformations,
            text_tokenizer=text_tokenizer,
            meta_path=meta_path_manufactures,
            prompt_template=prompt_template,
            model=model,
            cfg=cfg,
        )
    elif dataset_name == "caltech_256":
        transformations_caltech = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x
                ),
                torchvision.transforms.Normalize(
                    mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD
                ),
            ]
        )
        caltech_dataset = torchvision.datasets.Caltech256(
            root=root_dir, transform=transformations_caltech, download=True
        )

        meta_path = os.path.join(root_dir, "caltech256/caltech_labels.json")

        with open(meta_path, "r") as fp:
            meta_data = json.load(fp)

        pprint(meta_data)

        prompts = [prompt_template + " " + labels for labels, _ in meta_data.items()]

        tokenized_txts = [
            text_process(txt, text_tokenizer, cfg["text_model"]["max_seq_length"])
            for txt in prompts
        ]

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model)
        test_dl = td.DataLoader(
            caltech_dataset, batch_size=1, shuffle=False, num_workers=1
        )

        acc = evaluate(test_dl, model, txt_tensors)
        print(acc)

    elif dataset_name == "oxford_pets":
        oxford_pets_dataset = torchvision.datasets.OxfordIIITPet(
            root=root_dir,
            split="test",
            transform=transformations,
            target_types="category",
            download=True,
        )

        meta_path = os.path.join(root_dir, "oxford-iiit-pet/oxford_pets_labels.json")

        with open(meta_path, "r") as fp:
            meta_data = json.load(fp)

        pprint(dict(sorted(meta_data.items(), key=lambda x: x[1])))

        prompts = [prompt_template + " " + labels for labels, _ in meta_data.items()]

        tokenized_txts = [
            text_process(txt, text_tokenizer, cfg["text_model"]["max_seq_length"])
            for txt in prompts
        ]

        txt_tensors = get_text_tensors(text_captions=tokenized_txts, model=model)
        test_dl = td.DataLoader(
            oxford_pets_dataset, batch_size=1, shuffle=False, num_workers=1
        )

        acc = evaluate(test_dl, model, txt_tensors)
        print(acc)
