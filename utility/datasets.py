"""
@author: Adityam Ghosh
Date: 10-29-2023

"""
from typing import Any, List, Dict, Tuple, Callable, Optional

import torch
import torch.utils.data as td
import torchvision
import pandas as pd
import numpy as np
import utility.transform_data as T
import albumentations as alb

from PIL import Image


class TextVisualDataset(td.Dataset):
    def __init__(
        self,
        data: Any,
        text_tokenizer: Callable,
        config: Dict,
        resize: Optional[Tuple] = None,
        transformations: Callable = T.DEFAULT_IMAGE_TRANSFORM,
    ):
        assert isinstance(
            data, pd.DataFrame
        ), f"Expected data to be a pandas Dataframe. Found {type(data)}"
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.resize = resize
        self.transformations = transformations
        self.text_transformations = alb.Compose(
            [
                T.NormalizeCaption(
                    max_caption_length=config["text_model"]["max_seq_length"]
                )
            ]
        )
        self.cfg = config

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        rec = self.data.iloc[idx]
        img_path = rec["image_path"]
        text = rec["text"]

        neg_sample = self.data.loc[self.data.index != idx].sample(n=1, random_state=32)
        neg_img_path = neg_sample["image_path"].values[0]
        neg_text = neg_sample["text"].values[0]

        img = Image.open(img_path).convert("RGB")
        neg_img = Image.open(neg_img_path).convert("RGB")

        if self.resize is not None:
            img = img.resize(
                (self.resize[1], self.resize[0]), resample=Image.Resampling.BILINEAR
            )

            neg_img = neg_img.resize(
                (self.resize[1], self.resize[0]), resample=Image.Resampling.BILINEAR
            )

        img = np.array(img)
        neg_img = np.array(neg_img)

        trans_og_data = self.transformations(image=img, caption=text)
        trans_neg_data = self.transformations(image=neg_img, caption=text)

        img, text = trans_og_data["image"], trans_og_data["caption"]
        neg_img, neg_text = trans_neg_data["image"], trans_neg_data["caption"]

        img = np.transpose(img, (2, 0, 1))
        neg_img = np.transpose(neg_img, (2, 0, 1))

        img = torch.tensor(img, dtype=torch.float)
        neg_img = torch.tensor(neg_img, dtype=torch.float)

        # text_obj = self.text_transformations(caption=text)
        # neg_text_obj = self.text_transformations(caption=neg_text)

        # text = text_obj["caption"]
        # neg_text = neg_text_obj["caption"]

        tok_res = self.text_tokenizer(
            text,
            max_length=self.cfg["text_model"]["max_seq_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        neg_tok_res = self.text_tokenizer(
            neg_text,
            max_length=self.cfg["text_model"]["max_seq_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {"img": img, "txt": tok_res, "neg_img": neg_img, "neg_txt": neg_tok_res}


class ZeroShotTextVisualDataset(td.Dataset):
    def __init__(
        self,
        data: Any,
        transformations: Callable,
        config: Dict,
    ):
        assert isinstance(
            data, pd.DataFrame
        ), f"Expected data to be a pandas Dataframe. Found {type(data)}"
        self.data = data
        self.transformations = transformations
        self.cfg = config

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        rec = self.data.iloc[idx]
        img_path = rec["image_path"]
        label = rec["label"]

        img = Image.open(img_path).convert("RGB")
        img = self.transformations(img)

        return {
            "img": img,
            "label": torch.tensor(label, dtype=torch.long),
        }
