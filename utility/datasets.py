"""
@author: Adityam Ghosh
Date: 10-29-2023

"""
from typing import Any, List, Dict, Tuple, Callable

import torch.utils.data as td
import pandas as pd

from PIL import Image


class TextVisualDataset(td.Dataset):
    def __init__(self, data: Any, text_tokenizer: Callable, transformations: Callable, config: Dict):

        assert isinstance(
            data, pd.DataFrame), f"Expected data to be a pandas Dataframe. Found {type(data)}"
        self.data = data
        self.text_tokenizer = text_tokenizer
        self.transformations = transformations
        self.cfg = config

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx: int):

        rec = self.data.iloc[idx]
        img_path = rec["image_path"]
        text = rec["text"]

        img = Image.open(img_path).convert("RGB")
        img = self.transformations(img)

        tok_res = self.tokenizer(
            text, max_length=self.cfg["text_model"]["max_seq_length"])

        return {
            "img": img,
            "txt": tok_res
        }
