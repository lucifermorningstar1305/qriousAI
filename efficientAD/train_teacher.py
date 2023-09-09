from typing import Any, Optional, Callable, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.utils.data as td
import torchvision
import albumentations as A
import os
import sys
import wandb
import argparse

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, RichModelSummary
from pytorch_lightning.loggers.wandb import WandbLogger
from PIL import Image
from rich.progress import track
from collections import defaultdict
from ..utility.utils import print_table

def load_dataset(data_dir: str, split: str) -> Dict:
    """ Function to load dataset """
    
    dataset = defaultdict(lambda: list())

    for root, folders, files in os.walk(data_dir):
        if split in root:
            n_files = len(files)

            for i in track(range(n_files)):
                label = root.split("/")[-1]
                img_file = os.path.join(root, files[i])

                dataset["label"].append(label)
                dataset["img_path"].append(img_file)

    
    return dataset





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", "-d", type=str, required=True, help="the path of the pre-training dataset")

    args = parser.parse_args()

    data_dir = args.data_dir

    train_dataset = load_dataset(data_dir=data_dir, split="train")
    train_dataset = pd.DataFrame(train_dataset)

    val_dataset = load_dataset(data_dir=data_dir, split="val")
    val_dataset = pd.DataFrame(val_dataset)

    print_table(train_dataset, title="Training Dataset")
    

    

