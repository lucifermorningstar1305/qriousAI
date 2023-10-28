from typing import Any, Optional, Tuple, List, Dict, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import lightning.pytorch as pl
import torchvision
import random
import wandb
import pickle
import albumentations as A
import os

from rich.table import Table
from rich.console import Console
from rich.progress import track
from PIL import Image
from collections import defaultdict
from albumentations.core.transforms_interface import ImageOnlyTransform

random.seed(7)


def print_table(data: Any, title: Optional[str] = "Data", top_n: Optional[int] = 5):
    """ Function to print pandas dataframe with style """

    table = Table(title=title)

    colors = ["#f2593a", "#f2d63a", "#3af2cd", "#6ef23a", "#3aeff2",
              "#3a74f2", "#773af2", "#f23aec", "#f23a96", "#f23a71"]

    for i, col in enumerate(data.columns):

        if i == 0 or i == len(data.columns) - 1:
            table.add_column(col, justify="right",
                             style=colors[random.randint(0, len(colors)-1)])
        else:
            table.add_column(
                col, style=colors[random.randint(0, len(colors)-1)])

    if data.shape[0] < top_n:
        top_n = data.shape[0]

    top_data = data.head(top_n)
    for i in range(top_n):
        rec = top_data.iloc[i].map(lambda x: str(x)).tolist()
        table.add_row(*rec)

    console = Console()
    console.print(table)


def get_sample_images_w_prompts(ds: td.Dataset, num: Optional[int] = 8) -> torch.Tensor:
    """ Function to log reconstructed validation images to W&B """

    token_input_ids = torch.stack(
        [ds[i]["prompt_tokens"]["input_ids"] for i in range(num)], dim=0)
    token_attention_mask = torch.stack(
        [ds[i]["prompt_tokens"]["attention_mask"] for i in range(num)], dim=0)

    imgs = torch.stack([ds[i]["image"] for i in range(num)], dim=0)

    return {
        "image": imgs,
        "prompt_tokens": {"input_ids": token_input_ids,
                          "attention_mask": token_attention_mask}
    }


class GenerateCallback(pl.Callback):
    def __init__(self, samples: Dict, latent_dim: Optional[int] = 100, every_n_epochs: Optional[int] = 1):
        super().__init__()

        self.input_imgs = samples["image"]
        self.prompts = {k: v.squeeze()
                        for k, v in samples["prompt_tokens"].items()}
        self.latent_dim = latent_dim
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = self.input_imgs.to(pl_module.device)

            self.prompts = {k: v.to(pl_module.device)
                            for k, v in self.prompts.items()}

            with torch.no_grad():
                pl_module.eval()
                z = torch.randn(input_imgs.size(
                    0), self.latent_dim).type_as(input_imgs)

                z = z.view(z.size(0), self.latent_dim, 1, 1)

                text_embeddings = pl_module.bert_model(
                    **self.prompts).last_hidden_state[:, 0, :]  # [CLS] Token embedding
                reconstr_img = pl_module(z, text_embeddings)
                pl_module.train()

            std = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1,
                                                        1, 1).to(pl_module.device)
            mean = torch.Tensor([0.5, 0.5, 0.5]).reshape(-1,
                                                         1, 1).to(pl_module.device)
            reconstr_img = torch.clamp(reconstr_img * std + mean, 0, 1)

            imgs = torch.stack([input_imgs, reconstr_img], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(
                imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.log({
                "samples": [wandb.Image(grid, caption="Reconstructed Images")]
            }, step=trainer.global_step)


def as_anomaly(data_path: str) -> pd.DataFrame:
    """ Function to design the dataset as an anomaly detection dataset. """

    data = defaultdict(lambda: list())

    if not os.path.exists("./data/dataframe_for_anomaly.csv"):

        for root, folders, files in os.walk(data_path):
            n_files = len(files)

            for i in track(range(n_files)):

                img_file = files[i]

                label = root.split("/")[-1]
                split = root.split("/")[-2]

                if label in ["cat", "dog"]:
                    data["label"].append("normal")
                else:
                    data["label"].append("anomaly")

                data["class_label"].append(label)

                data["img_path"].append(os.path.join(root, img_file))
                data["split"].append(split)

        data = pd.DataFrame(data).sample(frac=1, random_state=32)
        data.to_csv("./data/dataframe_for_anomaly.csv", index=False)

    else:
        data = pd.read_csv("./data/dataframe_for_anomaly.csv")

    return data


def init_weights(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, .02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.)


def log_images(logger: Callable, org_img: torch.FloatTensor, gen_img: torch.FloatTensor, step: int):

    # std = torch.tensor([0.5, 0.5, 0.5], device=device).reshape(-1, 1, 1)
    # mean = torch.tensor([0.5, 0.5, 0.5], device=device).reshape(-1, 1, 1)
    # denorm_gen_imgs = torch.clamp((gen_imgs * std + mean), 0, 1)

    imgs = torch.stack([org_img, gen_img], dim=1).flatten(0, 1)
    grid = torchvision.utils.make_grid(
        imgs, nrow=2, normalize=True, range=(-1, 1))

    torchvision.utils.save_image(grid, "./media/images.png")
    logger.log(
        {"example": wandb.Image(grid, caption="Reconstructed Images")})
