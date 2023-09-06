from typing import Any, Optional, Tuple, List, Dict, Callable

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as td
import lightning.pytorch as pl
import torchvision
import random
import wandb
import pickle
import albumentations as A

from rich.table import Table
from rich.console import Console
from rich.progress import track
from PIL import Image

random.seed(7)


def print_table(data: Any, title: Optional[str]="Data", top_n: Optional[int]=5):
    """ Function to print pandas dataframe with style """

    table = Table(title=title)

    colors = ["#f2593a", "#f2d63a", "#3af2cd", "#6ef23a", "#3aeff2", 
              "#3a74f2", "#773af2", "#f23aec", "#f23a96", "#f23a71"]
    
    for i, col in enumerate(data.columns):

        if i == 0 or i == len(data.columns) - 1:
            table.add_column(col, justify="right", style=colors[random.randint(0, len(colors)-1)])
        else:
            table.add_column(col, style=colors[random.randint(0, len(colors)-1)])

    top_data = data.head(top_n)

    for i in range(top_n):
        table.add_row(*top_data.iloc[i].tolist())
    
    console = Console()
    console.print(table)

def get_val_images(ds: td.Dataset, num: Optional[int]=8) -> torch.Tensor:
    """ Function to log reconstructed validation images to W&B """
    return torch.stack([ds[i]["image"] for i in range(num)], dim=0)

class GenerateCallback(pl.Callback):
    def __init__(self, input_imgs: torch.Tensor, every_n_epochs: Optional[int]=1):
        super().__init__()

        self.input_imgs = input_imgs
        self.every_n_epochs = every_n_epochs

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        
        if trainer.current_epoch % self.every_n_epochs == 0:
            input_imgs = self.input_imgs.to(pl_module.device)

            with torch.no_grad():
                pl_module.eval()
                reconstr_img = pl_module(input_imgs)
                pl_module.train()
            
            imgs = torch.stack([input_imgs, reconstr_img], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.log({
                "samples": [wandb.Image(grid, caption="Reconstructed Images")]
            }, step=trainer.global_step)


def get_latent_reps_and_error(data: td.DataLoader, model: pl.LightningModule) -> Tuple[np.ndarray, np.ndarray]:
    """ Function to obtain the latent representation of the data """
    
    device = torch.device("cuda:0")
    model = model.to(device)

    representations = list()
    errors = list()

    with torch.no_grad():
        model.eval()

        for batch in track(data):
            x = batch["image"].to(device)
            latent_rep = model.encoder(x)
            representations.append(latent_rep)

            pred = model(x)
            error = F.mse_loss(pred, x, reduction="none").sum(dim=[1, 2, 3])
            errors.append(error)
            
        representations = torch.cat(representations, dim=0)
        errors = torch.cat(errors, dim=0)
    
    representations = representations.detach().cpu().numpy()
    errors = errors.detach().cpu().numpy()

    return representations, errors

def calc_density(representations: np.ndarray, kde_path: str) -> np.ndarray:
    """ Function to calculate the Kernel density """

    densities = list()

    kde_model = pickle.load(open(kde_path, "rb"))

    for i in track(range(representations.shape[0])):
        rep = representations[i].reshape(1, -1)

        density = kde_model.score_samples(rep)[0]
        densities.append(density)

    return np.asarray(densities)


def process_img(img_path: str) -> torch.Tensor:
    """ Function to preprocess image """

    img = Image.open(img_path)
    img = img.convert("RGB")

    img = img.resize((224, 224), resample=Image.Resampling.BILINEAR)
    img = np.array(img)

    transformation = A.Compose([A.Normalize(mean=.5, std=.5, always_apply=True)])
    transf_obj = transformation(image=img)
    transformed_img = transf_obj["image"]

    transformed_img = np.transpose(transformed_img, (2, 0, 1)).astype(np.float32)
    return torch.tensor(transformed_img, dtype=torch.float32)


def isAnomaly(img_tensor: torch.Tensor, model: pl.LightningModule, kde_model: Callable, error_threshold: float, kde_threshold: float) -> bool:
    """ Function to check if an image is an anomalous image or not """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        model.eval()

        latent_representation = model.encoder(img_tensor.to(device)).detach().cpu().numpy()
        density = kde_model.score_samples(latent_representation)[0]

        reconstructed_img = model(img_tensor.to(device))
        loss = F.mse_loss(reconstructed_img, img_tensor.to(device), reduction="none").sum(dim=[1, 2, 3])

    print(loss.item(), density, error_threshold, kde_threshold)
    if loss.item() > error_threshold or density < kde_threshold:
        return True
    
    else:
        return False
            



