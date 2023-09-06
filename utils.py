from typing import Any, Optional

import torch
import torch.utils.data as td
import lightning.pytorch as pl
import torchvision
import random
import wandb

from rich.table import Table
from rich.console import Console

random.seed(7)


def print_table(data: Any, title: Optional[str]="Data", top_n: Optional[int]=5):
    """ Function to print pandas dataframe with style """

    table = Table(title=title)

    colors = ["#f2593a", "#f2d63a", "#3af2cd", "#6ef23a", "#3aeff2", 
              "#3a74f2", "#773af2", "#f23aec", "#f23a96", "#f23a71"]
    
    for i, col in enumerate(data.columns):

        if i == 0 or i == len(data.columns) - 1:
            table.add_column(col, justify="right", style=colors[random.randint(0, len(colors))])
        else:
            table.add_column(col, style=colors[random.randint(0, len(colors))])

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