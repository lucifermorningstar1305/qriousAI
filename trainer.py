"""
@author: Adityam Ghosh
Date: 10-29-2023

"""
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from models.mobile_clip import MobileCLiP
from losses.cross_entropy import CrossEntropyWithLogits


class LitMobileCLiP(pl.LightningModule):
    def __init__(self, config: Dict):

        super().__init__()

        self.cfg = config

        self.clip_model = MobileCLiP(config)
        self.criterion_per_img = CrossEntropyWithLogits()
        self.criterion_per_txt = CrossEntropyWithLogits()

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple:

        logits_per_img, logits_per_txt, targets = self.clip_model(image, text)

        return logits_per_img, logits_per_txt, targets

    def _compute_loss(self, yhat_img: torch.Tensor, yhat_txt: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        loss_per_img = self.criterion_per_img(yhat_img, targets)
        loss_per_txt = self.criterion_per_txt(yhat_txt, targets)

        loss = (loss_per_img + loss_per_txt) / 2

        return loss.mean()

    def _common_steps(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:

        img, txt = batch["img"], batch["txt"]["input_ids"].squeeze()

        logits_per_img, logits_per_txt, targets = self(img, txt)

        loss = self._compute_loss(logits_per_img, logits_per_txt, targets)

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> Dict:

        loss = self._common_steps(batch, batch_idx)
        self.log("train_loss", loss.item(), prog_bar=True, on_step=True,
                 on_epoch=True, rank_zero_only=True, logger=True)

        return {
            "loss": loss
        }

    def validation_step(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> Dict:

        loss = self._common_steps(batch, batch_idx)

        self.log("val_loss", loss.item(), prog_bar=True, on_step=False,
                 on_epoch=True, rank_zero_only=True, logger=True)

        return {
            "val_loss": loss
        }

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, self.cfg["T_0"], eta_min=self.cfg["min_lr"], verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
