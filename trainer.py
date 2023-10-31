"""
@author: Adityam Ghosh
Date: 10-29-2023

"""
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import itertools

from models.mobile_clip import MobileCLiP
from losses.cross_entropy import CrossEntropyWithLogits


class LitMobileCLiP(pl.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()

        self.cfg = config

        self.clip_model = MobileCLiP(config)
        self.criterion_per_img = CrossEntropyWithLogits()
        self.criterion_per_txt = CrossEntropyWithLogits()

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple:
        logits_per_img, logits_per_txt, targets = self.clip_model(
            image, text, attn_mask
        )

        return logits_per_img, logits_per_txt, targets

    def _compute_loss(
        self, yhat_img: torch.Tensor, yhat_txt: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        loss_per_img = self.criterion_per_img(yhat_img, targets)
        loss_per_txt = self.criterion_per_txt(yhat_txt, targets.t())

        loss = (loss_per_img + loss_per_txt) / 2

        return loss.mean()

    def _common_steps(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        img, txt, attn_mask = (
            batch["img"],
            batch["txt"]["input_ids"].squeeze(),
            batch["txt"]["attention_mask"].squeeze(),
        )

        logits_per_img, logits_per_txt, targets = self(img, txt, attn_mask.float())

        loss = self._compute_loss(logits_per_img, logits_per_txt, targets)

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> Dict:
        loss = self._common_steps(batch, batch_idx)
        self.log(
            "train_loss",
            loss.item(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            rank_zero_only=True,
            logger=True,
        )

        return {"loss": loss}

    def validation_step(self, batch: torch.Tensor, batch_idx: torch.Tensor) -> Dict:
        loss = self._common_steps(batch, batch_idx)

        self.log(
            "val_loss",
            loss.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
            logger=True,
        )

        return {"val_loss": loss}

    def configure_optimizers(self) -> Any:
        params = [
            {
                "params": self.clip_model.img_model.parameters(),
                "lr": self.cfg["image_model"]["lr"],
            },
            {
                "params": self.clip_model.text_model.parameters(),
                "lr": self.cfg["text_model"]["lr"],
            },
            {
                "params": itertools.chain(
                    self.clip_model.img_projection.parameters(),
                    self.clip_model.text_projection.parameters(),
                ),
                "lr": self.cfg["projection_head_lr"],
                "weight_decay": self.cfg["projection_head_weight_decay"],
            },
        ]
        optimizer = torch.optim.Adam(params, weight_decay=0.0)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, self.cfg["T_0"], eta_min=self.cfg["min_lr"], verbose=True
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def encode_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_image(img_tensor)

    def encode_text(
        self, text_tensor: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.clip_model.encode_text(text_tensor, attn_mask=attn_mask)
