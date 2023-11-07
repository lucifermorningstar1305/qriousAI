"""
@author: Adityam Ghosh
Date: 10-29-2023

"""
from typing import Dict, List, Any, Tuple, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

torch.set_float32_matmul_precision("medium")


class LitCLIP(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-4, min_lr: float = 1e-8):
        super().__init__()

        self.model = model
        self.lr = lr
        self.min_lr = min_lr

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ) -> Tuple:
        model_out = self.model(
            input_ids=text["input_ids"],
            attention_mask=text["attention_mask"],
            pixel_values=image["pixel_values"],
            return_loss=True,
            return_dict=True,
        )

        return model_out

    def _common_steps(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        img, txt = batch["img"], batch["txt"]

        out = self(img, txt)

        return out["loss"]

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            1000,
            eta_min=self.min_lr,
            verbose=True,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def encode_image(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(pixel_values=img_tensor)

    def encode_text(
        self, text_tensor: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.model.get_text_features(
            input_ids=text_tensor, attention_mask=attn_mask
        )
