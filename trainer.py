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
from models.clip_lite_models import PriorDiscriminator

# from losses.cross_entropy import CrossEntropyWithLogits
from losses.jsd_info_max_loss import JSDInfoMaxLoss

torch.set_float32_matmul_precision("medium")


class LitMobileCLiP(pl.LightningModule):
    def __init__(self, config: Dict):
        super().__init__()

        self.cfg = config

        self.clip_model = MobileCLiP(config)
        self.criterion = JSDInfoMaxLoss()

        self.img_prior_d, self.txt_prior_d = None, None
        if self.cfg["image_model"]["prior"]:
            self.img_prior_d = PriorDiscriminator(
                inp_dim=self.cfg["image_model"]["output_dim"]
            )

        if self.cfg["text_model"]["prior"]:
            self.txt_prior_d = PriorDiscriminator(
                inp_dim=self.cfg["text_model"]["output_dim"]
            )

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        neg_image: Optional[torch.Tensor] = None,
        neg_text: Optional[torch.Tensor] = None,
        neg_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple:
        model_out = self.clip_model(
            image, text, attn_mask, neg_image, neg_text, neg_attn_mask
        )

        return model_out

    def _compute_loss(self, model_out: Dict) -> torch.Tensor:
        # loss_per_img = self.criterion_per_img(yhat_img, targets)
        # loss_per_txt = self.criterion_per_txt(yhat_txt, targets.t())

        IMG_PRIOR = None
        TXT_PRIOR = None

        if self.cfg["image_model"]["prior"]:
            img_prior = torch.rand_like(
                model_out["img_feats"], device=model_out["img_feats"].device
            )
            term_a = torch.log(self.img_prior_d(img_prior)).mean()
            term_b = torch.log(1 - self.img_prior_d(model_out["img_feats"])).mean()
            IMG_PRIOR = -term_a - term_b

        if self.cfg["text_model"]["prior"]:
            txt_prior = torch.rand_like(
                model_out["txt_feats"], device=model_out["txt_feats"].device
            )
            term_a = torch.log(self.txt_prior_d(txt_prior)).mean()
            term_b = torch.log(1 - self.txt_prior_d(model_out["txt_feats"])).mean()
            TXT_PRIOR = -term_a - term_b

        loss = self.criterion(
            model_out["Ej"], model_out["Em"], img_prior=IMG_PRIOR, txt_prior=TXT_PRIOR
        )
        return loss

    def _common_steps(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> torch.Tensor:
        # print(batch)
        img, txt, neg_img, neg_txt = (
            batch["img"],
            batch["txt"],
            batch["neg_img"],
            batch["neg_txt"],
        )

        txt_input_ids, txt_attn_mask = (
            txt["input_ids"].squeeze(),
            txt["attention_mask"].squeeze().float(),
        )
        neg_txt_input_ids, neg_txt_attn_mask = (
            neg_txt["input_ids"].squeeze(),
            neg_txt["attention_mask"].squeeze().float(),
        )

        out = self(
            img,
            txt_input_ids,
            txt_attn_mask,
            neg_image=neg_img,
            neg_text=neg_txt_input_ids,
            neg_attn_mask=neg_txt_attn_mask,
        )

        loss = self._compute_loss(out)

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

    def empty(self):
        return
        yield

    def configure_optimizers(self) -> Any:
        params = [
            {
                "params": self.clip_model.img_model.parameters(),
                "lr": self.cfg["image_model"]["lr"],
                "weight_decay": self.cfg["image_model"]["weight_decay"],
            },
            {
                "params": self.clip_model.text_model.parameters(),
                "lr": self.cfg["text_model"]["lr"],
                "weight_decay": self.cfg["text_model"]["weight_decay"],
            },
            {
                "params": itertools.chain(
                    self.clip_model.img_projection.parameters(),
                    self.clip_model.text_projection.parameters(),
                    self.img_prior_d.parameters()
                    if self.img_prior_d is not None
                    else self.empty(),
                    self.txt_prior_d.parameters()
                    if self.txt_prior_d is not None
                    else self.empty(),
                ),
                "lr": self.cfg["lr"],
                "weight_decay": self.cfg["weight_decay"],
            },
        ]
        optimizer = torch.optim.Adam(params=params, weight_decay=0.0)
        # optimizer = torch.optim.Adam(
        #     self.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["weight_decay"]
        # )

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
