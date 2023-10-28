"""
@author: Adityam Ghosh
Date: 10-29-2023

"""
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.image_models import MobileNetv1
from models.text_models import LiteTransformerEncoder


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout_rate: float):

        super().__init__()

        self.dropout_rate = dropout_rate

        self.proj_layer = nn.Linear(
            in_features=embedding_dim, out_features=projection_dim)

        self.gelu = nn.GELU()

        self.fc = nn.Linear(in_features=projection_dim,
                            out_features=projection_dim)

        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        proj_out = self.proj_layer(x)
        out = self.gelu(proj_out)
        out = self.fc(out)
        out = F.dropout(out, p=self.dropout_rate)
        out = out + proj_out
        out = self.layer_norm(out)

        return out


class MobileCLiP(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        self.config = config
        self.img_model = MobileNetv1(
            in_channels=3, out_dim=config["image_model"]["output_dim"], alpha=config["image_model"]["alpha"])

        self.text_model = LiteTransformerEncoder(config["text_model"])

        self.img_projection = ProjectionHead(embedding_dim=config["image_model"]["output_dim"],
                                             projection_dim=config["clip_model"]["proj_dim"], dropout_rate=config["clip_model"]["dropout_rate"])

        self.text_projection = ProjectionHead(embedding_dim=config["text_model"]["output_dim"],
                                              projection_dim=config["clip_model"]["proj_dim"], dropout_rate=config["clip_model"]["dropout_rate"])

        self.tau = nn.Parameter(torch.ones(
            []) * np.log(1 / config["clip_model"]["tau"]))

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> Tuple:

        img_proj = self.encode_image(image)
        txt_proj = self.encode_text(text)

        logit_scale = self.tau.exp()
        logits_per_img = logit_scale * img_proj @ txt_proj.t()
        logits_per_txt = logits_per_img.t()

        img_similarities = logit_scale * img_proj @ img_proj.t()
        txt_similarities = logit_scale * txt_proj @ txt_proj.t()

        targets = F.softmax(
            logit_scale * (img_similarities + txt_similarities) / 2, dim=-1)

        return logits_per_img, logits_per_txt, targets

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:

        img_out = self.img_model(x)
        img_proj = self.img_projection(img_out)

        return img_proj

    def encode_text(self, x: torch.Tensor) -> torch.Tensor:

        txt_out = self.text_model(x)[:, -1, :]
        txt_proj = self.text_projection(txt_out)

        return txt_proj