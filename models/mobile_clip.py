"""
@author: Adityam Ghosh
Date: 10-29-2023

"""
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.image_models import ImageEncoder
from models.text_models import LiteTransformerEncoder, ConvBertEncoder
from models.clip_lite_models import MIProjection
import sys

torch.set_float32_matmul_precision("medium")


class MobileCLiP(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        self.config = config

        self.img_model = ImageEncoder(
            model_name=config["image_model"]["model_name"],
            in_channels=3,
            alpha=config["image_model"]["alpha"],
        )

        self.text_model = None
        if self.config["text_model_name"] != "convbert":
            self.text_model = LiteTransformerEncoder(config["text_model"])
        else:
            self.text_model = ConvBertEncoder(config["convbert_model"])

        self.img_projection = MIProjection(
            inp_dim=config["image_model"]["output_dim"],
            proj_dim=config["clip_model"]["proj_dim"],
        )
        self.text_projection = MIProjection(
            inp_dim=config["text_model"]["output_dim"]
            if config["text_model_name"] != "convbert"
            else config["convbert_model"]["output_dim"],
            proj_dim=config["clip_model"]["proj_dim"],
        )

        self.tau = nn.Parameter(
            torch.ones([]) * np.log(1 / config["clip_model"]["tau"])
        )

    def _common_steps(
        self, img_feats: torch.Tensor, txt_feats: torch.Tensor
    ) -> torch.Tensor:
        img_feats = F.normalize(img_feats, p=2, dim=-1)
        txt_feats = F.normalize(txt_feats, p=2, dim=-1)
        logits_scale = self.tau.exp()
        logits = torch.einsum("n d, n d -> n", img_feats, txt_feats) * logits_scale
        return logits

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        neg_image: Optional[torch.Tensor] = None,
        neg_text: Optional[torch.Tensor] = None,
        neg_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple:
        if neg_text is None:
            img_feats = self.img_model(image)
            img_proj = self.img_projection(img_feats)

            # eos_embedding_loc = self.get_eos_embedding_pos(attn_mask)
            txt_feats = self.text_model(text, attn_mask)
            # txt_feats = txt_feats[torch.arange(text.size(0)), eos_embedding_loc, :]
            txt_proj = self.text_projection(txt_feats)

            Ej = -F.softplus(-self._common_steps(img_proj, txt_proj)).mean()

            # Negative pairs
            txt_feats_prime = torch.cat(
                (txt_feats[1:], txt_feats[0].unsqueeze(0)), dim=0
            )

            txt_proj_prime = self.text_projection(txt_feats_prime)
            Em = F.softplus(self._common_steps(img_proj, txt_proj_prime)).mean()

        elif neg_text is not None:
            img_feats = self.img_model(image)
            neg_img_feats = self.img_model(neg_image)

            img_feats_all = torch.cat((img_feats, neg_img_feats), dim=0)

            img_proj = self.img_projection(img_feats_all)

            txt_feats = self.text_model(text, attn_mask)
            neg_txt_feats = self.text_model(neg_text, neg_attn_mask)

            txt_feat_all = torch.cat((txt_feats, neg_txt_feats), dim=0)
            txt_proj = self.text_projection(txt_feat_all)

            Ej = -F.softplus(-self._common_steps(img_proj, txt_proj)).mean()

            # Shuffle text features so that half batch does not have hard negatives
            txt_feats_prime = torch.cat(
                (txt_feats[1:], txt_feats[0].unsqueeze(0)), dim=0
            )
            neg_txt_feat_all = torch.cat((neg_txt_feats, txt_feats_prime), dim=0)
            neg_txt_proj = self.text_projection(neg_txt_feat_all)

            Em = F.softplus(self._common_steps(img_proj, neg_txt_proj)).mean()

        return {
            "Ej": Ej,
            "Em": Em,
            "img_feats": img_feats,
            "txt_feats": txt_feats,
            "img_proj": img_proj,
            "txt_proj": txt_proj,
        }

    def get_eos_embedding_pos(self, attn_mask: torch.Tensor) -> List:
        attn_pos = torch.where(attn_mask == 1)[1]
        diffs = torch.diff(attn_pos)
        reset_indices = torch.where(diffs < 0)[0]
        max_elements = attn_pos[reset_indices].tolist()
        if len(reset_indices) == 0 or reset_indices[-1] != len(attn_pos) - 1:
            max_elements.append(attn_pos[-1].item())

        return max_elements

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        img_out = self.img_model(x)
        img_proj = self.img_projection(img_out)

        return img_proj

    def encode_text(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # eos_embedding_loc = self.get_eos_embedding_pos(attn_mask)
        # txt_out = self.text_model(x, attn_mask)[
        #     torch.arange(x.size(0)), eos_embedding_loc, :
        # ]
        txt_out = self.text_model(x, attn_mask)
        txt_proj = self.text_projection(txt_out)

        return txt_proj


# class ProjectionHead(nn.Module):
#     def __init__(self, embedding_dim: int, projection_dim: int, dropout_rate: float):
#         super().__init__()

#         self.proj_layer = nn.Linear(
#             in_features=embedding_dim, out_features=projection_dim
#         )

#         self.gelu = nn.GELU()

#         self.fc = nn.Linear(in_features=projection_dim, out_features=projection_dim)

#         self.layer_norm = nn.LayerNorm(projection_dim)
#         self.dropout = nn.Dropout(p=dropout_rate)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         proj_out = self.proj_layer(x)
#         out = self.gelu(proj_out)
#         out = self.fc(out)
#         out = self.dropout(out)
#         out = out + proj_out
#         out = self.layer_norm(out)

#         return out
