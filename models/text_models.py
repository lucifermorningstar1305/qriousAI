"""
@author: Adityam Ghosh
Date: 10-15-2023

"""
from typing import List, Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.text_conv_models import LightWeightConvBlock
from models.text_transformers import TransformersEncoderBlock, PositionalEncoding


class LiteTransformerBlock(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        self.embed_dim = config["embedding_dim"]

        self.embedding = nn.Embedding(
            num_embeddings=config["num_embeddings"], embedding_dim=config["embedding_dim"])

        module_dict = nn.ModuleDict({
            "trans_encoder": TransformersEncoderBlock(input_dim=config["embedding_dim"], num_heads=config["transformer_encoder"]["num_heads"],
                                                      dropout_rate=config["transformer_encoder"]["dropout_rate"]),

            "lconv_block": LightWeightConvBlock(input_dim=config["embedding_dim"], num_heads=config["lconv"]["num_heads"],
                                                kernel_size=config["lconv"]["kernel_size"], padding=config["lconv"]["padding"],
                                                weight_softmax=config["lconv"]["weight_softmax"], dropout_rate=config["lconv"]["dropout_rate"])
        })

        self.n_blocks = nn.ModuleList([
            module_dict for _ in range(config["n_blocks"])
        ])

        self.pos_encoding = PositionalEncoding(
            embed_dim=config["embedding_dim"], max_seq_length=config["max_seq_length"])

        self.ffn = nn.Linear(
            in_features=config["embedding_dim"], out_features=config["output_embedding_dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        An implementation of the LiteTransformerBlock Forward function

        Parameters:
        -----------
        :param x: a tensor of shape (B x T)

        Returns:
        --------
        a tensor of shape (B x T x C)
        """
        x = self.embedding(x)

        for block in self.n_blocks:

            x_left = x[:, :, :self.embed_dim]
            x_right = x[:, :, self.embed_dim:]

            x_left_out = block["trans_encoder"](x_left)
            x_right_out = block["lconv_block"](x_right)

            concat_x = torch.concat([x_left_out, x_right_out], dim=-1)

            x = concat_x

        x = self.ffn(x)

        return x
