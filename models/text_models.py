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


class LiteTransformerEncoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        self.embed_dim = config["embedding_dim"]
        dropout_rate = config["dropout_rate"]

        self.embedding = nn.Embedding(
            num_embeddings=config["num_embeddings"],
            embedding_dim=config["embedding_dim"],
        )

        self.n_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "trans_encoder": TransformersEncoderBlock(
                            input_dim=config["embedding_dim"] // 2,
                            num_heads=config["transformer_encoder"]["num_heads"],
                            dropout_rate=config["transformer_encoder"]["dropout_rate"],
                        ),
                        "lconv_block": LightWeightConvBlock(
                            input_dim=config["embedding_dim"] // 2,
                            num_heads=config["lconv"]["num_heads"],
                            kernel_size=config["lconv"]["kernel_size"][i],
                            padding=config["lconv"]["padding"][i]
                            if isinstance(config["lconv"]["padding"], list)
                            else config["lconv"]["padding"],
                            weight_softmax=config["lconv"]["weight_softmax"][i]
                            if isinstance(config["lconv"]["weight_softmax"], list)
                            else config["lconv"]["weight_softmax"],
                            dropout_rate=config["lconv"]["dropout_rate"][i]
                            if isinstance(config["lconv"]["dropout_rate"], list)
                            else config["lconv"]["dropout_rate"],
                        ),
                        "layer_norm1": nn.LayerNorm(config["embedding_dim"]),
                        "layer_norm2": nn.LayerNorm(config["embedding_dim"]),
                        "ffn": nn.Sequential(
                            nn.Linear(
                                in_features=config["embedding_dim"],
                                out_features=config["embedding_dim"] * 4,
                            ),
                            nn.ReLU(inplace=True),
                            nn.Linear(
                                in_features=config["embedding_dim"] * 4,
                                out_features=config["embedding_dim"],
                            ),
                        ),
                    }
                )
                for i in range(config["n_blocks"])
            ]
        )

        self.pos_encoding = PositionalEncoding(
            embed_dim=config["embedding_dim"] // 2,
            max_seq_length=config["max_seq_length"],
        )

        self.out_layer = nn.Linear(
            in_features=config["embedding_dim"], out_features=config["output_dim"]
        )

        self.dropout = nn.Dropout(p=dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

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
            x_left = x[:, :, : self.embed_dim // 2]
            x_right = x[:, :, self.embed_dim // 2 :]

            x_left_out = block["trans_encoder"](self.pos_encoding(x_left))
            x_right_out = block["lconv_block"](x_right)

            concat_x = torch.concat([x_left_out, x_right_out], dim=-1)

            add_norm1 = block["layer_norm1"](x + concat_x)
            add_norm1 = self.dropout(add_norm1)
            ffn_out = block["ffn"](add_norm1)
            add_norm2 = block["layer_norm2"](add_norm1 + ffn_out)

            x = add_norm2

        x = self.out_layer(x)

        return x
