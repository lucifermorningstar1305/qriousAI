"""
@author: Adityam Ghosh
Date: 10-29-2023

"""
from typing import List, Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionWiseFFN(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.ffn1 = nn.Linear(in_features=embed_dim, out_features=2 * embed_dim)
        self.ffn2 = nn.Linear(in_features=2 * embed_dim, out_features=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp = self.ffn1(x)
        temp = F.relu(temp, inplace=True)
        out = self.ffn2(temp)

        return out


class TransformersEncoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: Optional[int] = 1,
        dropout_rate: Optional[float] = 0.2,
    ):
        super().__init__()

        self.multi_attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )

        self.ffn = PositionWiseFFN(embed_dim=input_dim)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        attn_x, _ = self.multi_attn(x, x, x, key_padding_mask=attn_mask)
        add_norm1 = self.layer_norm1(x + attn_x)

        add_norm1 = self.dropout(add_norm1)

        ffn_out = self.ffn(add_norm1)
        add_norm2 = self.layer_norm2(add_norm1 + ffn_out)

        return add_norm2


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_seq_length: int):
        super().__init__()

        self.embed_dim = embed_dim

        pe = torch.zeros(max_seq_length, embed_dim)

        for pos in range(max_seq_length):
            for i in range(0, embed_dim, 2):
                pe[pos, i] = math.sin(pos / math.pow(10_000, (2 * i / embed_dim)))
                pe[pos, i + 1] = math.cos(
                    pos / math.pow(10_000, (2 * (i + 1) / embed_dim))
                )

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * math.sqrt(self.embed_dim)
        seq_len = x.size(1)

        x += torch.autograd.Variable(self.pe[:, :seq_len, :], requires_grad=False)

        return x
