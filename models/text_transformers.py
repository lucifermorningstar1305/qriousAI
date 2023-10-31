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
        temp = F.relu(temp)
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

        # self.ffn = PositionWiseFFN(embed_dim=input_dim)
        # self.layer_norm1 = nn.LayerNorm(input_dim)
        # self.layer_norm2 = nn.LayerNorm(input_dim)
        # self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        attn_x, _ = self.multi_attn(
            x, x, x, key_padding_mask=attn_mask, need_weights=False
        )

        return attn_x
        # attn_x = self.dropout(attn_x)
        # add_norm1 = self.layer_norm1(x + attn_x)

        # ffn_out = self.ffn(add_norm1)
        # ffn_out = self.dropout(ffn_out)
        # add_norm2 = self.layer_norm2(add_norm1 + ffn_out)

        # return add_norm2


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_seq_length: int, dropout_rate: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(p=dropout_rate)

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_seq_length, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        x = x.transpose(0, 1)  # (B, T, C) -> (T, B, C)
        x = x + self.pe[:seq_len]
        x = x.transpose(0, 1)  # (T, B, C) -> (B, T, C)
        x = self.dropout(x)

        return x
