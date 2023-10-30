"""
@author: Adityam Ghosh
Date: 10-15-2023

"""
from typing import List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightConv(nn.Module):

    """Implementation of LightConv from: https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/lightweight_convolution.py"""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        kernel_size: int,
        padding: Any,
        weight_softmax: bool,
        bias: bool,
        dropout_rate: float,
    ):
        super().__init__()

        assert (
            input_dim % num_heads == 0
        ), f"Expected the number of heads to be divisible by the dimension of the model or the number of channels. Found input_dim={input_dim} and H={num_heads}"

        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.dropout = nn.Dropout(p=dropout_rate)

        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_dim))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

        if self.bias:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A implementation of the forward function for the LightConv Module

        Parameters:
        ----------
        :param x: tensor of size (B x T x C)

        Returns:
        --------
        tensor of size (B x T x C)
        """

        B, T, C = x.size()
        H = self.num_heads

        weight = self.weight

        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        weight = self.dropout(weight)

        x = x.view(-1, H, T)

        output = F.conv1d(x, weight=weight, padding=self.padding, groups=self.num_heads)

        output = output.view(B, T, C)

        if self.bias is not None:
            x += self.bias.view(1, -1, 1)

        return output


class LightWeightConvBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: Optional[int] = 1,
        kernel_size: Optional[int] = 1,
        padding: Optional[str | int] = "same",
        weight_softmax: Optional[bool] = True,
        bias: Optional[bool] = False,
        dropout_rate: Optional[float] = 0.2,
    ):
        super().__init__()

        self.proj = nn.Linear(in_features=input_dim, out_features=2 * input_dim)

        self.lconv = LightConv(
            input_dim=input_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            padding=padding,
            weight_softmax=weight_softmax,
            bias=bias,
            dropout_rate=dropout_rate,
        )

        self.fc = nn.Linear(in_features=input_dim, out_features=input_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        attn_mask = attn_mask.unsqueeze(2)
        if attn_mask is not None:
            x = x.masked_fill(attn_mask == 0, 0)
        x = F.glu(x, dim=-1)  # Converts the (B, T, 2*C) -> (B, T, C)
        x = self.lconv(x)
        x = self.fc(x)

        return x
