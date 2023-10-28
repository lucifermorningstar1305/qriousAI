"""
@author: Adityam Ghosh
Date: 10-14-2023

"""
from typing import List, Tuple, Dict, Any, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeperableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | Tuple, stride: int | Tuple, padding: int | Tuple):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels),

            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0),

            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MobileNetv1(nn.Module):
    def __init__(self, in_channels: int, alpha: Optional[float] = 1.):

        super().__init__()

        assert alpha > 0 and alpha <= 1., "Expected range of alpha to be in range (0, 1]"
        assert type(
            in_channels) is int, "Expected in_channels to be an integer value"

        new_dw = int(alpha * 32)

        self.model = nn.Sequential()

        self.model.add_module("first_layer", nn.Conv2d(
            in_channels=in_channels, out_channels=new_dw, kernel_size=3, stride=2, padding=1))

        self.model.add_module(
            "first_layer_bn", nn.BatchNorm2d(num_features=new_dw))

        self.model.add_module("first_layer_act", nn.ReLU(inplace=True))

        self.add_module("depthwise_sep_conv_1", DepthwiseSeperableConv(
            in_channels=new_dw, out_channels=new_dw*2, kernel_size=3, stride=1, padding=1))
        self.add_module("depthwise_sep_conv_2", DepthwiseSeperableConv(
            in_channels=new_dw*2, out_channels=new_dw*4, kernel_size=3, stride=2, padding=1))
        self.add_module("depthwise_sep_conv_3", DepthwiseSeperableConv(
            in_channels=new_dw*4, out_channels=new_dw*4, kernel_size=3, stride=1, padding=1))
        self.add_module("depthwise_sep_conv_4", DepthwiseSeperableConv(
            in_channels=new_dw*4, out_channels=new_dw*8, kernel_size=3, stride=2, padding=1))
        self.add_module("depthwise_sep_conv_5", DepthwiseSeperableConv(
            in_channels=new_dw*8, out_channels=new_dw*8, kernel_size=3, stride=1, padding=1))
        self.add_module("depthwise_sep_conv_6", DepthwiseSeperableConv(
            in_channels=new_dw*8, out_channels=new_dw*16, kernel_size=3, stride=2, padding=1))

        for i in range(5):
            self.add_module(f"depthwise_sep_conv_{i+7}", DepthwiseSeperableConv(
                in_channels=new_dw*16, out_channels=new_dw*16, kernel_size=3, stride=1, padding=1))

        self.add_module(f"depthwise_sep_conv_12", DepthwiseSeperableConv(
            in_channels=new_dw*16, out_channels=new_dw*32, kernel_size=3, stride=2, padding=1))
        self.add_module(f"depthwise_sep_conv_13", DepthwiseSeperableConv(
            in_channels=new_dw*32, out_channels=new_dw*32, kernel_size=3, stride=2, padding=1))

        self.add_module(f"avg_pool", nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.add_module(f"flatten", nn.Flatten())
        self.add_module(f"fc", nn.Linear(
            in_features=1024, out_features=512))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
