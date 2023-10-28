"""
@author: Adityam Ghosh
Date: 10-29-2023

"""
from typing import Any, Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyWithLogits(nn.Module):
    def __init__(self, reduction: Optional[str] = "none"):

        assert reduction in [
            "none", "mean", "sum"], f"Expected reduction to be either sum/mean/none. Found {reduction}"

        self.reduction = reduction

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        yhat_log_softmax = F.log_softmax(yhat, dim=-1)
        loss = (-y * yhat_log_softmax).sum(1)

        if self.reduction == "sum":
            return loss.sum()

        elif self.reduction == "mean":
            return loss.mean()

        else:
            return loss
