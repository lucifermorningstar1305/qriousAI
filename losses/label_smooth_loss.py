"""
Source: https://discuss.pytorch.org/t/what-is-the-formula-for-cross-entropy-loss-with-label-smoothing/149848
Date: 10-14-2023

"""

from typing import Optional
import torch
import torch.nn as nn


class CrossEntropyLabelSmoothing(nn.Module):
    def __init__(self, n_classes: int, smoothing: Optional[float] = .1, reduction: Optional[str] = "mean"):
        super().__init__()

        assert reduction in [
            "mean", "sum"], "Expected reduction to be either sum/mean"

        eps = smoothing / n_classes
        self.negative = eps
        self.positive = (1 - smoothing) + eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred = pred.log_softmax(dim=-1)

        true_vals = torch.zeros_like(pred)
        true_vals.fill_(self.negative)

        true_vals.scatter_(
            dim=1, index=target.data.unsqueeze(1), src=self.positive)

        if self.reduction == "sum":
            return torch.sum(-true_vals * pred, dim=1)
        else:
            return torch.sum(-true_vals*pred, dim=1).mean()
