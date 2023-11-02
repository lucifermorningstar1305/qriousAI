import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Callable, Optional, List, Tuple, Any


class JSDInfoMaxLoss(nn.Module):
    def __init__(self, prior_wt: float = 0.1):
        super().__init__()
        self.prior_wt = prior_wt

    def forward(
        self,
        Ej: torch.Tensor,
        Em: torch.Tensor,
        img_prior: Optional[torch.Tensor] = None,
        txt_prior: Optional[torch.Tensor] = None,
    ):
        PRIOR = torch.tensor(0.0, device=Ej.device)

        if img_prior is not None:
            PRIOR += img_prior

        if txt_prior is not None:
            PRIOR += txt_prior

        CROSS_MODAL_LOSS = Em - Ej

        TOTAL_LOSS = ((1.0 - self.prior_wt) * CROSS_MODAL_LOSS) + (
            self.prior_wt * PRIOR
        )

        return TOTAL_LOSS
