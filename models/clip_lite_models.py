import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MIProjection(nn.Module):
    def __init__(self, inp_dim: int, proj_dim: int, bln: bool = True):
        super().__init__()

        self.inp_dim = inp_dim
        self.proj_dim = proj_dim
        self.feat_nonlinear = nn.Sequential(
            nn.Linear(in_features=inp_dim, out_features=proj_dim),
            nn.BatchNorm1d(num_features=proj_dim),
            nn.ReLU(),
            nn.Linear(in_features=proj_dim, out_features=proj_dim),
        )

        self.feat_shortcut = nn.Linear(in_features=inp_dim, out_features=proj_dim)
        self.feature_block_ln = nn.LayerNorm(proj_dim)

        self.bln = bln
        self.reset_parameters()

    def reset_parameters(self):
        eye_mask = torch.zeros(self.proj_dim, self.inp_dim).bool()

        for i in range(self.inp_dim):
            eye_mask[i, i] = 1

        self.feat_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.feat_shortcut.weight.data.masked_fill_(eye_mask, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feat_nonlinear(x) + self.feat_shortcut(x)
        if self.bln:
            h = self.feature_block_ln(h)

        return h


class PriorDiscriminator(nn.Module):
    def __init__(self, inp_dim: int):
        super().__init__()

        self.l0 = nn.Linear(in_features=inp_dim, out_features=1000)
        self.l1 = nn.Linear(in_features=1000, out_features=200)
        self.l2 = nn.Linear(in_features=200, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.l0(x))
        out = F.relu(self.l1(out))
        return torch.sigmoid(self.l2(out))
