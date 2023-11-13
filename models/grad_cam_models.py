import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()

        model = model.model.mobile_model
        self.features_conv = model.features
        self.avg_pool = model.avgpool
        self.classifier = model.classifier

        self.gradients = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features_conv(x)
        h = feats.register_hook(self.activations_hook)

        pooled_feats = self.avg_pool(feats)
        bs = pooled_feats.size(0)

        pooled_feats = pooled_feats.view(bs, -1)

        out = self.classifier(pooled_feats)

        return out

    def activations_hook(self, grad: torch.Tensor) -> torch.Tensor:
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x: torch.Tensor) -> torch.Tensor:
        return self.features_conv(x)
