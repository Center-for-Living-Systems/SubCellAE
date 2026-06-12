"""
model.py
========
Binary classifier for adhesion screening built on any timm backbone.

Supports EfficientNet, ResNet, and Vision Transformer (ViT) backbones.
For ViT models, `img_size` is forwarded so timm interpolates the positional
embeddings to the correct resolution (e.g. 64×64 or 128×128).
"""

from __future__ import annotations

from typing import Optional

import timm
import torch
import torch.nn as nn

# Backbones whose timm constructor accepts an explicit `img_size` argument.
# CNN backbones (EfficientNet, ResNet) are spatially flexible and do NOT need it.
_VIT_PREFIXES = ("vit_", "deit_", "swin_", "beit_", "eva_", "mvitv2_")


def _is_vit(backbone: str) -> bool:
    return any(backbone.startswith(p) for p in _VIT_PREFIXES)


class ScreeningClassifier(nn.Module):
    """Binary classifier built on a timm backbone.

    Parameters
    ----------
    backbone : str
        timm model name (e.g. ``"efficientnet_b0"``, ``"resnet18"``,
        ``"vit_tiny_patch16_224"``).
    pretrained : bool
        Load ImageNet pretrained weights.
    dropout : float
        Dropout rate applied before the final linear layer.
    img_size : int, optional
        Input spatial resolution.  Only forwarded to ViT-family models so
        timm can interpolate positional embeddings to the target grid size.
        Ignored for CNN backbones (they are inherently size-agnostic).
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
        img_size: Optional[int] = None,
    ):
        super().__init__()
        self.backbone_name = backbone

        kwargs: dict = dict(pretrained=pretrained, num_classes=0)
        if img_size is not None and _is_vit(backbone):
            kwargs["img_size"] = img_size

        self.backbone = timm.create_model(backbone, **kwargs)
        feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logit (shape: [B]).  Use BCEWithLogitsLoss for training."""
        features = self.backbone(x)
        return self.head(features).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probability for class 1 (adhesion), shape [B]."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# Keep old name as an alias so existing code doesn't break
EfficientNetScreener = ScreeningClassifier
