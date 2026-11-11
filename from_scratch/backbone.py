"""
Vision Feature Extractor Backbone implemented from scratch in PyTorch.
Extracts hierarchical multi-scale feature maps from raw pixel values
and maps pooled RoI regions to dense visual representation vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .region_proposal import roi_pool


class ScratchVisionBackbone(nn.Module):
    """
    4-Stage Deep Convolutional Feature Pyramid Backbone.
    Input: (B, 3, H, W) normalized image tensor.
    Output: Feature map of stride 16, shape (B, 256, H/16, W/16).
    """

    def __init__(self, visual_embed_dim: int = 128):
        super().__init__()
        self.visual_embed_dim = visual_embed_dim

        # Stage 1: Stride 2
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Stage 2: Stride 4
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Stage 3: Stride 8
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Stage 4: Stride 16
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # RoI Projection Head (maps 7x7 RoI pooled patches into visual embedding space)
        self.roi_fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, visual_embed_dim)
        )

    def extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv stages."""
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return f4

    def extract_roi_embeddings(
        self,
        feature_map: torch.Tensor,
        boxes: torch.Tensor,
        spatial_scale: float = 1.0 / 16.0
    ) -> torch.Tensor:
        """
        Applies RoI pooling over the feature map and projects regions
        to normalized visual embedding vectors.
        feature_map: (1, 256, H/16, W/16)
        boxes: (N, 4) in [x1, y1, x2, y2]
        Returns: (N, visual_embed_dim)
        """
        # Pooled shape: (N, 256, 7, 7)
        pooled = roi_pool(feature_map, boxes, output_size=(7, 7), spatial_scale=spatial_scale)
        flattened = pooled.view(pooled.size(0), -1)
        embeddings = self.roi_fc(flattened)
        # L2-normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
