"""
Class-Agnostic Bounding Box Regressor implemented from scratch:
- Predicts bounding box refinement offsets (dx, dy, dw, dh)
  to snap anchor proposals precisely onto physical object boundaries.
"""

import torch
import torch.nn as nn


class ScratchBoxRegressor(nn.Module):
    """
    Lightweight 2-layer MLP predicting class-agnostic bounding box delta offsets.
    """

    def __init__(self, visual_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )

        # Initialize weights with small random normal and zero bias
        nn.init.normal_(self.mlp[-1].weight, std=0.001)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, visual_features: torch.Tensor, anchor_boxes: torch.Tensor) -> torch.Tensor:
        """
        visual_features: (num_boxes, visual_dim)
        anchor_boxes: (num_boxes, 4) in [x1, y1, x2, y2]
        Returns: refined_boxes: (num_boxes, 4) in [x1, y1, x2, y2]
        """
        deltas = self.mlp(visual_features)  # (N, 4) -> [tx, ty, tw, th]

        x1 = anchor_boxes[:, 0]
        y1 = anchor_boxes[:, 1]
        x2 = anchor_boxes[:, 2]
        y2 = anchor_boxes[:, 3]

        wa = torch.clamp(x2 - x1, min=1.0)
        ha = torch.clamp(y2 - y1, min=1.0)
        cxa = x1 + 0.5 * wa
        cya = y1 + 0.5 * ha

        tx = deltas[:, 0]
        ty = deltas[:, 1]
        tw = torch.clamp(deltas[:, 2], min=-2.0, max=2.0)
        th = torch.clamp(deltas[:, 3], min=-2.0, max=2.0)

        cx_pred = cxa + tx * wa * 0.1
        cy_pred = cya + ty * ha * 0.1
        w_pred = wa * torch.exp(tw * 0.1)
        h_pred = ha * torch.exp(th * 0.1)

        x1_pred = cx_pred - 0.5 * w_pred
        y1_pred = cy_pred - 0.5 * h_pred
        x2_pred = cx_pred + 0.5 * w_pred
        y2_pred = cy_pred + 0.5 * h_pred

        refined = torch.stack([x1_pred, y1_pred, x2_pred, y2_pred], dim=-1)
        return refined
