"""
Zero-Shot Object Detection Loss Functions implemented from scratch:
1. InfoNCE / Triplet Margin Ranking Loss for Visual-Semantic Contrastive Alignment
2. Smooth L1 (Huber) Loss for Bounding Box Coordinate Regression
3. Combined Multi-Task Zero-Shot Objective
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroShotContrastiveLoss(nn.Module):
    """
    Contrastive alignment loss pulling matching visual regions and text queries together
    while pushing non-matching semantic concepts apart.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        visual_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        target_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        visual_embeddings: (N, D) normalized region representations
        text_embeddings: (C, D) normalized query embeddings
        target_labels: (N,) ground-truth class indices in [0, C-1]
        """
        # (N, C) cosine similarities scaled by temperature
        sim_matrix = torch.matmul(visual_embeddings, text_embeddings.t()) / self.temperature
        loss = F.cross_entropy(sim_matrix, target_labels)
        return loss


class SmoothL1BoxLoss(nn.Module):
    """
    Smooth L1 (Huber) regression loss for bounding box offset regression.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, predicted_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        predicted_boxes: (N, 4) in [x1, y1, x2, y2]
        target_boxes: (N, 4) in [x1, y1, x2, y2]
        """
        diff = torch.abs(predicted_boxes - target_boxes)
        smooth_l1 = torch.where(diff < self.beta, 0.5 * (diff ** 2) / self.beta, diff - 0.5 * self.beta)
        return smooth_l1.sum(dim=-1).mean()


class ZeroShotMultiTaskLoss(nn.Module):
    """
    Joint multi-task objective:
    L_total = L_contrastive + lambda_box * L_box
    """

    def __init__(self, lambda_box: float = 1.0, temperature: float = 0.07):
        super().__init__()
        self.contrastive_loss = ZeroShotContrastiveLoss(temperature=temperature)
        self.box_loss = SmoothL1BoxLoss()
        self.lambda_box = lambda_box

    def forward(
        self,
        visual_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        l_align = self.contrastive_loss(visual_embeds, text_embeds, target_labels)
        l_box = self.box_loss(pred_boxes, target_boxes)
        total_loss = l_align + self.lambda_box * l_box
        return total_loss, l_align, l_box
