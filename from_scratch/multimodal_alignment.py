"""
Multimodal Visual-Semantic Alignment Head implemented from scratch:
- Computes pairwise cosine similarity matrix between visual region proposals
  and open-vocabulary text query embeddings.
- Applies temperature scaling and sigmoid probability calibration.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScratchVisualSemanticAligner(nn.Module):
    """
    Projects visual region features and text embeddings into a calibrated
    joint metric space and computes zero-shot class probabilities.
    """

    def __init__(self, embed_dim: int = 128, initial_temperature: float = 0.07):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable log-temperature parameter (bounded to prevent numerical instability)
        self.log_temp = nn.Parameter(torch.tensor(math.log(1.0 / initial_temperature)))
        # Learnable background logit bias
        self.bg_bias = nn.Parameter(torch.tensor(-1.5))

    def compute_similarity(
        self,
        visual_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        visual_embeddings: (num_boxes, embed_dim) L2-normalized
        text_embeddings: (num_queries, embed_dim) L2-normalized
        Returns: (num_boxes, num_queries) cosine similarity matrix in [-1.0, 1.0]
        """
        v_norm = F.normalize(visual_embeddings, p=2, dim=-1)
        t_norm = F.normalize(text_embeddings, p=2, dim=-1)
        # Matrix multiplication gives cosine similarity
        sim = torch.matmul(v_norm, t_norm.t())
        return sim

    def forward(
        self,
        visual_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates calibrated zero-shot probability scores for all box-query pairs.
        Returns: (num_boxes, num_queries) probability values in [0.0, 1.0]
        """
        cosine_sim = self.compute_similarity(visual_embeddings, text_embeddings)
        temp = torch.exp(torch.clamp(self.log_temp, max=math.log(100.0)))

        # Temperature-scaled logit with background rejection bias
        logits = cosine_sim * temp + self.bg_bias
        probs = torch.sigmoid(logits)
        return probs
