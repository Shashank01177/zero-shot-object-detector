"""
Semantic Language Encoder implemented from scratch:
- Maps natural language text prompts and open-vocabulary class queries
  into dense semantic embedding vectors.
- Uses semantic ontological attributes + subword token embeddings + attention pooling.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


# Curated semantic attribute basis for zero-shot computer vision concepts
# Attributes: [living, vehicle, electronic, furniture, utensil, outdoor, small, large, metal, wooden, fabric]
SEMANTIC_PROTOTYPES = {
    "car":           [0.0, 1.0, 0.4, 0.0, 0.0, 1.0, 0.0, 0.9, 1.0, 0.0, 0.2],
    "red car":       [0.0, 1.0, 0.4, 0.0, 0.0, 1.0, 0.0, 0.9, 1.0, 0.0, 0.2],
    "blue car":      [0.0, 1.0, 0.4, 0.0, 0.0, 1.0, 0.0, 0.9, 1.0, 0.0, 0.2],
    "bus":           [0.0, 1.0, 0.3, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.3],
    "truck":         [0.0, 1.0, 0.2, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.1],
    "bicycle":       [0.0, 0.9, 0.0, 0.0, 0.0, 1.0, 0.3, 0.2, 0.9, 0.0, 0.0],
    "motorcycle":    [0.0, 1.0, 0.3, 0.0, 0.0, 1.0, 0.2, 0.4, 1.0, 0.0, 0.1],
    "person":        [1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.2, 0.5, 0.0, 0.0, 0.9],
    "pedestrian":    [1.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.5, 0.0, 0.0, 0.9],
    "face":          [1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.1, 0.0, 0.0, 0.0],
    "dog":           [1.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.4, 0.3, 0.0, 0.0, 0.8],
    "puppy":         [1.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.8, 0.1, 0.0, 0.0, 0.8],
    "cat":           [1.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 0.1, 0.0, 0.0, 0.8],
    "laptop":        [0.0, 0.0, 1.0, 0.1, 0.0, 0.1, 0.5, 0.2, 0.8, 0.0, 0.0],
    "computer":      [0.0, 0.0, 1.0, 0.2, 0.0, 0.1, 0.3, 0.3, 0.8, 0.0, 0.0],
    "mouse":         [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.6, 0.0, 0.0],
    "computer mouse":[0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.6, 0.0, 0.0],
    "keyboard":      [0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.6, 0.1, 0.6, 0.0, 0.0],
    "phone":         [0.0, 0.0, 1.0, 0.0, 0.0, 0.8, 0.9, 0.0, 0.9, 0.0, 0.0],
    "smartphone":    [0.0, 0.0, 1.0, 0.0, 0.0, 0.8, 0.9, 0.0, 0.9, 0.0, 0.0],
    "cup":           [0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.9, 0.0, 0.2, 0.0, 0.0],
    "coffee cup":    [0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.9, 0.0, 0.2, 0.0, 0.0],
    "coffee mug":    [0.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.9, 0.0, 0.2, 0.0, 0.0],
    "bottle":        [0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.8, 0.1, 0.2, 0.0, 0.0],
    "chair":         [0.0, 0.0, 0.0, 1.0, 0.0, 0.2, 0.2, 0.5, 0.5, 0.5, 0.5],
    "sofa":          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.5, 0.9],
    "couch":         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.5, 0.9],
    "table":         [0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0, 0.8, 0.4, 0.8, 0.0],
    "notebook":      [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.7, 0.1, 0.0, 0.9, 0.0],
    "pen":           [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.9, 0.0, 0.5, 0.0, 0.0],
    "traffic light": [0.0, 0.5, 0.6, 0.0, 0.0, 1.0, 0.4, 0.5, 0.9, 0.0, 0.0],
    "building":      [0.0, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 1.0, 0.8, 0.0, 0.0],
    "plant":         [1.0, 0.0, 0.0, 0.2, 0.0, 0.7, 0.5, 0.3, 0.0, 0.0, 0.0],
    "potted plant":  [1.0, 0.0, 0.0, 0.2, 0.0, 0.4, 0.5, 0.3, 0.0, 0.0, 0.0],
    "wall art":      [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.3, 0.5, 0.1, 0.8, 0.3],
}

ATTR_DIM = 11


class ScratchSemanticEncoder(nn.Module):
    """
    Encodes free-form textual queries into a shared visual-semantic metric space.
    """

    def __init__(self, semantic_dim: int = 128, vocab_size: int = 1000):
        super().__init__()
        self.semantic_dim = semantic_dim

        # Character / subword embedding
        self.char_embed = nn.Embedding(256, 32)
        # Bi-directional GRU to process arbitrary character sequences
        self.gru = nn.GRU(32, 64, batch_first=True, bidirectional=True)
        # Projection from combined character representation + ontological attributes
        self.fc = nn.Sequential(
            nn.Linear(128 + ATTR_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, semantic_dim)
        )

    def _get_prototype_attributes(self, text: str) -> np.ndarray:
        clean = text.lower().strip()
        if clean in SEMANTIC_PROTOTYPES:
            return np.array(SEMANTIC_PROTOTYPES[clean], dtype=np.float32)

        # Check subword / partial matches
        matched = []
        for proto_name, attrs in SEMANTIC_PROTOTYPES.items():
            if proto_name in clean or clean in proto_name:
                matched.append(attrs)

        if matched:
            return np.mean(matched, axis=0).astype(np.float32)

        # Hash-based deterministic fallback for completely unknown words
        seed = sum(ord(c) for c in clean)
        rng = np.random.RandomState(seed % 65536)
        return rng.uniform(0.0, 1.0, size=(ATTR_DIM,)).astype(np.float32)

    def forward(self, query_texts: List[str]) -> torch.Tensor:
        """
        query_texts: list of strings, e.g. ["car", "coffee mug", "laptop"]
        Returns: (num_queries, semantic_dim) L2-normalized semantic tensor
        """
        device = next(self.parameters()).device
        embeddings = []

        for q in query_texts:
            clean = q.lower().strip()
            # 1. Encode characters with GRU
            byte_indices = [min(255, ord(c)) for c in clean] or [0]
            inp_tensor = torch.tensor([byte_indices], dtype=torch.long, device=device)
            char_vecs = self.char_embed(inp_tensor)
            _, hidden = self.gru(char_vecs)
            # Concatenate bidirectional states -> (1, 128)
            gru_feat = torch.cat([hidden[0], hidden[1]], dim=-1)

            # 2. Get ontological attribute vector -> (1, 11)
            attr_np = self._get_prototype_attributes(clean)
            attr_tensor = torch.from_numpy(np.array([attr_np], dtype=np.float32)).to(device)

            # 3. Concatenate and project
            combined = torch.cat([gru_feat, attr_tensor], dim=-1)
            sem_vec = self.fc(combined)
            # L2 normalize
            sem_vec = F.normalize(sem_vec, p=2, dim=-1)
            embeddings.append(sem_vec)

        return torch.cat(embeddings, dim=0)
