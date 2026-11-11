"""
Zero-Shot Object Detector - From-Scratch Implementation
Algorithms, Neural Modules, and Visual-Semantic Projections implemented without blackbox APIs.
"""

from .zero_shot_detector import ScratchZeroShotDetector
from .region_proposal import ScratchRegionProposalGenerator, roi_pool
from .semantic_encoder import ScratchSemanticEncoder
from .multimodal_alignment import ScratchVisualSemanticAligner

__all__ = [
    "ScratchZeroShotDetector",
    "ScratchRegionProposalGenerator",
    "roi_pool",
    "ScratchSemanticEncoder",
    "ScratchVisualSemanticAligner",
]
