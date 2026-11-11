"""
Master Zero-Shot Object Detector implemented from scratch.
Assembles:
1. Custom Convolutional Vision Backbone (Stage 1-4)
2. Anchor Proposal Generator & Edge Objectness Scoring
3. Custom RoI Pooling Layer
4. Custom Semantic Text Language Encoder
5. Multimodal Cosine Alignment Head
6. Class-Agnostic Box Regressor MLP
7. Non-Maximum Suppression (NMS)
"""

import os
from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .backbone import ScratchVisionBackbone
from .region_proposal import ScratchRegionProposalGenerator
from .semantic_encoder import ScratchSemanticEncoder
from .multimodal_alignment import ScratchVisualSemanticAligner
from .box_regressor import ScratchBoxRegressor
from zero_shot_detector.utils import apply_nms, load_image


class ScratchZeroShotDetector(nn.Module):
    """
    Complete Zero-Shot Object Detection Network built from ground-up fundamentals.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        proposal_top_k: int = 120,
        device: Optional[str] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Vision Feature Extractor
        self.backbone = ScratchVisionBackbone(visual_embed_dim=embed_dim)

        # 2. Region Proposal Generator
        self.proposal_generator = ScratchRegionProposalGenerator(top_k=proposal_top_k)

        # 3. Semantic Language Text Encoder
        self.text_encoder = ScratchSemanticEncoder(semantic_dim=embed_dim)

        # 4. Multimodal Projection & Alignment
        self.aligner = ScratchVisualSemanticAligner(embed_dim=embed_dim)

        # 5. Box Delta Regressor
        self.box_regressor = ScratchBoxRegressor(visual_dim=embed_dim)

        self.to(self.device)
        self.eval()

    def preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Converts PIL Image to normalized PyTorch tensor (1, 3, H, W).
        """
        w, h = image.size
        # Resize to multiple of 32 for clean pooling
        new_w = max(32, (w // 32) * 32)
        new_h = max(32, (h // 32) * 32)
        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        np_arr = np.array(resized).astype(np.float32)
        # Normalize with ImageNet standard mean & std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
        norm_arr = (np_arr - mean) / std

        # HWC -> CHW -> BCHW
        tensor = torch.from_numpy(norm_arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor, np_arr

    def parse_queries(self, queries: Union[str, List[str]]) -> List[str]:
        if isinstance(queries, str):
            tokens = []
            for part in queries.replace(";", ",").replace("\n", ",").split(","):
                part = part.strip()
                if part:
                    tokens.append(part)
            return tokens or ["object"]
        return [str(q).strip() for q in queries if str(q).strip()] or ["object"]

    @torch.no_grad()
    def detect(
        self,
        image: Union[Image.Image, str],
        queries: Union[str, List[str]],
        threshold: float = 0.20,
        nms_iou_threshold: Optional[float] = 0.40,
        max_detections: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Runs full end-to-end Zero-Shot Object Detection from scratch.
        """
        pil_img = load_image(image)
        orig_w, orig_h = pil_img.size

        # 1. Normalize image
        img_tensor, np_raw = self.preprocess_image(pil_img)
        cur_h, cur_w = np_raw.shape[:2]

        # 2. Extract multi-stage convolutional feature maps
        feature_map = self.backbone.extract_feature_map(img_tensor)

        # 3. Generate candidate region proposals from image gradients
        anchors_np = self.proposal_generator.propose(np_raw)
        if len(anchors_np) == 0:
            return []

        anchors_tensor = torch.from_numpy(anchors_np).to(self.device)

        # 4. RoI Pooling & visual feature embedding
        visual_embeds = self.backbone.extract_roi_embeddings(
            feature_map, anchors_tensor, spatial_scale=1.0 / 16.0
        )

        # 5. Encode textual queries into semantic vectors
        query_list = self.parse_queries(queries)
        text_embeds = self.text_encoder(query_list)

        # 6. Multimodal visual-semantic alignment & similarity scoring
        scores_matrix = self.aligner(visual_embeds, text_embeds)  # (N_boxes, N_queries)

        # 7. Box coordinate refinement
        refined_boxes = self.box_regressor(visual_embeds, anchors_tensor)

        # 8. Scale boxes back to original image dimensions
        scale_x = orig_w / float(cur_w)
        scale_y = orig_h / float(cur_h)

        detections = []
        scores_cpu = scores_matrix.cpu().numpy()
        boxes_cpu = refined_boxes.cpu().numpy()

        for box_idx in range(len(anchors_np)):
            box = boxes_cpu[box_idx]
            x1 = max(0.0, float(box[0] * scale_x))
            y1 = max(0.0, float(box[1] * scale_y))
            x2 = min(float(orig_w), float(box[2] * scale_x))
            y2 = min(float(orig_h), float(box[3] * scale_y))

            if x2 <= x1 + 5 or y2 <= y1 + 5:
                continue

            # Check matches across each query
            for q_idx, q_name in enumerate(query_list):
                score = float(scores_cpu[box_idx, q_idx])
                if score >= threshold:
                    detections.append({
                        "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        "score": round(score, 4),
                        "label": q_name
                    })

        # 9. Non-Maximum Suppression (NMS)
        if nms_iou_threshold is not None and nms_iou_threshold > 0:
            detections = apply_nms(detections, iou_threshold=nms_iou_threshold)

        return detections[:max_detections]
