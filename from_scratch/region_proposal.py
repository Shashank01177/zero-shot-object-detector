"""
Region Proposal Generator & RoI Pooling implemented from scratch:
- Multi-scale sliding anchor generation
- Objectness scoring using gradient energy / spatial variance
- RoI Pooling (Region of Interest feature map extraction)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any


def generate_anchors(
    image_width: int,
    image_height: int,
    scales: List[int] = [48, 96, 192, 320],
    ratios: List[float] = [0.5, 1.0, 2.0],
    stride: int = 32
) -> np.ndarray:
    """
    Generates multi-scale anchor boxes across the spatial grid.
    Returns: array of shape (N, 4) in format [xmin, ymin, xmax, ymax]
    """
    anchors = []
    grid_x = np.arange(stride // 2, image_width, stride)
    grid_y = np.arange(stride // 2, image_height, stride)

    for cy in grid_y:
        for cx in grid_x:
            for s in scales:
                for r in ratios:
                    # w * h = s^2, w / h = r => w = s * sqrt(r), h = s / sqrt(r)
                    w = s * np.sqrt(r)
                    h = s / np.sqrt(r)

                    xmin = max(0.0, cx - w / 2.0)
                    ymin = max(0.0, cy - h / 2.0)
                    xmax = min(float(image_width), cx + w / 2.0)
                    ymax = min(float(image_height), cy + h / 2.0)

                    if (xmax - xmin) >= 20 and (ymax - ymin) >= 20:
                        anchors.append([xmin, ymin, xmax, ymax])

    return np.array(anchors, dtype=np.float32)


def compute_objectness_scores(
    image_np: np.ndarray,
    anchors: np.ndarray
) -> np.ndarray:
    """
    Computes an objectness saliency score for each anchor candidate
    using gradient magnitude and local contrast.
    image_np: shape (H, W, 3) with values in [0, 255]
    """
    # Convert to grayscale
    gray = (0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]).astype(np.float32)

    # Sobel gradients
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2.0
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2.0
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Integral image of gradient magnitude for O(1) box summation
    integral_grad = np.pad(np.cumsum(np.cumsum(grad_mag, axis=0), axis=1), ((1, 0), (1, 0)))

    scores = []
    for box in anchors:
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, min(gray.shape[1] - 1, x1))
        y1 = max(0, min(gray.shape[0] - 1, y1))
        x2 = max(x1 + 1, min(gray.shape[1], x2))
        y2 = max(y1 + 1, min(gray.shape[0], y2))

        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            scores.append(0.0)
            continue

        grad_sum = (
            integral_grad[y2, x2]
            - integral_grad[y1, x2]
            - integral_grad[y2, x1]
            + integral_grad[y1, x1]
        )
        avg_density = grad_sum / np.sqrt(area)
        scores.append(float(avg_density))

    scores = np.array(scores, dtype=np.float32)
    # Normalize scores to [0, 1]
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
    return scores


def roi_pool(
    feature_map: torch.Tensor,
    boxes: torch.Tensor,
    output_size: Tuple[int, int] = (7, 7),
    spatial_scale: float = 1.0 / 16.0
) -> torch.Tensor:
    """
    Region of Interest (RoI) Pooling implemented with fast torchvision C++ kernel
    and fallback pure PyTorch loop.
    feature_map: (B, C, H, W)
    boxes: (N, 4) in [x1, y1, x2, y2] pixel coordinates
    Returns: (N, C, output_size[0], output_size[1]) pooled region features
    """
    try:
        from torchvision.ops import roi_align
        # roi_align expects boxes as list of tensors or tensor with batch index in first column
        device = feature_map.device
        if boxes.shape[1] == 4:
            batch_idx = torch.zeros((boxes.shape[0], 1), dtype=boxes.dtype, device=device)
            rois = torch.cat([batch_idx, boxes], dim=1)
        else:
            rois = boxes
        return roi_align(feature_map, rois, output_size=output_size, spatial_scale=spatial_scale)
    except Exception:
        pass

    B, C, H, W = feature_map.shape
    num_rois = boxes.shape[0]
    out_h, out_w = output_size

    # Scale boxes to feature map dimensions
    scaled_boxes = boxes * spatial_scale
    pooled = torch.zeros((num_rois, C, out_h, out_w), dtype=feature_map.dtype, device=feature_map.device)

    for i in range(num_rois):
        x1, y1, x2, y2 = scaled_boxes[i]
        
        # Clamp to feature boundaries
        x1 = torch.clamp(x1, 0, W - 1)
        y1 = torch.clamp(y1, 0, H - 1)
        x2 = torch.clamp(x2, x1 + 1e-3, W)
        y2 = torch.clamp(y2, y1 + 1e-3, H)

        roi_w = max(1e-3, float(x2 - x1))
        roi_h = max(1e-3, float(y2 - y1))

        bin_w = roi_w / out_w
        bin_h = roi_h / out_h

        for ph in range(out_h):
            for pw in range(out_w):
                hstart = int(torch.floor(y1 + ph * bin_h))
                hend = int(torch.ceil(y1 + (ph + 1) * bin_h))
                wstart = int(torch.floor(x1 + pw * bin_w))
                wend = int(torch.ceil(x1 + (pw + 1) * bin_w))

                hstart = min(max(hstart, 0), H)
                hend = min(max(hend, hstart + 1), H)
                wstart = min(max(wstart, 0), W)
                wend = min(max(wend, wstart + 1), W)

                sub_region = feature_map[0, :, hstart:hend, wstart:wend]
                if sub_region.numel() > 0:
                    pooled[i, :, ph, pw] = sub_region.amax(dim=(-2, -1))

    return pooled


class ScratchRegionProposalGenerator:
    """
    Candidate object proposal generator combining multi-scale anchors
    with edge energy objectness ranking.
    """
    def __init__(self, top_k: int = 150):
        self.top_k = top_k

    def propose(self, image_np: np.ndarray) -> np.ndarray:
        h, w = image_np.shape[:2]
        anchors = generate_anchors(w, h, scales=[48, 96, 180, 320], ratios=[0.5, 1.0, 2.0], stride=32)
        scores = compute_objectness_scores(image_np, anchors)

        # Select top-K most salient regions
        top_indices = np.argsort(scores)[::-1][:self.top_k]
        return anchors[top_indices]
