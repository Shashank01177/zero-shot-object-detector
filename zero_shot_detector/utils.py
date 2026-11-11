"""
Utility functions for Zero-Shot Object Detector:
- Image loading (path, URL, PIL, bytes, base64)
- Bounding box calculations and conversions
- Non-Maximum Suppression (NMS)
- Export formats (COCO JSON, YOLO TXT, Pascal VOC XML)
"""

import io
import base64
import json
import urllib.request
from typing import List, Dict, Any, Union, Tuple
from PIL import Image
import numpy as np


def load_image(image_input: Union[str, bytes, Image.Image]) -> Image.Image:
    """
    Loads an image from various sources:
    - File path (str)
    - Base64 data URL (str)
    - Web URL (str starting with http:// or https://)
    - Raw bytes
    - PIL.Image.Image instance
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")

    if isinstance(image_input, str):
        # Base64 string
        if image_input.startswith("data:image"):
            header, encoded = image_input.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif len(image_input) > 200 and not image_input.startswith(("http://", "https://")):
            # Plain base64 string
            try:
                img_bytes = base64.b64decode(image_input)
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                pass

        # Web URL
        if image_input.startswith(("http://", "https://")):
            req = urllib.request.Request(image_input, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")

        # Local file path
        return Image.open(image_input).convert("RGB")

    raise ValueError(f"Unsupported image input type: {type(image_input)}")


def image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 90) -> str:
    """
    Converts a PIL Image to a base64 Data URL string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format, quality=quality)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Computes Intersection over Union (IoU) of two bounding boxes.
    Format: [xmin, ymin, xmax, ymax]
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    intersection_area = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    box_a_area = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    box_b_area = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - intersection_area

    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area


def apply_nms(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False
) -> List[Dict[str, Any]]:
    """
    Applies Non-Maximum Suppression (NMS) to eliminate duplicate/overlapping bounding boxes.
    detections is a list of dicts with:
    {
        "box": [xmin, ymin, xmax, ymax],
        "score": float,
        "label": str
    }
    """
    if not detections:
        return []

    # Sort descending by confidence score
    sorted_dets = sorted(detections, key=lambda d: d.get("score", 0.0), reverse=True)
    kept_dets: List[Dict[str, Any]] = []

    for det in sorted_dets:
        box = det["box"]
        label = det.get("label", "")
        should_keep = True

        for kept in kept_dets:
            if not class_agnostic and kept.get("label", "") != label:
                continue

            iou = compute_iou(box, kept["box"])
            if iou >= iou_threshold:
                should_keep = False
                break

        if should_keep:
            kept_dets.append(det)

    return kept_dets


def export_to_coco_format(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    image_name: str = "image.jpg"
) -> Dict[str, Any]:
    """
    Exports zero-shot detections into standardized COCO JSON format.
    """
    categories = {}
    cat_id_counter = 1

    coco_annotations = []
    for i, det in enumerate(detections):
        lbl = det.get("label", "unknown")
        if lbl not in categories:
            categories[lbl] = cat_id_counter
            cat_id_counter += 1

        xmin, ymin, xmax, ymax = det["box"]
        w = max(0.0, xmax - xmin)
        h = max(0.0, ymax - ymin)

        coco_annotations.append({
            "id": i + 1,
            "image_id": 1,
            "category_id": categories[lbl],
            "bbox": [round(xmin, 2), round(ymin, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2),
            "score": round(det.get("score", 1.0), 4),
            "iscrowd": 0
        })

    coco_dict = {
        "images": [{
            "id": 1,
            "file_name": image_name,
            "width": image_width,
            "height": image_height
        }],
        "categories": [
            {"id": cid, "name": name, "supercategory": "object"}
            for name, cid in categories.items()
        ],
        "annotations": coco_annotations
    }
    return coco_dict


def export_to_yolo_format(
    detections: List[Dict[str, Any]],
    image_width: int,
    image_height: int,
    label_map: Dict[str, int] = None
) -> Tuple[str, Dict[str, int]]:
    """
    Exports zero-shot detections into YOLO format lines:
    <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
    """
    if label_map is None:
        unique_labels = sorted(list({det.get("label", "object") for det in detections}))
        label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}

    lines = []
    for det in detections:
        xmin, ymin, xmax, ymax = det["box"]
        cid = label_map.get(det.get("label", ""), 0)

        # Normalize to [0, 1]
        xc = ((xmin + xmax) / 2.0) / image_width
        yc = ((ymin + ymax) / 2.0) / image_height
        bw = (xmax - xmin) / image_width
        bh = (ymax - ymin) / image_height

        lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return "\n".join(lines), label_map
