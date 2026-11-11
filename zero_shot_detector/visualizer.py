"""
Visualizer for Zero-Shot Object Detection:
- Bounding box rendering with stylish aesthetic badges
- High-contrast, color-coded class palettes
- Crop extraction for detected object gallery
- Heatmap / detection mask overlays
"""

from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Curated palette of modern vibrant colors (HEX)
DISTINCT_COLORS = [
    "#3B82F6",  # Blue
    "#10B981",  # Emerald
    "#F59E0B",  # Amber
    "#EC4899",  # Pink
    "#8B5CF6",  # Purple
    "#06B6D4",  # Cyan
    "#EF4444",  # Red
    "#84CC16",  # Lime
    "#F97316",  # Orange
    "#14B8A6",  # Teal
    "#6366F1",  # Indigo
    "#E11D48",  # Rose
]

def hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


class DetectionVisualizer:
    def __init__(self, color_palette: List[str] = None):
        self.palette = color_palette or DISTINCT_COLORS
        self.class_color_cache: Dict[str, Tuple[int, int, int]] = {}

    def get_color_for_label(self, label: str) -> Tuple[int, int, int]:
        clean_lbl = label.strip().lower()
        if clean_lbl not in self.class_color_cache:
            idx = len(self.class_color_cache) % len(self.palette)
            self.class_color_cache[clean_lbl] = hex_to_rgb(self.palette[idx])
        return self.class_color_cache[clean_lbl]

    def draw_detections(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]],
        box_thickness: int = 3,
        show_scores: bool = True,
        fill_alpha: float = 0.12
    ) -> Image.Image:
        """
        Draws bounding boxes and labels onto a PIL Image.
        Returns a copy of the image with annotations.
        """
        annotated = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        draw = ImageDraw.Draw(annotated)

        # Attempt to load a default or standard TTF font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", size=max(14, int(image.height * 0.022)))
        except Exception:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

        for det in detections:
            box = det["box"]  # [xmin, ymin, xmax, ymax]
            label = det.get("label", "object")
            score = det.get("score", 1.0)

            xmin, ymin, xmax, ymax = box
            # Ensure coordinates are within image boundaries
            xmin = max(0, min(image.width - 1, xmin))
            ymin = max(0, min(image.height - 1, ymin))
            xmax = max(0, min(image.width - 1, xmax))
            ymax = max(0, min(image.height - 1, ymax))

            if xmax <= xmin or ymax <= ymin:
                continue

            color_rgb = self.get_color_for_label(label)
            r, g, b = color_rgb

            # Draw translucent fill inside box
            if fill_alpha > 0:
                fill_color = (r, g, b, int(fill_alpha * 255))
                overlay_draw.rectangle([xmin, ymin, xmax, ymax], fill=fill_color)

            # Draw solid bounding box border
            for i in range(box_thickness):
                draw.rectangle(
                    [xmin - i, ymin - i, xmax + i, ymax + i],
                    outline=(r, g, b, 255)
                )

            # Prepare text badge
            if show_scores:
                text = f"{label} {int(score * 100)}%"
            else:
                text = f"{label}"

            # Calculate text dimensions
            if hasattr(draw, "textbbox") and font:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            else:
                text_w = len(text) * 8
                text_h = 14

            pad_x = 6
            pad_y = 4
            badge_h = text_h + pad_y * 2
            badge_w = text_w + pad_x * 2

            # Position badge above box, or inside if near top edge
            badge_ymin = ymin - badge_h
            if badge_ymin < 0:
                badge_ymin = ymin

            badge_ymax = badge_ymin + badge_h
            badge_xmax = min(image.width, xmin + badge_w)

            # Draw badge background
            draw.rectangle([xmin, badge_ymin, badge_xmax, badge_ymax], fill=(r, g, b, 240))
            
            # Draw badge text in crisp white
            draw.text((xmin + pad_x, badge_ymin + pad_y), text, fill=(255, 255, 255, 255), font=font)

        # Composite translucent overlay with solid annotations
        annotated = Image.alpha_composite(annotated, overlay)
        return annotated.convert("RGB")

    def extract_crops(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]],
        padding: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Extracts cropped image patches for each detection for close-up examination.
        """
        crops = []
        w, h = image.size

        for i, det in enumerate(detections):
            xmin, ymin, xmax, ymax = det["box"]
            pad_xmin = max(0, int(xmin - padding))
            pad_ymin = max(0, int(ymin - padding))
            pad_xmax = min(w, int(xmax + padding))
            pad_ymax = min(h, int(ymax + padding))

            if pad_xmax > pad_xmin and pad_ymax > pad_ymin:
                crop_img = image.crop((pad_xmin, pad_ymin, pad_xmax, pad_ymax))
                crops.append({
                    "index": i + 1,
                    "label": det.get("label", "object"),
                    "score": det.get("score", 1.0),
                    "box": det["box"],
                    "crop_image": crop_img,
                    "width": pad_xmax - pad_xmin,
                    "height": pad_ymax - pad_ymin
                })

        return crops
