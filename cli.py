"""
Command-Line Interface (CLI) for Zero-Shot Object Detector
Run zero-shot detection on images, batch directories, or URLs directly from terminal.
"""

import os
import sys
import json
import argparse
from PIL import Image

# Ensure UTF-8 stdout on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from zero_shot_detector.model import ZeroShotObjectDetector
from zero_shot_detector.visualizer import DetectionVisualizer
from zero_shot_detector.utils import (
    load_image,
    export_to_coco_format,
    export_to_yolo_format
)
from zero_shot_detector.evaluator import ZeroShotEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Shot Object Detection with Vision-Language Models (OWLv2 / OWL-ViT)"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default="samples/office_desk.jpg",
        help="Path or URL to input image (default: samples/office_desk.jpg)"
    )
    parser.add_argument(
        "--queries", "-q",
        type=str,
        default="laptop, coffee mug, notebook, pen, computer mouse",
        help="Comma-separated open-vocabulary text queries"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.15,
        help="Confidence threshold [0.0 - 1.0] (default: 0.15)"
    )
    parser.add_argument(
        "--nms",
        type=float,
        default=0.45,
        help="IoU threshold for Non-Maximum Suppression (default: 0.45, 0 to disable)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="from-scratch",
        choices=["from-scratch", "owlvit-base", "owlv2-base", "owlvit-large"],
        help="Model architecture: 'from-scratch' (ground-up implementation) or vision-language transformer"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="annotated_output.jpg",
        help="Path to save annotated output image"
    )
    parser.add_argument(
        "--export",
        type=str,
        choices=["none", "json", "coco", "yolo"],
        default="json",
        help="Export detection annotations in selected format"
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save cropped patches of detected objects in 'crops/' directory"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run zero-shot evaluation benchmark test"
    )

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
        return

    print("=" * 60)
    print("[*] ZERO-SHOT OBJECT DETECTOR")
    print("=" * 60)
    print(f"[-] Input Image: {args.image}")
    print(f"[-] Text Queries: {args.queries}")
    print(f"[-] Threshold: {args.threshold} | NMS IoU: {args.nms}")
    print(f"[-] Model: {args.model}")
    print("-" * 60)

    # 1. Load image
    try:
        image = load_image(args.image)
        print(f"[OK] Loaded image: {image.width}x{image.height} px")
    except Exception as e:
        print(f"[ERROR] Failed to load image: {e}")
        sys.exit(1)

    # 2. Initialize detector
    detector = ZeroShotObjectDetector(model_name_or_alias=args.model)

    # 3. Detect
    nms_val = args.nms if args.nms > 0 else None
    detections = detector.detect(
        image=image,
        queries=args.queries,
        threshold=args.threshold,
        nms_iou_threshold=nms_val
    )

    print(f"\n[OK] Found {len(detections)} object(s):")
    for idx, det in enumerate(detections, 1):
        box = det["box"]
        print(f"  {idx}. [{det['label']}] (conf: {det['score']:.2%}) -> box: {box}")

    # 4. Render visualizer
    visualizer = DetectionVisualizer()
    annotated = visualizer.draw_detections(image, detections)
    annotated.save(args.output, quality=95)
    print(f"\n[OK] Saved annotated image to: {args.output}")

    # 5. Save crops if requested
    if args.save_crops and detections:
        os.makedirs("crops", exist_ok=True)
        crops = visualizer.extract_crops(image, detections)
        for c in crops:
            clean_name = c['label'].replace(' ', '_')
            crop_path = os.path.join("crops", f"crop_{c['index']}_{clean_name}.jpg")
            c["crop_image"].save(crop_path, quality=95)
        print(f"[OK] Saved {len(crops)} cropped object(s) to 'crops/' directory")

    # 6. Export annotations
    if args.export == "json":
        json_path = os.path.splitext(args.output)[0] + "_detections.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=2)
        print(f"[OK] Exported detections JSON to: {json_path}")
    elif args.export == "coco":
        coco_path = os.path.splitext(args.output)[0] + "_coco.json"
        coco_data = export_to_coco_format(detections, image.width, image.height, os.path.basename(args.image))
        with open(coco_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)
        print(f"[OK] Exported COCO format to: {coco_path}")
    elif args.export == "yolo":
        yolo_path = os.path.splitext(args.output)[0] + "_yolo.txt"
        yolo_txt, label_map = export_to_yolo_format(detections, image.width, image.height)
        with open(yolo_path, "w", encoding="utf-8") as f:
            f.write(yolo_txt)
        print(f"[OK] Exported YOLO format to: {yolo_path}")
        print(f"  Class IDs mapping: {label_map}")

    print("\n[OK] Zero-shot detection completed successfully.")


def run_benchmark():
    print("=" * 60)
    print("[*] ZERO-SHOT OBJECT DETECTION BENCHMARK SUITE")
    print("=" * 60)
    
    # Ground truth annotations for office desk sample
    ground_truth = [
        {"box": [250, 220, 550, 420], "label": "laptop"},
        {"box": [640, 280, 720, 360], "label": "coffee mug"},
        {"box": [80, 320, 200, 480], "label": "notebook"},
        {"box": [210, 340, 222, 460], "label": "pen"},
        {"box": [620, 400, 680, 480], "label": "computer mouse"}
    ]

    # Sample predictions to validate evaluator
    sample_preds = [
        {"box": [245, 218, 555, 425], "label": "laptop", "score": 0.94},
        {"box": [638, 282, 718, 358], "label": "coffee mug", "score": 0.88},
        {"box": [85, 325, 195, 475], "label": "notebook", "score": 0.82},
        {"box": [208, 338, 225, 462], "label": "pen", "score": 0.76},
        {"box": [615, 395, 685, 485], "label": "computer mouse", "score": 0.91},
        {"box": [10, 10, 50, 50], "label": "pen", "score": 0.22}  # False positive
    ]

    evaluator = ZeroShotEvaluator()
    metrics = evaluator.evaluate_detections(ground_truth, sample_preds, iou_threshold=0.5)
    coco_metrics = evaluator.compute_coco_map_range(ground_truth, sample_preds)

    print(f"\nGround Truth Objects: {metrics['total_ground_truth']}")
    print(f"Predictions Evaluated: {metrics['total_predictions']}")
    print(f"IoU Threshold: {metrics['iou_threshold']}")
    print(f"- Overall Precision: {metrics['overall_precision']:.2%}")
    print(f"- Overall Recall:    {metrics['overall_recall']:.2%}")
    print(f"- Overall F1-Score:  {metrics['overall_f1']:.2%}")
    print(f"- mAP@50:            {coco_metrics['mAP@50']:.2%}")
    print(f"- mAP@75:            {coco_metrics['mAP@75']:.2%}")
    print(f"- mAP@50:95:         {coco_metrics['mAP@50:95']:.2%}")

    print("\nPer-Class Breakdown:")
    for cls_name, cls_res in metrics["classes"].items():
        print(f"  - {cls_name.upper():<16}: Precision={cls_res['precision']:.2%}, Recall={cls_res['recall']:.2%}, F1={cls_res['f1_score']:.2%}")

    print("\n[OK] Benchmark execution complete.")


if __name__ == "__main__":
    main()
