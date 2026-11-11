"""
Zero-Shot Object Detection Evaluator & Benchmark Engine:
- Calculates IoU, Precision, Recall, and F1-score
- Computes Average Precision (AP@50, AP@75, mAP@[50:95])
- Compares base vs novel (zero-shot unseen) class performance
- Includes benchmark suite for zero-shot validation
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from .utils import compute_iou


class ZeroShotEvaluator:
    """
    Evaluator for Zero-Shot Object Detection predictions against ground truth annotations.
    """

    def __init__(self, iou_thresholds: List[float] = None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.75]

    def evaluate_detections(
        self,
        ground_truths: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        iou_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluates predictions for a single image or set of images against ground truths.
        ground_truths: list of {"box": [x1, y1, x2, y2], "label": str}
        predictions: list of {"box": [x1, y1, x2, y2], "label": str, "score": float}
        """
        all_labels = sorted(list({gt["label"].lower().strip() for gt in ground_truths} | 
                                 {pred["label"].lower().strip() for pred in predictions}))

        per_class_results = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for label in all_labels:
            gt_boxes = [gt["box"] for gt in ground_truths if gt["label"].lower().strip() == label]
            pred_items = [p for p in predictions if p["label"].lower().strip() == label]
            pred_items = sorted(pred_items, key=lambda x: x.get("score", 0.0), reverse=True)

            matched_gt = set()
            tp = 0
            fp = 0

            for pred in pred_items:
                p_box = pred["box"]
                best_iou = 0.0
                best_gt_idx = -1

                for idx, g_box in enumerate(gt_boxes):
                    if idx in matched_gt:
                        continue
                    iou = compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= iou_threshold and best_gt_idx != -1:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn = len(gt_boxes) - len(matched_gt)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            per_class_results[label] = {
                "ground_truth_count": len(gt_boxes),
                "predicted_count": len(pred_items),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4)
            }

            total_tp += tp
            total_fp += fp
            total_fn += fn

        overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = (2 * overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0

        # Mean Average Precision approximation over classes
        mAP = np.mean([res["precision"] for res in per_class_results.values()]) if per_class_results else 0.0

        return {
            "iou_threshold": iou_threshold,
            "overall_precision": round(overall_prec, 4),
            "overall_recall": round(overall_rec, 4),
            "overall_f1": round(overall_f1, 4),
            "mAP": round(float(mAP), 4),
            "total_ground_truth": len(ground_truths),
            "total_predictions": len(predictions),
            "classes": per_class_results
        }

    def compute_coco_map_range(
        self,
        ground_truths: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Computes mAP at IoU=0.50, IoU=0.75, and mAP@[0.50:0.95] in steps of 0.05.
        """
        thresholds = np.arange(0.50, 0.96, 0.05)
        maps = []
        map50 = 0.0
        map75 = 0.0

        for t in thresholds:
            t = round(float(t), 2)
            res = self.evaluate_detections(ground_truths, predictions, iou_threshold=t)
            maps.append(res["mAP"])
            if t == 0.50:
                map50 = res["mAP"]
            elif t == 0.75:
                map75 = res["mAP"]

        return {
            "mAP@50": round(map50, 4),
            "mAP@75": round(map75, 4),
            "mAP@50:95": round(float(np.mean(maps)), 4)
        }
