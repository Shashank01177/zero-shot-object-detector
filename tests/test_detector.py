"""
Comprehensive Unit & Integration Tests for Zero-Shot Object Detector.
"""

import os
import sys
import unittest
import json
from PIL import Image

from zero_shot_detector.utils import (
    load_image,
    image_to_base64,
    compute_iou,
    apply_nms,
    export_to_coco_format,
    export_to_yolo_format
)
from zero_shot_detector.visualizer import DetectionVisualizer
from zero_shot_detector.evaluator import ZeroShotEvaluator
from zero_shot_detector.model import ZeroShotObjectDetector
from app import app


class TestZeroShotObjectDetector(unittest.TestCase):

    def setUp(self):
        self.test_img = Image.new("RGB", (200, 200), color=(100, 150, 200))
        self.visualizer = DetectionVisualizer()
        self.evaluator = ZeroShotEvaluator()
        self.detector = ZeroShotObjectDetector(model_name_or_alias="owlvit-base", lazy_load=True)
        self.client = app.test_client()

    def test_compute_iou(self):
        # Identical boxes
        box_a = [10.0, 10.0, 50.0, 50.0]
        self.assertAlmostEqual(compute_iou(box_a, box_a), 1.0)

        # Disjoint boxes
        box_b = [100.0, 100.0, 150.0, 150.0]
        self.assertEqual(compute_iou(box_a, box_b), 0.0)

        # Partial overlap (50% intersection)
        # Box 1: [0, 0, 10, 10] -> area 100
        # Box 2: [5, 0, 15, 10] -> area 100
        # Intersection: [5, 0, 10, 10] -> area 50
        # Union: 100 + 100 - 50 = 150 -> IoU = 50/150 = 0.3333
        b1 = [0.0, 0.0, 10.0, 10.0]
        b2 = [5.0, 0.0, 15.0, 10.0]
        self.assertAlmostEqual(compute_iou(b1, b2), 50.0 / 150.0, places=3)

    def test_apply_nms(self):
        # Two highly overlapping boxes for same label
        detections = [
            {"box": [10.0, 10.0, 50.0, 50.0], "score": 0.95, "label": "dog"},
            {"box": [11.0, 11.0, 50.0, 50.0], "score": 0.85, "label": "dog"}, # Duplicate
            {"box": [100.0, 100.0, 150.0, 150.0], "score": 0.90, "label": "cat"}
        ]
        kept = apply_nms(detections, iou_threshold=0.5)
        self.assertEqual(len(kept), 2)
        labels = [d["label"] for d in kept]
        self.assertIn("dog", labels)
        self.assertIn("cat", labels)
        self.assertEqual(kept[0]["score"], 0.95)

    def test_visualizer_and_crops(self):
        detections = [
            {"box": [20.0, 20.0, 80.0, 80.0], "score": 0.92, "label": "cup"},
            {"box": [100.0, 100.0, 160.0, 160.0], "score": 0.88, "label": "book"}
        ]
        annotated = self.visualizer.draw_detections(self.test_img, detections)
        self.assertEqual(annotated.size, self.test_img.size)

        crops = self.visualizer.extract_crops(self.test_img, detections)
        self.assertEqual(len(crops), 2)
        self.assertEqual(crops[0]["label"], "cup")

    def test_coco_export_format(self):
        detections = [
            {"box": [10.0, 20.0, 50.0, 60.0], "score": 0.90, "label": "car"}
        ]
        coco = export_to_coco_format(detections, 100, 100, "test.jpg")
        self.assertIn("images", coco)
        self.assertIn("categories", coco)
        self.assertIn("annotations", coco)
        self.assertEqual(len(coco["annotations"]), 1)
        self.assertEqual(coco["categories"][0]["name"], "car")

    def test_yolo_export_format(self):
        detections = [
            {"box": [10.0, 20.0, 50.0, 60.0], "score": 0.90, "label": "car"}
        ]
        yolo_txt, label_map = export_to_yolo_format(detections, 100, 100)
        self.assertIn("car", label_map)
        parts = yolo_txt.split()
        self.assertEqual(len(parts), 5) # class_id xc yc w h

    def test_evaluator_metrics(self):
        gt = [
            {"box": [10, 10, 50, 50], "label": "car"}
        ]
        preds = [
            {"box": [12, 12, 48, 48], "label": "car", "score": 0.95}
        ]
        metrics = self.evaluator.evaluate_detections(gt, preds, iou_threshold=0.5)
        self.assertEqual(metrics["overall_precision"], 1.0)
        self.assertEqual(metrics["overall_recall"], 1.0)
        self.assertEqual(metrics["overall_f1"], 1.0)

    def test_detector_inference(self):
        dets = self.detector.detect(self.test_img, queries="bottle, cup, phone", threshold=0.1)
        self.assertIsInstance(dets, list)

    def test_api_status(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("model", data)
        self.assertIn("device", data)

    def test_api_samples(self):
        res = self.client.get("/api/samples")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("samples", data)

    def test_api_detect_base64(self):
        b64 = image_to_base64(self.test_img)
        res = self.client.post("/api/detect", json={
            "image": b64,
            "queries": "box, object",
            "threshold": 0.1,
            "nms": 0.4
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("annotated_image", data)
        self.assertIn("detections", data)


if __name__ == "__main__":
    unittest.main()
