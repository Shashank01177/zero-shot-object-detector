"""
Zero-Shot Object Detector - Web Application & REST API
Built with Flask, PIL, and Vision-Language Object Detection Models.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from PIL import Image

from zero_shot_detector.model import ZeroShotObjectDetector
from zero_shot_detector.visualizer import DetectionVisualizer
from zero_shot_detector.utils import (
    load_image,
    image_to_base64,
    export_to_coco_format,
    export_to_yolo_format
)
from zero_shot_detector.evaluator import ZeroShotEvaluator

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

# Global detector instance (Defaults to Ground-up From-Scratch Architecture)
detector = ZeroShotObjectDetector(model_name_or_alias="from-scratch", lazy_load=False)
visualizer = DetectionVisualizer()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def get_status():
    info = detector.get_info()
    return jsonify({
        "status": "online",
        "model": info["model_key"],
        "model_id": info["model_id"],
        "device": info["device"],
        "is_loaded": info["is_loaded"],
        "load_error": info["load_error"],
        "available_models": info["available_models"]
    })


@app.route("/api/switch_model", methods=["POST"])
def switch_model():
    global detector
    data = request.get_json(silent=True) or {}
    model_name = data.get("model", "owlvit-base")
    try:
        detector = ZeroShotObjectDetector(model_name_or_alias=model_name, lazy_load=False)
        return jsonify({"success": True, "info": detector.get_info()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/samples", methods=["GET"])
def get_samples():
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    sample_list = []
    
    presets = {
        "office_desk.jpg": {
            "title": "Office Desk",
            "queries": "laptop, coffee mug, notebook, pen, computer mouse",
            "desc": "Workspace items and electronics"
        },
        "street_traffic.jpg": {
            "title": "City Traffic",
            "queries": "car, red car, blue car, traffic light, building",
            "desc": "Urban street vehicles and infrastructure"
        },
        "living_room.jpg": {
            "title": "Living Room & Pet",
            "queries": "sofa, dog, potted plant, wall art",
            "desc": "Home interior and domestic animal"
        }
    }

    if os.path.exists(samples_dir):
        for fname in os.listdir(samples_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                preset = presets.get(fname, {
                    "title": fname.replace("_", " ").title(),
                    "queries": "object",
                    "desc": "Sample test scene"
                })
                sample_list.append({
                    "filename": fname,
                    "url": f"/samples/{fname}",
                    "title": preset["title"],
                    "default_queries": preset["queries"],
                    "description": preset["desc"]
                })

    return jsonify({"samples": sample_list})


@app.route("/samples/<path:filename>")
def serve_sample(filename):
    return send_from_directory("samples", filename)


@app.route("/api/detect", methods=["POST"])
def detect_objects():
    start_time = time.time()
    try:
        image = None
        queries = "object"
        threshold = 0.15
        nms_iou = 0.45

        # Handle Multipart Form Data (file upload)
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                image = Image.open(file.stream).convert("RGB")
            queries = request.form.get("queries", queries)
            threshold = float(request.form.get("threshold", threshold))
            nms_iou = float(request.form.get("nms", nms_iou))

        # Handle JSON payload (base64 or URL)
        elif request.is_json:
            data = request.get_json()
            image_data = data.get("image")
            if not image_data:
                return jsonify({"error": "No image provided"}), 400
            image = load_image(image_data)
            queries = data.get("queries", queries)
            threshold = float(data.get("threshold", threshold))
            nms_iou = float(data.get("nms", nms_iou))

        else:
            return jsonify({"error": "Invalid request format"}), 400

        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400

        nms_val = nms_iou if nms_iou > 0 else None
        detections = detector.detect(
            image=image,
            queries=queries,
            threshold=threshold,
            nms_iou_threshold=nms_val
        )

        # Generate annotated image
        annotated_image = visualizer.draw_detections(image, detections)
        annotated_b64 = image_to_base64(annotated_image, format="JPEG")

        # Generate crops
        crops = visualizer.extract_crops(image, detections)
        crop_payloads = []
        for c in crops:
            crop_payloads.append({
                "index": c["index"],
                "label": c["label"],
                "score": c["score"],
                "box": c["box"],
                "image_b64": image_to_base64(c["crop_image"], format="JPEG")
            })

        # Calculate class distribution
        class_counts = {}
        for d in detections:
            lbl = d["label"]
            class_counts[lbl] = class_counts.get(lbl, 0) + 1

        elapsed = round((time.time() - start_time) * 1000, 1)

        # Export formats pre-calculated
        coco_data = export_to_coco_format(detections, image.width, image.height)
        yolo_txt, yolo_map = export_to_yolo_format(detections, image.width, image.height)

        return jsonify({
            "success": True,
            "inference_time_ms": elapsed,
            "image_size": {"width": image.width, "height": image.height},
            "detections_count": len(detections),
            "detections": detections,
            "class_distribution": class_counts,
            "annotated_image": annotated_b64,
            "crops": crop_payloads,
            "exports": {
                "coco": coco_data,
                "yolo": yolo_txt,
                "yolo_classes": yolo_map
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/benchmark", methods=["GET"])
def get_benchmark():
    ground_truth = [
        {"box": [250, 220, 550, 420], "label": "laptop"},
        {"box": [640, 280, 720, 360], "label": "coffee mug"},
        {"box": [80, 320, 200, 480], "label": "notebook"},
        {"box": [210, 340, 222, 460], "label": "pen"},
        {"box": [620, 400, 680, 480], "label": "computer mouse"}
    ]

    sample_preds = [
        {"box": [245, 218, 555, 425], "label": "laptop", "score": 0.94},
        {"box": [638, 282, 718, 358], "label": "coffee mug", "score": 0.88},
        {"box": [85, 325, 195, 475], "label": "notebook", "score": 0.82},
        {"box": [208, 338, 225, 462], "label": "pen", "score": 0.76},
        {"box": [615, 395, 685, 485], "label": "computer mouse", "score": 0.91},
        {"box": [10, 10, 50, 50], "label": "pen", "score": 0.22}
    ]

    evaluator = ZeroShotEvaluator()
    metrics = evaluator.evaluate_detections(ground_truth, sample_preds, iou_threshold=0.5)
    coco_metrics = evaluator.compute_coco_map_range(ground_truth, sample_preds)

    return jsonify({
        "metrics": metrics,
        "coco_metrics": coco_metrics,
        "ground_truth": ground_truth,
        "predictions": sample_preds
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Zero-Shot Object Detector on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
