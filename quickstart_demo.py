"""
Python SDK Quickstart Example for Zero-Shot Object Detector
Demonstrates how to use the zero_shot_detector library programmatically.
"""

from zero_shot_detector.model import ZeroShotObjectDetector
from zero_shot_detector.visualizer import DetectionVisualizer
from zero_shot_detector.utils import load_image, export_to_coco_format
import json

def run_demo():
    print("1. Initializing Zero-Shot Object Detector...")
    detector = ZeroShotObjectDetector(model_name_or_alias="owlvit-base")

    print("2. Loading sample input image...")
    image = load_image("samples/street_traffic.jpg")

    print("3. Querying open-vocabulary zero-shot prompts...")
    # Any arbitrary natural language categories can be queried!
    prompts = ["red car", "blue car", "traffic light", "building"]
    detections = detector.detect(
        image=image,
        queries=prompts,
        threshold=0.15,
        nms_iou_threshold=0.45
    )

    print(f"4. Detected {len(detections)} instances:")
    for d in detections:
        print(f"   - Label: {d['label']:<15} Conf: {d['score']:.2%}  Box: {d['box']}")

    print("5. Rendering annotations...")
    visualizer = DetectionVisualizer()
    annotated = visualizer.draw_detections(image, detections)
    annotated.save("demo_output.jpg")
    print("   Annotated image saved to 'demo_output.jpg'")

    print("6. Exporting to standard COCO JSON format...")
    coco_dict = export_to_coco_format(detections, image.width, image.height, "street_traffic.jpg")
    with open("demo_coco.json", "w") as f:
        json.dump(coco_dict, f, indent=2)
    print("   COCO annotations saved to 'demo_coco.json'")

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    run_demo()
