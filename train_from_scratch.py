"""
End-to-End Training Pipeline for Zero-Shot Object Detection from Scratch:
- Seen Classes (Base) used for training: ["car", "person", "dog", "chair"]
- Unseen Novel Classes for Zero-Shot Evaluation: ["laptop", "bicycle", "coffee mug", "potted plant"]
- Optimizes Cross-Modal Contrastive Loss + Box Regression Loss
"""

import os
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from PIL import Image

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from from_scratch.zero_shot_detector import ScratchZeroShotDetector
from from_scratch.loss import ZeroShotMultiTaskLoss
from zero_shot_detector.visualizer import DetectionVisualizer


def run_training_experiment(epochs: int = 15, save_path: str = "scratch_model.pth"):
    print("=" * 65)
    print("[*] ZERO-SHOT OBJECT DETECTOR: FROM-SCRATCH TRAINING EXPERIMENT")
    print("=" * 65)

    # 1. Zero-Shot Protocol Definition
    SEEN_CLASSES = ["car", "person", "dog", "chair"]
    UNSEEN_NOVEL_CLASSES = ["laptop", "coffee mug", "potted plant", "traffic light"]

    print(f"[-] Seen (Base) Training Classes: {SEEN_CLASSES}")
    print(f"[-] Unseen Novel Zero-Shot Classes: {UNSEEN_NOVEL_CLASSES}")
    print("    (Note: Model will NOT observe any bounding box or visual data of unseen classes)")
    print("-" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = ScratchZeroShotDetector(embed_dim=128, device=device)
    detector.train()

    criterion = ZeroShotMultiTaskLoss(lambda_box=0.8, temperature=0.07)
    optimizer = optim.Adam(detector.parameters(), lr=1e-3, weight_decay=1e-4)

    # 2. Synthetic seen training batch generation
    print("1. Synthesizing seen-class training batches...")
    num_samples = 40
    batch_data = []

    for i in range(num_samples):
        # Create canvas
        canvas = np.zeros((256, 256, 3), dtype=np.uint8) + 200
        cls_idx = i % len(SEEN_CLASSES)
        cls_name = SEEN_CLASSES[cls_idx]

        # Draw a synthetic colored box representing seen class
        x1 = np.random.randint(20, 100)
        y1 = np.random.randint(20, 100)
        w = np.random.randint(50, 110)
        h = np.random.randint(50, 110)
        x2 = min(255, x1 + w)
        y2 = min(255, y1 + h)

        color = [50 + (cls_idx * 50) % 200, 80, 180 - (cls_idx * 40) % 180]
        canvas[y1:y2, x1:x2] = color

        img_pil = Image.fromarray(canvas)
        batch_data.append({
            "image": img_pil,
            "target_box": [float(x1), float(y1), float(x2), float(y2)],
            "label_idx": cls_idx,
            "label_name": cls_name
        })

    # 3. Training Loop
    print(f"\n2. Training visual-semantic projection weights over {epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_total_loss = 0.0
        epoch_align_loss = 0.0
        epoch_box_loss = 0.0

        for item in batch_data:
            optimizer.zero_grad()

            # Encode text inside step for clean autograd graph
            seen_text_embeds = detector.text_encoder(SEEN_CLASSES)

            img_tensor, np_raw = detector.preprocess_image(item["image"])
            feat_map = detector.backbone.extract_feature_map(img_tensor)

            # Use target box with small jitter as proposal
            jitter = np.random.uniform(-4, 4, size=4)
            anchor = np.array(item["target_box"]) + jitter
            anchor_tensor = torch.from_numpy(np.array([anchor], dtype=np.float32)).to(device)

            # RoI visual features
            vis_embed = detector.backbone.extract_roi_embeddings(feat_map, anchor_tensor)
            # Refined boxes
            pred_boxes = detector.box_regressor(vis_embed, anchor_tensor)

            target_box_tensor = torch.from_numpy(np.array([item["target_box"]], dtype=np.float32)).to(device)
            target_label_tensor = torch.tensor([item["label_idx"]], dtype=torch.long, device=device)

            # Compute multi-task zero-shot loss
            loss, l_align, l_box = criterion(
                vis_embed,
                seen_text_embeds,
                pred_boxes,
                target_label_tensor,
                target_box_tensor
            )

            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            epoch_align_loss += l_align.item()
            epoch_box_loss += l_box.item()

        avg_loss = epoch_total_loss / len(batch_data)
        avg_align = epoch_align_loss / len(batch_data)
        avg_box = epoch_box_loss / len(batch_data)

        if epoch % 3 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} (Align: {avg_align:.4f}, Box L1: {avg_box:.4f})")

    elapsed = time.time() - start_time
    print(f"\n[OK] Training completed in {elapsed:.2f}s")

    # 4. Save model weights
    torch.save(detector.state_dict(), save_path)
    print(f"[OK] Model weights saved to: {save_path}")

    # 5. ZERO-SHOT INFERENCE EVALUATION ON UNSEEN CLASSES
    print("\n" + "=" * 65)
    print("[*] EVALUATING ZERO-SHOT DETECTION ON UNSEEN NOVEL CATEGORIES")
    print("=" * 65)
    print(f"Querying UNSEEN classes: {UNSEEN_NOVEL_CLASSES}")

    detector.eval()
    test_image_path = "samples/office_desk.jpg"
    test_queries = "laptop, coffee mug, computer mouse"

    detections = detector.detect(
        test_image_path,
        queries=test_queries,
        threshold=0.20,
        nms_iou_threshold=0.40
    )

    print(f"\n[OK] Successfully detected {len(detections)} instances of UNSEEN classes:")
    for idx, d in enumerate(detections[:5], 1):
        print(f"  {idx}. [{d['label']}] (conf: {d['score']:.2%}) -> box: {d['box']}")

    # Visualize
    visualizer = DetectionVisualizer()
    test_img = Image.open(test_image_path)
    annotated = visualizer.draw_detections(test_img, detections)
    annotated.save("scratch_detected_unseen.jpg")
    print("\n[OK] Saved Zero-Shot inference output to: scratch_detected_unseen.jpg")
    print("=" * 65)


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    run_training_experiment(epochs=epochs)
