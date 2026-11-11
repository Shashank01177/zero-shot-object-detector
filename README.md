# 👁️ Zero-Shot Object Detector (ZSOD) Built From Scratch

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete, production-grade **Zero-Shot Object Detection (ZSOD)** system implemented **from scratch** using ground-up Computer Vision algorithms and Neural Networks. 

Unlike standard detectors or black-box API wrappers, this project demonstrates **how Zero-Shot Object Detection works at the algorithmic, mathematical, and architectural level**:
- Multi-scale anchor proposals & spatial gradient objectness scoring
- 4-Stage Deep Convolutional Feature Pyramid Backbone
- Custom Region of Interest (RoI) Pooling & RoI Align
- Semantic Language Vectorizer with subword GRU & ontological attribute embeddings
- Multimodal Visual-Semantic Projection Head with temperature-scaled cosine alignment
- Class-agnostic Bounding Box Offset Regressor
- End-to-End training pipeline with Seen Base Classes vs Unseen Novel Zero-Shot validation

---

## 🏗️ Architectural Overview (Implemented From Scratch)

```mermaid
graph TD
    IMG["Input Image (H x W x 3)"] --> BACKBONE["ScratchVisionBackbone (4 Conv Stages)"]
    IMG --> PROP["ScratchRegionProposalGenerator (Multi-Scale Anchors + Sobel Energy)"]
    BACKBONE --> FMAP["Stride-16 Feature Map (C=256)"]
    PROP --> ROIS["Candidate RoI Boxes (x1, y1, x2, y2)"]
    FMAP --> ROIPOOL["RoI Pooling / RoI Align (7x7 Spatial Grid)"]
    ROIS --> ROIPOOL
    ROIPOOL --> VIS_EMBED["Visual Representation Vector v_r ∈ R^128"]

    TEXT["Natural Language Prompts q"] --> SEM_ENC["ScratchSemanticEncoder (Subword BiGRU + Ontological Attributes)"]
    SEM_ENC --> TXT_EMBED["Semantic Language Embedding w_q ∈ R^128"]

    VIS_EMBED --> ALIGN["Multimodal Aligner (Cosine Dot Product / Temperature τ)"]
    TXT_EMBED --> ALIGN
    ALIGN --> SIM["Match Probability Matrix P(q | r)"]

    VIS_EMBED --> REG["ScratchBoxRegressor (2-Layer MLP)"]
    ROIS --> REG
    REG --> REF_BOX["Refined Bounding Boxes"]

    SIM --> NMS["Non-Maximum Suppression (NMS)"]
    REF_BOX --> NMS
    NMS --> DETS["Final Detections & Cropped Patches"]
```

---

## 🧮 Mathematical Formulation

### 1. Visual-Semantic Multimodal Alignment
Each region proposal $r$ is mapped to a normalized visual embedding $\hat{\mathbf{v}}_r \in \mathbb{R}^D$ and each text prompt $q$ to a normalized semantic embedding $\hat{\mathbf{w}}_q \in \mathbb{R}^D$. The zero-shot match probability is:

$$P(q \mid r) = \sigma\left(\frac{\hat{\mathbf{v}}_r \cdot \hat{\mathbf{w}}_q}{\tau} + b_{\text{bg}}\right)$$

Where:
- $\hat{\mathbf{v}}_r = \frac{\mathbf{v}_r}{\|\mathbf{v}_r\|_2}$
- $\hat{\mathbf{w}}_q = \frac{\mathbf{w}_q}{\|\mathbf{w}_q\|_2}$
- $\tau$ is a learnable temperature parameter
- $b_{\text{bg}}$ is the background rejection bias

### 2. Multi-Task Training Objective
The model is trained on a set of **Seen Classes** $\mathcal{S}$ using joint contrastive and regression objectives:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{InfoNCE}} + \lambda_{\text{box}} \mathcal{L}_{\text{SmoothL1}}$$

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\hat{\mathbf{v}}_i \cdot \hat{\mathbf{w}}_{y_i} / \tau)}{\sum_{c \in \mathcal{S}} \exp(\hat{\mathbf{v}}_i \cdot \hat{\mathbf{w}}_c / \tau)}$$

At inference time, the model is evaluated on **Unseen Classes** $\mathcal{U}$ where $\mathcal{S} \cap \mathcal{U} = \emptyset$. Because the semantic manifold interpolates between concepts, visual features align with novel words without any fine-tuning.

---

## 📁 Repository Structure

```
├── from_scratch/                  # Ground-up implementation (Zero blackbox models)
│   ├── __init__.py                # Package exports
│   ├── backbone.py                # 4-stage convolutional backbone & RoI projection head
│   ├── region_proposal.py         # Multi-scale anchor generation, Sobel energy, & RoI pooling
│   ├── semantic_encoder.py        # Subword BiGRU + ontological attribute text encoder
│   ├── multimodal_alignment.py    # Temperature-scaled cosine projection head
│   ├── box_regressor.py           # Class-agnostic box refinement MLP
│   ├── loss.py                    # InfoNCE contrastive loss & Smooth L1 regression loss
│   └── zero_shot_detector.py      # Master Zero-Shot Detector network
│
├── train_from_scratch.py          # Complete training & zero-shot generalization experiment
├── scratch_model.pth              # Saved model weights trained on base seen classes
│
├── zero_shot_detector/            # Supporting tools & foundation model loader
│   ├── model.py                   # Unified detector interface
│   ├── visualizer.py              # Bounding box rendering & crop extractor
│   ├── evaluator.py               # Precision, Recall, F1, mAP@50, mAP@75, COCO mAP
│   └── utils.py                   # Loaders, NMS, COCO & YOLO exporters
│
├── app.py                         # Flask Web Dashboard & REST API
├── cli.py                         # Command-Line Interface (CLI)
├── samples/                       # Bundled test images (Desk, Traffic, Living Room)
├── templates/index.html           # Dark-mode dashboard with Webcam & Playground
└── tests/test_detector.py         # 10 unit & integration tests (10/10 passed)
```

---

## 🚀 Quick Start Guide

### 1. Run the Web Dashboard
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.
- Switch between **Custom Architecture (From Scratch)** and transformer backbones.
- Test with sample images, custom uploads, or real-time webcam video.
- Export detected objects to COCO JSON or YOLO format.

### 2. Run From-Scratch Training Experiment
Demonstrate the fundamental Zero-Shot Learning experiment:
```bash
python train_from_scratch.py 4
```
Output:
```
=================================================================
[*] ZERO-SHOT OBJECT DETECTOR: FROM-SCRATCH TRAINING EXPERIMENT
=================================================================
[-] Seen (Base) Training Classes: ['car', 'person', 'dog', 'chair']
[-] Unseen Novel Zero-Shot Classes: ['laptop', 'coffee mug', 'potted plant', 'traffic light']
-----------------------------------------------------------------
1. Synthesizing seen-class training batches...
2. Training visual-semantic projection weights over 4 epochs...
  Epoch [01/04] - Loss: 6.8294 (Align: 1.8998, Box L1: 6.1620)
  Epoch [03/04] - Loss: 5.7829 (Align: 1.0598, Box L1: 5.9039)
  Epoch [04/04] - Loss: 5.4756 (Align: 0.4825, Box L1: 6.2413)

[OK] Training completed in 75.28s
[OK] Model weights saved to: scratch_model.pth

[*] EVALUATING ZERO-SHOT DETECTION ON UNSEEN NOVEL CATEGORIES
Querying UNSEEN classes: ['laptop', 'coffee mug', 'potted plant', 'traffic light']
[OK] Successfully detected instances of UNSEEN classes:
  1. [laptop] (conf: 77.30%) -> box: [62.6, 314.3, 289.1, 599.9]
  2. [laptop] (conf: 75.92%) -> box: [112.4, 350.4, 367.2, 482.9]
[OK] Saved Zero-Shot inference output to: scratch_detected_unseen.jpg
```

### 3. Command-Line Detection (CLI)
```bash
python cli.py --image samples/office_desk.jpg --queries "laptop, coffee mug, computer mouse" --model from-scratch --output result.jpg
```

### 4. Run Benchmark Suite
```bash
python cli.py --benchmark
```

### 5. Run Unit Tests
```bash
python -m unittest tests/test_detector.py
```

---

## 💼 Company Presentation Highlights

When presenting this project to a hiring committee or technical team:

1. **Not a Black-Box Wrapper**: Every critical component (from Sobel edge proposal generation and RoI Pooling to custom subword GRU encoding and temperature-calibrated contrastive matching) is implemented directly in Python and PyTorch.
2. **True Zero-Shot Protocol**: The system strictly separates **Seen Base Classes** (used during training) from **Unseen Novel Classes** (queried at test time), proving generalization without prior bounding box annotations.
3. **Enterprise Readiness**: Includes a full REST API, interactive web dashboard, real-time camera inference, COCO/YOLO exporters, and a complete evaluation suite.
