# Sentinel – Disaster Intelligence System

Sentinel is an AI-driven disaster damage detection system that analyzes post-disaster satellite imagery and automatically identifies structural damage using deep learning segmentation.

The system converts raw satellite images into actionable damage heatmaps and coverage metrics to support faster disaster response and infrastructure assessment.

---

## 🚀 Problem Statement

After natural disasters such as hurricanes, floods, and earthquakes, emergency agencies receive massive volumes of satellite imagery.

Manual inspection is:
- Slow
- Resource-intensive
- Inconsistent
- Difficult to scale across large regions

There is a critical need for automated, scalable damage assessment that highlights high-risk zones quickly and reliably.

Sentinel addresses this challenge using AI-based image segmentation and heatmap analytics.

---

## 🧠 Solution Overview

Sentinel uses a fine-tuned YOLOv8 segmentation model trained on the xBD disaster dataset to:

- Detect damaged structures at pixel level
- Generate segmentation masks with confidence scores
- Compute damage coverage percentage per tile
- Produce confidence-weighted heatmaps
- Overlay heatmaps on original satellite images

The system transforms raw imagery into structured damage intelligence.

---

## 🏗️ System Architecture

### 1️⃣ Data Layer
- xBD disaster dataset
- Post-disaster satellite tiles (.tif)
- Ground truth damage masks

### 2️⃣ Processing Layer
- xBD to YOLO segmentation format conversion
- Image resizing and normalization

### 3️⃣ Model Layer
- YOLOv8 Segmentation (fine-tuned)
- PyTorch backend

### 4️⃣ Analytics & Visualization Layer
- Mask extraction
- Damage percentage calculator
- Confidence-weighted heatmap engine
- Overlay generation module

### 5️⃣ Output Layer
- Visual damage heatmap
- Damage coverage metric
- Actionable disaster insights

---

## 📊 Model Performance (Prototype)

Validation Results (Subset Training):

- Mask mAP@50: ~4.18%
- Box mAP@50: ~4.89%
- Mask Precision: ~16.7%
- Mask Recall: ~7.7%

These results represent an early-stage prototype trained on a limited dataset subset. Performance improves with larger backbone models, extended training, and larger datasets.

---

## 🛠️ Technology Stack

- Python
- PyTorch
- Ultralytics YOLOv8 (Segmentation)
- OpenCV
- NumPy
- xBD Disaster Dataset

Designed for scalable deployment on CPU and accelerator-based environments.

---

##🔮 **Future Scope**

-Sentinel is designed as a scalable disaster intelligence framework. Future enhancements include:
-Upgrading to larger segmentation backbones (YOLOv8s/l)
-Multi-temporal change detection (pre vs post-disaster comparison)
-Region-scale batch inference pipelines
-Real-time satellite stream processing
-Infrastructure severity classification (minor, major, destroyed)
-Cloud and distributed deployment
-Integration with disaster management dashboards
-Resource allocation and prioritization engines

---
