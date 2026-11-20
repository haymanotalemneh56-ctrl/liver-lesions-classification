# Liver Lesion Classification & Detection on CT Images

Deep learning pipelines developed for automated hepatic lesion analysis using a clinical dataset of **3,142 abdominal CT images** (Wudassie Diagnostic Center). *(Note: Raw patient CT scans are excluded from this repository to protect patient privacy.)*

## 1. Multi-Class Liver Lesion Classification (`cnn_liver_lesion_classification.py`)
* **Task:** 3-class classification (`normal`, `benign`, `malignant`) on a 70% / 15% / 15% train/validation/test split using balanced class weights and two-stage fine-tuning.
* **Architectures Benchmarked:** `VGG19`, `ResNet50`, `DenseNet121`, `EfficientNetB3`, and `MobileNetV2`.
* **Top Performance:** `VGG19` achieved **96.68% test accuracy** and **0.97 macro-averaged precision**.

## 2. Hepatic Lesion Localization (`yolov11_liver_lesion_detection.py`)
* **Task:** Real-time detection and classification of `benign` and `malignant` liver lesions.
* **Architectures Benchmarked:** `YOLOv11n`, `YOLOv11s`, and `YOLOv11m`.
* **Top Performance:** `YOLOv11s` achieved **87.69% mAP50-95**.
