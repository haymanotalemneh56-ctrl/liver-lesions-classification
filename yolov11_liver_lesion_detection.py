# =========================================================
# YOLOv11 MULTI-MODEL MEDICAL PIPELINE
# =========================================================

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ultralytics import YOLO

# =========================================================
# DATASET PATH
# =========================================================
ROOT = "/content/drive/MyDrive/yolov11_resplit"
data_yaml = "/content/data.yaml"


# =========================================================
# DATA CONFIG (2 CLASSES)
# =========================================================
data = {
    "train": f"{ROOT}/train/images",
    "val": f"{ROOT}/val/images",
    "test": f"{ROOT}/test/images",
    "nc": 2,
    "names": ["benign", "malignant"]
}

with open(data_yaml, "w") as f:
    yaml.dump(data, f)

print("Dataset ready")

# =========================================================
# MODELS
# =========================================================
models = {
    "YOLOv11n": "yolo11n.pt",
    "YOLOv11s": "yolo11s.pt",
    "YOLOv11m": "yolo11m.pt"
}

results_all = []

# =========================================================
# TRAIN + EVALUATION LOOP
# =========================================================
for name, weight in models.items():

    print("\n========================")
    print("TRAINING:", name)
    print("========================")

    model = YOLO(weight)

    # TRAIN
    model.train(
        data=data_yaml,
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,

        optimizer="AdamW",
        lr0=0.0002,
        weight_decay=0.0003,
        cos_lr=True,

        mosaic=0.1,
        mixup=0.0,
        copy_paste=0.0,

        hsv_h=0.01,
        hsv_s=0.15,
        hsv_v=0.15,

        degrees=2,
        translate=0.02,
        scale=0.15,

        workers=2,
        patience=25,
        close_mosaic=10,

        project="/content/runs",
        name=name
    )

    # LOAD BEST MODEL
    best_model = YOLO(f"/content/runs/{name}/weights/best.pt")

    # VALIDATION
    metrics = best_model.val(data=data_yaml)

    # =====================================================
    # NATIVE YOLO CONFUSION MATRIX
    # =====================================================
    cm_native = metrics.confusion_matrix.matrix

    plt.figure(figsize=(5,4))
    sns.heatmap(cm_native, annot=True, fmt=".2f", cmap="Greens")
    plt.title(f"{name} - Native Confusion Matrix")
    plt.show()

    # =====================================================
    # CLINICAL 3x3 MATRIX
    # =====================================================
    cm = cm_native

    cm_3x3 = np.zeros((3,3))
    cm_3x3[0,0] = cm[0,0]
    cm_3x3[0,1] = cm[0,1]
    cm_3x3[1,0] = cm[1,0]
    cm_3x3[1,1] = cm[1,1]

    normal = len(os.listdir(f"{ROOT}/test/images"))
    cm_3x3[2,2] = max(normal - cm.sum(), 1)

    cm_norm = cm_3x3 / (cm_3x3.sum(axis=1, keepdims=True) + 1e-6)

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["benign","malignant","normal"],
        yticklabels=["benign","malignant","normal"]
    )
    plt.title(f"{name} - Normalized Confusion Matrix")
    plt.show()

    # =====================================================
    # LOSS CURVES
    # =====================================================
    csv_path = f"/content/runs/{name}/results.csv"
    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df["train/box_loss"], label="box")
    plt.plot(df["train/cls_loss"], label="cls")
    plt.plot(df["train/dfl_loss"], label="dfl")
    plt.title(f"{name} Train Loss")
    plt.legend()
    plt.show()

    # =====================================================
    # STORE RESULTS
    # =====================================================
    results_all.append({
        "Model": name,
        "Precision": metrics.box.mp,
        "Recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map
    })

# =========================================================
# MODEL COMPARISON TABLE
# =========================================================
df = pd.DataFrame(results_all)

print("\n================ MODEL COMPARISON ================\n")
print(df)

# =========================================================
# MODEL COMPARISON PLOT (mAP50-95)
# =========================================================
plt.figure(figsize=(6,4))
plt.bar(df["Model"], df["mAP50-95"])
plt.title("YOLOv11 Model Comparison (mAP50-95)")
plt.ylabel("Score")
plt.show()
