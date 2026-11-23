"""
Train YOLOv8 to detect Saudi driving license cards.
Optimized for Mac Mini (Apple Silicon GPU using MPS)
Usage: python train.py
"""
from ultralytics import YOLO
import torch

def get_device():
    if torch.backends.mps.is_available():
        return "mps"   # Apple GPU
    elif torch.cuda.is_available():
        return 0       # NVIDIA GPU
    else:
        return "cpu"   # fallback

def train():
    device = get_device()

    if device == "mps":
        print("Device: Apple Silicon GPU (MPS)")
    elif device == 0:
        print("Device: NVIDIA GPU (CUDA)")
    else:
        print("Device: CPU (slow — consider using GPU)")

    model = YOLO("yolov8s.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=200,
        imgsz=640,
        batch=16,  # 🔥 Reduce batch size for Mac GPU stability
        patience=20,

        # Optimization
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,

        # Augmentation
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Logging
        project="runs/train",
        name="license_detector",
        save=True,
        save_period=10,

        # Mac optimization
        workers=4,   # 🔥 Reduce workers for macOS stability
        device=device,
        verbose=True,
    )

    metrics = model.val()

    print(f"\nmAP50:     {metrics.box.map50:.4f}  (target > 0.90)")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print("\nBest model: runs/train/license_detector/weights/best.pt")

    if metrics.box.map50 < 0.80:
        print("[WARNING] mAP50 < 0.80 — collect more images and retrain.")

if __name__ == "__main__":
    train()
