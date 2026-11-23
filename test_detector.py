"""
Test the trained YOLO model on test images.
Works even if your runs/ folder structure is messy.

Usage:
    python test_detector.py
    python test_detector.py --model <path_to_best.pt>
"""

import cv2
import json
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


# ✅ Detect best device (Mac GPU / CUDA / CPU)
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return 0
    else:
        return "cpu"


# ✅ Find ALL possible best.pt files (even in wrong folders)
def find_all_models():
    return list(Path(".").rglob("best.pt"))


# ✅ Get latest model automatically
def get_latest_model():
    models = find_all_models()
    if not models:
        return None

    # Sort by latest modified
    models = sorted(models, key=lambda x: x.stat().st_mtime, reverse=True)

    print("\n🔍 Available models:")
    for m in models[:5]:
        print(f"   - {m}")

    return str(models[0])


def test_model(model_path=None,
               images_dir="dataset/images/test",
               output_dir="test_results",
               conf=0.5):

    # ✅ Auto-detect model if not provided
    if model_path is None:
        model_path = get_latest_model()

    if model_path is None or not Path(model_path).exists():
        print("\n❌ ERROR: No trained model found!")
        print("👉 Run: python train.py")
        print("👉 Or pass manually: --model path/to/best.pt")
        return

    print(f"\n✅ Using model: {model_path}")

    device = get_device()
    print(f"✅ Device: {device}")

    model = YOLO(model_path)

    Path(output_dir).mkdir(exist_ok=True)

    image_files = (
        list(Path(images_dir).glob("*.jpg")) +
        list(Path(images_dir).glob("*.jpeg")) +
        list(Path(images_dir).glob("*.png"))
    )

    if not image_files:
        print(f"\n❌ No images found in: {images_dir}")
        return

    print(f"\n📸 Found {len(image_files)} images\n")

    results_log = []

    for img_path in image_files:
        results = model(str(img_path), conf=conf, device=device, verbose=False)

        for r in results:
            output_path = Path(output_dir) / img_path.name
            cv2.imwrite(str(output_path), r.plot())

            dets = []
            for b in r.boxes:
                dets.append({
                    "class": model.names[int(b.cls[0])],
                    "confidence": round(float(b.conf[0]), 4),
                    "bbox": [round(x, 1) for x in b.xyxy[0].tolist()]
                })

            mark = "✓" if dets else "✗"
            conf_str = f"({dets[0]['confidence']:.1%})" if dets else "(no detection)"

            print(f"{mark}  {img_path.name}  {conf_str}")

            results_log.append({
                "image": img_path.name,
                "detections": dets
            })

    # ✅ Save JSON results
    with open(Path(output_dir) / "results.json", "w") as f:
        json.dump(results_log, f, indent=2)

    detected = sum(1 for r in results_log if r["detections"])

    print("\n📊 Summary")
    print(f"Detected {detected}/{len(image_files)} images")
    print(f"Results saved in → {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=None, help="Path to best.pt")
    parser.add_argument("--images", default="dataset/images/test")
    parser.add_argument("--output", default="test_results")
    parser.add_argument("--conf", type=float, default=0.5)

    args = parser.parse_args()

    test_model(args.model, args.images, args.output, args.conf)