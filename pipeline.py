"""
Full pipeline: YOLO detect → crop → pre-process → VLM OCR → post-process → validate
English‑only dates. No Hijri extraction or fallback.
"""

from __future__ import annotations

import asyncio
import cv2
import json
import logging
import os
import sys
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ultralytics import YOLO

from ocr_extractor import (
    Config as OCRConfig,
    ImageValidationError,
    check_together_health,
    async_check_together_health,
    clear_cache,
    extract_license_data,
    async_extract_license_data,
    extract_dob_corrective,
    async_extract_dob_corrective,
    validate_extracted_data,
    validate_image,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ocr_pipeline")

DEFAULT_MODEL_PATHS = [
    "runs/detect/run/train/license_detector/weights/best.pt",
    "runs/train/license_detector/weights/best.pt",
    "runs/detect/train/weights/best.pt",
    "best.pt",
]

def find_model() -> str | None:
    for p in DEFAULT_MODEL_PATHS:
        if os.path.exists(p):
            return p
    for root, _, files in os.walk("runs"):
        for f in files:
            if f == "best.pt":
                return os.path.join(root, f)
    return None

# ── Blur / quality gate ──────────────────────────────────────────────────────
# Reject crops whose Laplacian variance is below this threshold.
# Tune via env: lower = more permissive, higher = stricter.
BLUR_VARIANCE_THRESHOLD: float = float(os.getenv("BLUR_VARIANCE_THRESHOLD", "5.0"))


def compute_blur_score(image_path: str) -> float:
    """Return Laplacian variance of a grayscale image — higher = sharper."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def is_too_blurry(image_path: str, threshold: float | None = None) -> tuple[bool, float]:
    """(is_blurry, score). Compared against BLUR_VARIANCE_THRESHOLD by default."""
    score = compute_blur_score(image_path)
    th = BLUR_VARIANCE_THRESHOLD if threshold is None else threshold
    return (score < th, score)


_yolo_lock = threading.Lock()


class LicenseAuthPipeline:
    def __init__(self, model_path: str | None = None):
        if model_path is None:
            model_path = find_model()
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found. Searched: {DEFAULT_MODEL_PATHS}")
        print(f"[Model] Loaded from: {model_path}")
        self.detector = YOLO(model_path)
        self.confidence_threshold = 0.70

    def detect_license(self, image_path: str) -> dict:
        with _yolo_lock:
            results = self.detector(image_path, conf=self.confidence_threshold, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": self.detector.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].cpu().numpy().astype(int).tolist()
                })
        if not detections:
            return {"detected": False, "reason": "No license card detected."}
        license_detections = [d for d in detections if d["class"] != "not_license"]
        if not license_detections:
            return {"detected": False, "reason": "This does not appear to be a Saudi driving license."}
        best = max(license_detections, key=lambda x: x["confidence"])
        return {"detected": True, **best}

    def crop_license(self, image_path: str, bbox: list, padding: int = 15) -> str:
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
        cropped = img[y1:y2, x1:x2]
        out = str(Path(image_path).parent / (Path(image_path).stem + "_cropped.jpg"))
        cv2.imwrite(out, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return out

    def process(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            return {"success": False, "stage": "input", "message": f"File not found: {image_path}"}
        print(f"\nProcessing: {image_path}")
        print("─" * 45)
        print("Stage 1: Detecting license card (YOLO)...")
        det = self.detect_license(image_path)
        if not det["detected"]:
            print(f"  ✗  {det['reason']}")
            return {"success": False, "stage": "detection", "message": det["reason"]}
        print(f"  ✓  Detected ({det['confidence']:.1%} confidence)")
        print("Stage 2: Cropping license region...")
        cropped = self.crop_license(image_path, det["bbox"])
        print(f"  ✓  Cropped → {cropped}")
        print("Stage 3: Extracting text (VLM)...")
        ocr = extract_license_data(cropped)
        if not ocr["success"]:
            print(f"  ✗  OCR failed: {ocr['error']}")
            return {"success": False, "stage": "ocr", "message": ocr["error"]}
        print("  ✓  Text extracted")
        print("Stage 4: Validating fields...")
        validation = validate_extracted_data(ocr["data"])
        if validation["valid"]:
            print("  ✓  All fields valid")
        else:
            for issue in validation["issues"]:
                print(f"  ⚠  {issue}")
        print("─" * 45)
        return {
            "success": True,
            "detection_confidence": det["confidence"],
            "license_data": ocr["data"],
            "validation": validation,
            "cropped_image_path": cropped
        }

def _envbool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")

class PipelineConfig:
    MAX_WORKERS:       int  = int(os.getenv("PIPELINE_WORKERS",   "2"))
    OUTPUT_DIR:        str  = os.getenv("PIPELINE_OUTPUT_DIR",    "pipeline_output")
    OUTPUT_FORMAT:     str  = os.getenv("PIPELINE_OUTPUT_FORMAT", "json")
    SAVE_FAILED:       bool = _envbool("PIPELINE_SAVE_FAILED",   True)
    ENABLE_PREPROCESS: bool = _envbool("PIPELINE_PREPROCESS",    True)
    TARGET_WIDTH:      int  = int(os.getenv("PIPELINE_TARGET_W", "1600"))
    MIN_WIDTH:         int  = int(os.getenv("PIPELINE_MIN_W",    "1100"))
    SHARPEN:           bool = _envbool("PIPELINE_SHARPEN",       True)
    REDACT_NAME:       bool = _envbool("PIPELINE_REDACT_NAME",   False)
    NORMALISE_DATES:   bool = _envbool("PIPELINE_NORM_DATES",    True)
    FAIL_FAST:         bool = _envbool("PIPELINE_FAIL_FAST",     False)
    CLEAR_CACHE_START: bool = _envbool("PIPELINE_CLEAR_CACHE",   False)

    # ── Individual stage toggles ──────────────────────────────────────────────
    PP_EXIF_ORIENT:        bool = _envbool("PP_EXIF_ORIENT",        True)
    PP_WHITE_BALANCE:      bool = _envbool("PP_WHITE_BALANCE",      True)
    PP_SHADOW_REMOVE:      bool = _envbool("PP_SHADOW_REMOVE",      True)
    PP_CONTRAST_STRETCH:   bool = _envbool("PP_CONTRAST_STRETCH",   True)
    PP_MOIRE_REDUCE:       bool = _envbool("PP_MOIRE_REDUCE",       True)
    PP_GLARE_INPAINT:      bool = _envbool("PP_GLARE_INPAINT",      True)
    PP_CLAHE:              bool = _envbool("PP_CLAHE",              True)
    PP_AUTO_GAMMA:         bool = _envbool("PP_AUTO_GAMMA",         True)
    PP_DENOISE:            bool = _envbool("PP_DENOISE",            True)
    PP_DOC_WARP:           bool = _envbool("PP_DOC_WARP",           True)
    PP_FINE_DESKEW:        bool = _envbool("PP_FINE_DESKEW",        True)
    PP_MULTISCALE_SHARPEN: bool = _envbool("PP_MULTISCALE_SHARPEN", True)
    PP_TEXT_AWARE_SHARP:   bool = _envbool("PP_TEXT_AWARE_SHARP",   True)
    PP_QUALITY_GUARD:      bool = _envbool("PP_QUALITY_GUARD",      True)
    PP_QUALITY_LOG:        bool = _envbool("PP_QUALITY_LOG",        True)

    PP_MAX_DESKEW_DEG:     float = float(os.getenv("PP_MAX_DESKEW_DEG", "8.0"))
    PP_JPEG_QUALITY:       int   = int(os.getenv("PP_JPEG_QUALITY",    "96"))

pcfg = PipelineConfig()

@dataclass
class PipelineResult:
    image_path:           str
    success:              bool
    data:                 Optional[dict] = None
    validation:           Optional[dict] = None
    error:                Optional[str]  = None
    stage:                Optional[str]  = None
    detection_confidence: float          = 0.0
    cropped_image_path:   Optional[str]  = None
    duration_ms:          float          = 0.0
    preprocessed:         bool           = False
    timestamp:            str            = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    def to_dict(self) -> dict: return asdict(self)

@dataclass
class PipelineMetrics:
    total:     int   = 0
    succeeded: int   = 0
    failed:    int   = 0
    skipped:   int   = 0
    total_ms:  float = 0.0
    @property
    def success_rate(self) -> float: return (self.succeeded / self.total * 100) if self.total else 0.0
    @property
    def avg_ms(self) -> float: return (self.total_ms / self.succeeded) if self.succeeded else 0.0
    def summary(self) -> dict: return {"total": self.total, "succeeded": self.succeeded, "failed": self.failed, "success_rate": f"{self.success_rate:.1f}%", "avg_ms": f"{self.avg_ms:.0f}", "total_ms": f"{self.total_ms:.0f}"}

import numpy as np

# ── EXIF orientation (PIL fallback) ──────────────────────────────────────────
def _apply_exif_orientation(bgr: "np.ndarray", image_path: str) -> "np.ndarray":
    try:
        from PIL import Image, ExifTags
        with Image.open(image_path) as pim:
            exif = pim.getexif() if hasattr(pim, "getexif") else None
            if not exif:
                return bgr
            orient_key = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)
            if orient_key is None:
                return bgr
            o = exif.get(orient_key, 1)
        if o == 3:
            return cv2.rotate(bgr, cv2.ROTATE_180)
        if o == 6:
            return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        if o == 8:
            return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return bgr
    except Exception:
        return bgr

# ── Quality metrics ──────────────────────────────────────────────────────────
# ── Blur-aware recovery ─────────────────────────────────────────────────────
# Trigger threshold (Laplacian variance). Below this the crop is treated as
# heavily blurred and the recovery stack kicks in.
BLUR_RECOVERY_THRESHOLD: float = float(os.getenv("BLUR_RECOVERY_THRESHOLD", "150.0"))
# 2x upscale when below this stricter threshold (very blurred -> super-res-style boost).
BLUR_UPSCALE_THRESHOLD:  float = float(os.getenv("BLUR_UPSCALE_THRESHOLD",  "80.0"))


def _wiener_deblur(bgr: "np.ndarray", psf_size: int = 3, k: float = 0.012) -> "np.ndarray":
    """Lightweight frequency-domain Wiener deconvolution per channel.
    Assumes a small Gaussian point-spread function — safe for mild-to-moderate blur.
    """
    psf = cv2.getGaussianKernel(psf_size, 0)
    psf = (psf @ psf.T).astype(np.float32)
    h, w = bgr.shape[:2]
    padded_psf = np.zeros((h, w), np.float32)
    ph, pw = psf.shape
    padded_psf[:ph, :pw] = psf
    padded_psf = np.roll(padded_psf, -ph // 2, axis=0)
    padded_psf = np.roll(padded_psf, -pw // 2, axis=1)
    H = np.fft.fft2(padded_psf)
    H_conj = np.conj(H)
    denom = H * H_conj + k
    out = np.empty_like(bgr)
    for c in range(3):
        G = np.fft.fft2(bgr[..., c].astype(np.float32))
        F_hat = G * H_conj / denom
        recon = np.real(np.fft.ifft2(F_hat))
        out[..., c] = np.clip(recon, 0, 255).astype(np.uint8)
    return out


def _strong_unsharp(bgr: "np.ndarray", amount: float = 1.8, radius: int = 3) -> "np.ndarray":
    """Heavier unsharp than _simple_sharpen — used only on blurred crops."""
    blur = cv2.GaussianBlur(bgr, (0, 0), radius)
    sharpened = cv2.addWeighted(bgr, 1 + amount, blur, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _blur_aware_recover(bgr: "np.ndarray", before: dict) -> "np.ndarray":
    """If the original crop was below BLUR_RECOVERY_THRESHOLD, run a recovery
    stack: optional 2x upscale → Wiener deconvolution → strong unsharp →
    extra CLAHE on the L channel. No-op on sharp images so well-focused
    uploads are untouched.
    """
    bv = before.get("blur_var", 0.0)
    if bv >= BLUR_RECOVERY_THRESHOLD:
        return bgr
    work = bgr
    # Very blurry -> upscale first so deconvolution has more pixels to work with
    if bv < BLUR_UPSCALE_THRESHOLD:
        work = cv2.resize(work, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # Wiener pass tuned to mild Gaussian blur; harder pass when extremely blurry
    psf = 5 if bv < BLUR_UPSCALE_THRESHOLD else 3
    work = _wiener_deblur(work, psf_size=psf, k=0.015 if bv < BLUR_UPSCALE_THRESHOLD else 0.010)
    # Heavier unsharp because Wiener alone leaves edges soft
    work = _strong_unsharp(work, amount=1.6 if bv < BLUR_UPSCALE_THRESHOLD else 1.2, radius=2)
    # Extra local contrast — readability of small dates depends on it
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    work = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    # If we upscaled, bring it back down so downstream stages see the working size
    if work.shape[:2] != bgr.shape[:2]:
        work = cv2.resize(work, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_AREA)
    log.info(f"blur-aware recovery applied (blur_var={bv:.1f}, threshold={BLUR_RECOVERY_THRESHOLD})")
    return work


def _quality_metrics(bgr: "np.ndarray") -> dict:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {
        "blur_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean":     float(np.mean(gray)),
        "std":      float(np.std(gray)),
        "p1":       float(np.percentile(gray, 1)),
        "p99":      float(np.percentile(gray, 99)),
    }

# ── Document quad detection + perspective warp ───────────────────────────────
def _order_quad(pts: "np.ndarray") -> "np.ndarray":
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def _detect_document_quad(bgr: "np.ndarray"):
    try:
        h, w = bgr.shape[:2]
        total_area = float(h * w)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.25 * total_area or area > 0.99 * total_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
                best = approx
                best_area = area
        if best is None:
            return None
        return _order_quad(best)
    except Exception:
        return None

def _perspective_warp(bgr: "np.ndarray", quad: "np.ndarray") -> "np.ndarray":
    tl, tr, br, bl = quad
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if W < 50 or H < 50:
        return bgr
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_CUBIC)

# ── Fine deskew via Hough lines ──────────────────────────────────────────────
def _fine_deskew(bgr: "np.ndarray", max_deg: float = 8.0) -> "np.ndarray":
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        lines = cv2.HoughLines(edges, 1, np.pi / 720, threshold=200)
        if lines is None or len(lines) == 0:
            return bgr
        angles = []
        for rt in lines[:200]:
            theta = rt[0][1]
            deg = (theta * 180.0 / np.pi) - 90.0
            if -max_deg <= deg <= max_deg:
                angles.append(deg)
        if not angles:
            return bgr
        angle = float(np.median(angles))
        if abs(angle) < 0.3:
            return bgr
        h, w = bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return bgr

# ── Color / illumination ─────────────────────────────────────────────────────
def _gray_world_white_balance(bgr: "np.ndarray") -> "np.ndarray":
    means = cv2.mean(bgr)[:3]
    b_avg, g_avg, r_avg = means
    if min(b_avg, g_avg, r_avg) < 1:
        return bgr
    gray_avg = (b_avg + g_avg + r_avg) / 3.0
    b = np.clip(bgr[:, :, 0].astype(np.float32) * (gray_avg / b_avg), 0, 255).astype(np.uint8)
    g = np.clip(bgr[:, :, 1].astype(np.float32) * (gray_avg / g_avg), 0, 255).astype(np.uint8)
    r = np.clip(bgr[:, :, 2].astype(np.float32) * (gray_avg / r_avg), 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])

def _remove_shadows(bgr: "np.ndarray") -> "np.ndarray":
    try:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        ksize = max(3, int(min(l.shape) / 25) | 1)
        bg = cv2.GaussianBlur(l, (ksize, ksize), 0)
        l_flat = cv2.subtract(l.astype(np.int16), bg.astype(np.int16))
        l_flat = cv2.add(l_flat, 128)
        l_flat = np.clip(l_flat, 0, 255).astype(np.uint8)
        return cv2.cvtColor(cv2.merge([l_flat, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        return bgr

def _percentile_contrast_stretch(bgr: "np.ndarray", lo: float = 1.0, hi: float = 99.0) -> "np.ndarray":
    out = bgr.copy()
    for c in range(3):
        ch = bgr[:, :, c]
        p_lo, p_hi = np.percentile(ch, [lo, hi])
        if p_hi - p_lo > 8:
            stretched = (ch.astype(np.float32) - p_lo) * (255.0 / (p_hi - p_lo))
            out[:, :, c] = np.clip(stretched, 0, 255).astype(np.uint8)
    return out

# ── Artifact removal ─────────────────────────────────────────────────────────
def _reduce_moire(bgr: "np.ndarray") -> "np.ndarray":
    med = cv2.medianBlur(bgr, 3)
    return cv2.bilateralFilter(med, d=5, sigmaColor=25, sigmaSpace=25)

def _reduce_glare(bgr: "np.ndarray") -> "np.ndarray":
    try:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(mask) > 0.08 * mask.size:
            return bgr
        if cv2.countNonZero(mask) == 0:
            return bgr
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)
        return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    except Exception:
        return bgr

# ── Local contrast / gamma / denoise ─────────────────────────────────────────
def _clahe_l(bgr: "np.ndarray") -> "np.ndarray":
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

def _auto_gamma(bgr: "np.ndarray", metrics: dict) -> "np.ndarray":
    mean = metrics.get("mean", 128.0)
    if mean <= 0 or mean >= 255:
        return bgr
    try:
        gamma = float(np.log(0.5) / np.log(mean / 255.0))
    except Exception:
        return bgr
    gamma = max(0.55, min(1.4, gamma))
    if abs(gamma - 1.0) < 0.05:
        return bgr
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(bgr, table)

def _denoise(bgr: "np.ndarray", metrics: dict) -> "np.ndarray":
    if metrics.get("blur_var", 0) > 400:
        return bgr
    strength = 35 if metrics.get("blur_var", 0) < 150 else 25
    return cv2.bilateralFilter(bgr, d=5, sigmaColor=strength, sigmaSpace=35)

# ── Sharpening ───────────────────────────────────────────────────────────────
def _text_edge_mask(bgr: "np.ndarray") -> "np.ndarray":
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    mag = cv2.convertScaleAbs(sobel)
    mag = cv2.GaussianBlur(mag, (0, 0), sigmaX=1.5)
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return (norm.astype(np.float32) / 255.0)[..., None]

def _multiscale_unsharp(bgr: "np.ndarray", metrics: dict) -> "np.ndarray":
    blur_var = metrics.get("blur_var", 200)
    if blur_var < 60:
        amt_f, amt_c = 0.85, 0.40
    elif blur_var < 200:
        amt_f, amt_c = 0.65, 0.30
    else:
        amt_f, amt_c = 0.40, 0.20
    f = cv2.GaussianBlur(bgr, (0, 0), sigmaX=0.9)
    c = cv2.GaussianBlur(bgr, (0, 0), sigmaX=2.4)
    base = bgr.astype(np.float32)
    high_f = base - f.astype(np.float32)
    high_c = base - c.astype(np.float32)
    sharp = base + amt_f * high_f + amt_c * high_c
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    if pcfg.PP_TEXT_AWARE_SHARP:
        mask = _text_edge_mask(bgr)
        blended = mask * sharp.astype(np.float32) + (1.0 - mask) * (0.5 * sharp.astype(np.float32) + 0.5 * base)
        return np.clip(blended, 0, 255).astype(np.uint8)
    return sharp

def _simple_sharpen(bgr: "np.ndarray", metrics: dict) -> "np.ndarray":
    blur_var = metrics.get("blur_var", 200)
    amt = 0.8 if blur_var < 60 else (0.6 if blur_var < 200 else 0.4)
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.2)
    sharp = cv2.addWeighted(bgr, 1 + amt, blurred, -amt, 0)
    return sharp

# ── Resize to working width ──────────────────────────────────────────────────
def _resize_to_working(bgr: "np.ndarray") -> "np.ndarray":
    h, w = bgr.shape[:2]
    if w < pcfg.MIN_WIDTH:
        scale = pcfg.MIN_WIDTH / w
        return cv2.resize(bgr, (pcfg.MIN_WIDTH, int(h * scale)), interpolation=cv2.INTER_CUBIC)
    if w > pcfg.TARGET_WIDTH:
        scale = pcfg.TARGET_WIDTH / w
        return cv2.resize(bgr, (pcfg.TARGET_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    return bgr

# ── Quality guard ────────────────────────────────────────────────────────────
def _is_worse(before: dict, after: dict) -> bool:
    if after["blur_var"] < 0.55 * before["blur_var"] and before["blur_var"] > 80:
        return True
    if after["std"] < 0.65 * before["std"] and before["std"] > 25:
        return True
    if after["mean"] < 25 or after["mean"] > 235:
        return True
    return False

# ── Atomic JPEG write ────────────────────────────────────────────────────────
def _atomic_write_jpeg(bgr: "np.ndarray", out_path: Path, quality: int = 96) -> str:
    tmp = out_path.with_name(out_path.name + ".tmp" + out_path.suffix)
    ok = cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise IOError(f"Failed to write {tmp}")
    os.replace(str(tmp), str(out_path))
    return str(out_path)

# ── Main preprocessing pipeline ──────────────────────────────────────────────
def preprocess_image(image_path: str) -> str:
    if not pcfg.ENABLE_PREPROCESS:
        return image_path
    try:
        img = cv2.imread(image_path)
        if img is None:
            # Fallback: try PIL for exotic formats then convert
            try:
                from PIL import Image
                pim = Image.open(image_path).convert("RGB")
                img = cv2.cvtColor(np.array(pim), cv2.COLOR_RGB2BGR)
            except Exception:
                return image_path

        if pcfg.PP_EXIF_ORIENT:
            img = _apply_exif_orientation(img, image_path)

        before = _quality_metrics(img)

        # Geometric correction (at full resolution)
        if pcfg.PP_DOC_WARP:
            quad = _detect_document_quad(img)
            if quad is not None:
                img = _perspective_warp(img, quad)

        if pcfg.PP_FINE_DESKEW:
            img = _fine_deskew(img, max_deg=pcfg.PP_MAX_DESKEW_DEG)

        # Resize to working width
        img = _resize_to_working(img)

        # Color correction
        if pcfg.PP_WHITE_BALANCE:
            img = _gray_world_white_balance(img)
        if pcfg.PP_SHADOW_REMOVE:
            img = _remove_shadows(img)
        if pcfg.PP_CONTRAST_STRETCH:
            img = _percentile_contrast_stretch(img)

        # Artifact removal
        if pcfg.PP_MOIRE_REDUCE:
            img = _reduce_moire(img)
        if pcfg.PP_GLARE_INPAINT:
            img = _reduce_glare(img)

        # Local contrast / tone
        if pcfg.PP_CLAHE:
            img = _clahe_l(img)
        if pcfg.PP_AUTO_GAMMA:
            img = _auto_gamma(img, before)

        # Denoise
        if pcfg.PP_DENOISE:
            img = _denoise(img, before)

        # Blur-aware recovery: aggressively rescue heavily blurred crops
        img = _blur_aware_recover(img, before)

        # Sharpen
        if pcfg.SHARPEN:
            if pcfg.PP_MULTISCALE_SHARPEN:
                img = _multiscale_unsharp(img, before)
            else:
                img = _simple_sharpen(img, before)

        after = _quality_metrics(img)
        if pcfg.PP_QUALITY_LOG:
            log.info(
                f"preprocess metrics: blur {before['blur_var']:.1f}→{after['blur_var']:.1f}  "
                f"std {before['std']:.1f}→{after['std']:.1f}  "
                f"mean {before['mean']:.1f}→{after['mean']:.1f}"
            )

        if pcfg.PP_QUALITY_GUARD and _is_worse(before, after):
            log.warning("Preprocess quality regressed — using original image")
            return image_path

        out_path = Path(image_path).with_name(f"{Path(image_path).stem}_pp.jpg")
        return _atomic_write_jpeg(img, out_path, quality=pcfg.PP_JPEG_QUALITY)
    except Exception as e:
        log.warning(f"preprocess_image failed ({e}) — using original image")
        return image_path

def cleanup_preprocessed(original: str, preprocessed: str) -> None:
    if original != preprocessed:
        Path(preprocessed).unlink(missing_ok=True)

_NULL_STRINGS = {"null", "none", "n/a", "na", "not available", "not present", "not visible", "-", ""}

def postprocess_data(data: dict) -> dict:
    """
    Only trim strings and normalise separators.
    Additionally, if date_of_issue equals date_of_expiry or date_of_birth,
    set date_of_issue to null (hallucination fix).
    """
    result = dict(data)

    # Convert string "null" / "none" / empty → Python None for every field
    for k, v in result.items():
        if isinstance(v, str) and v.strip().lower() in _NULL_STRINGS:
            result[k] = None

    if pcfg.NORMALISE_DATES:
        for f in ("date_of_birth", "date_of_issue", "date_of_expiry"):
            if result.get(f) and isinstance(result[f], str):
                result[f] = result[f].strip().replace(".", "/").replace("-", "/")
    for f in ("full_name_english", "full_name_arabic", "nationality", "category", "blood_group"):
        if result.get(f) and isinstance(result[f], str):
            result[f] = result[f].strip()
    if pcfg.REDACT_NAME:
        result["full_name_english"] = "***REDACTED***"
        result["full_name_arabic"] = "***REDACTED***"

    # Fix hallucinated date_of_issue: if it equals expiry or birth, nullify it
    doi = result.get("date_of_issue")
    doe = result.get("date_of_expiry")
    dob = result.get("date_of_birth")
    if doi and doe and str(doi).strip() == str(doe).strip():
        log.warning(f"date_of_issue equals date_of_expiry ('{doi}') – setting to null (no English ISS)")
        result["date_of_issue"] = None
    if doi and dob and str(doi).strip() == str(dob).strip():
        log.warning(f"date_of_issue equals date_of_birth ('{doi}') – setting to null (no English ISS)")
        result["date_of_issue"] = None

    # A license cannot be issued in the same month+year as the holder's birth — must be hallucination
    doi = result.get("date_of_issue")
    if doi and dob:
        doi_parts = str(doi).strip().split("/")
        dob_parts = str(dob).strip().split("/")
        if len(doi_parts) == 3 and len(dob_parts) == 3:
            if doi_parts[1] == dob_parts[1] and doi_parts[2] == dob_parts[2]:
                log.warning(f"date_of_issue ('{doi}') shares month+year with date_of_birth ('{dob}') — setting to null")
                result["date_of_issue"] = None

    # Hijri contamination detection for date_of_birth (log only — do NOT null the field):
    # Saudi Hijri years are always 14XX. The model sometimes replaces the Gregorian day with "14"
    # (the first two digits of the Hijri year). Detection: if DOB day == "14" AND ISS or EXP
    # day is NOT "14", the DOB day is likely a Hijri contamination.
    dob = result.get("date_of_birth")
    doi = result.get("date_of_issue")
    doe = result.get("date_of_expiry")
    if dob:
        dob_parts = str(dob).strip().split("/")
        if len(dob_parts) == 3 and dob_parts[0] == "14":
            other_days = []
            for d in (doi, doe):
                if d:
                    p = str(d).strip().split("/")
                    if len(p) == 3:
                        other_days.append(p[0])
            if other_days and any(d != "14" for d in other_days):
                log.warning(f"date_of_birth day '14' looks like Hijri year contamination (ISS/EXP days={other_days}) — model should have read the English DOB day instead")

    return result

# ── Async pipeline ─────────────────────────────────────────────────────────────
_YOLO_SEMAPHORE = asyncio.Semaphore(int(os.getenv("MAX_YOLO_CONCURRENT", "2")))

async def async_process_one(image_path: str, pipeline: LicenseAuthPipeline) -> PipelineResult:
    start_ms = time.monotonic() * 1000
    try:
        validate_image(image_path)
    except ImageValidationError as e:
        return PipelineResult(image_path=image_path, success=False, error=str(e), stage="input", duration_ms=time.monotonic()*1000 - start_ms)
    async with _YOLO_SEMAPHORE:
        det = await asyncio.to_thread(pipeline.detect_license, image_path)
    if not det["detected"]:
        return PipelineResult(image_path=image_path, success=False, error=det["reason"], stage="detection", duration_ms=time.monotonic()*1000 - start_ms)
    cropped = await asyncio.to_thread(pipeline.crop_license, image_path, det["bbox"])
    # Quality gate: blur check on the crop before any preprocessing / VLM call
    blurry, blur_score = await asyncio.to_thread(is_too_blurry, cropped)
    if blurry:
        return PipelineResult(
            image_path=image_path,
            success=False,
            error=f"Image too blurry to extract (blur score {blur_score:.1f} < {BLUR_VARIANCE_THRESHOLD})",
            stage="quality",
            detection_confidence=det["confidence"],
            cropped_image_path=cropped,
            duration_ms=time.monotonic()*1000 - start_ms,
        )
    preprocessed = await asyncio.to_thread(preprocess_image, cropped)
    was_preprocessed = (preprocessed != cropped)
    try:
        ocr = await async_extract_license_data(preprocessed)
        if not ocr["success"]:
            return PipelineResult(image_path=image_path, success=False, error=ocr.get("error"), stage="ocr", detection_confidence=det["confidence"], cropped_image_path=cropped, preprocessed=was_preprocessed, duration_ms=time.monotonic()*1000 - start_ms)
        processed_data = postprocess_data(ocr["data"])
        # Corrective DOB pass: if main extraction returned day "14" (Hijri contamination), re-extract with focused prompt
        dob_check = processed_data.get("date_of_birth")
        if dob_check and isinstance(dob_check, str):
            dob_parts = dob_check.strip().split("/")
            if len(dob_parts) == 3 and dob_parts[0] == "14":
                log.warning(f"DOB day='14' detected, running corrective extraction pass...")
                corrected = await async_extract_dob_corrective(preprocessed)
                if corrected:
                    log.info(f"DOB corrected: '{dob_check}' → '{corrected}'")
                    processed_data["date_of_birth"] = corrected
        validation = validate_extracted_data(processed_data)
        return PipelineResult(image_path=image_path, success=True, data=processed_data, validation=validation, stage="complete", detection_confidence=det["confidence"], cropped_image_path=cropped, preprocessed=was_preprocessed, duration_ms=time.monotonic()*1000 - start_ms)
    finally:
        await asyncio.to_thread(cleanup_preprocessed, cropped, preprocessed)

async def async_process_batch(image_paths: list[str], pipeline: LicenseAuthPipeline, max_concurrent: int = 10) -> tuple[list[PipelineResult], PipelineMetrics]:
    metrics = PipelineMetrics(total=len(image_paths))
    results = [None] * len(image_paths)
    sem = asyncio.Semaphore(max_concurrent)
    async def process_one(idx: int, path: str):
        async with sem:
            res = await async_process_one(path, pipeline)
            results[idx] = res
            metrics.total_ms += res.duration_ms
            if res.success:
                metrics.succeeded += 1
            else:
                metrics.failed += 1
    await asyncio.gather(*[process_one(i, p) for i, p in enumerate(image_paths)])
    return results, metrics

async def async_run_pipeline(image_paths: list[str], model_path: Optional[str] = None, max_concurrent: Optional[int] = None) -> dict:
    if pcfg.CLEAR_CACHE_START:
        clear_cache()
    if OCRConfig.VLM_BACKEND == "together":
        health = await async_check_together_health()
        if not health["healthy"]:
            return {"results": [], "metrics": {}, "error": health["error"]}
    pipeline = LicenseAuthPipeline(model_path=model_path)
    results, metrics = await async_process_batch(image_paths, pipeline, max_concurrent=max_concurrent or 10)
    return {"results": [r.to_dict() for r in results if r is not None], "metrics": metrics.summary()}

def _process_one_sync(image_path: str, pipeline: LicenseAuthPipeline) -> PipelineResult:
    start = time.monotonic() * 1000
    try:
        validate_image(image_path)
    except ImageValidationError as e:
        return PipelineResult(image_path=image_path, success=False, error=str(e), stage="input", duration_ms=time.monotonic()*1000 - start)
    det = pipeline.detect_license(image_path)
    if not det["detected"]:
        return PipelineResult(image_path=image_path, success=False, error=det["reason"], stage="detection", duration_ms=time.monotonic()*1000 - start)
    cropped = pipeline.crop_license(image_path, det["bbox"])
    preprocessed = preprocess_image(cropped)
    was_preprocessed = (preprocessed != cropped)
    try:
        ocr = extract_license_data(preprocessed)
        if not ocr["success"]:
            return PipelineResult(image_path=image_path, success=False, error=ocr.get("error"), stage="ocr", detection_confidence=det["confidence"], cropped_image_path=cropped, preprocessed=was_preprocessed, duration_ms=time.monotonic()*1000 - start)
        processed = postprocess_data(ocr["data"])
        # Corrective DOB pass: if main extraction returned day "14" (Hijri contamination), re-extract with focused prompt
        dob_check = processed.get("date_of_birth")
        if dob_check and isinstance(dob_check, str):
            dob_parts = dob_check.strip().split("/")
            if len(dob_parts) == 3 and dob_parts[0] == "14":
                log.warning(f"DOB day='14' detected, running corrective extraction pass...")
                corrected = extract_dob_corrective(preprocessed)
                if corrected:
                    log.info(f"DOB corrected: '{dob_check}' → '{corrected}'")
                    processed["date_of_birth"] = corrected
        validation = validate_extracted_data(processed)
        return PipelineResult(image_path=image_path, success=True, data=processed, validation=validation, stage="complete", detection_confidence=det["confidence"], cropped_image_path=cropped, preprocessed=was_preprocessed, duration_ms=time.monotonic()*1000 - start)
    finally:
        cleanup_preprocessed(cropped, preprocessed)

def run_pipeline(image_paths: list[str], model_path: Optional[str] = None, workers: Optional[int] = None) -> dict:
    if pcfg.CLEAR_CACHE_START:
        clear_cache()
    if OCRConfig.VLM_BACKEND == "together":
        health = check_together_health()
        if not health["healthy"]:
            return {"results": [], "metrics": {}, "error": health["error"]}
    pipeline = LicenseAuthPipeline(model_path=model_path)
    metrics = PipelineMetrics(total=len(image_paths))
    results = []
    with ThreadPoolExecutor(max_workers=workers or pcfg.MAX_WORKERS) as pool:
        futures = {pool.submit(_process_one_sync, p, pipeline): i for i, p in enumerate(image_paths)}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            metrics.total_ms += res.duration_ms
            if res.success:
                metrics.succeeded += 1
            else:
                metrics.failed += 1
    return {"results": [r.to_dict() for r in results], "metrics": metrics.summary()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*", default=["test_license.jpg"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--workers", type=int, default=pcfg.MAX_WORKERS)
    args = parser.parse_args()
    if args.health:
        print(json.dumps(check_together_health(), indent=2))
        sys.exit(0)
    if args.batch:
        allowed = set(OCRConfig.ALLOWED_EXTS)
        paths = [str(p) for p in Path(".").iterdir() if p.suffix.lower() in allowed]
        if not paths:
            log.error("No supported images found.")
            sys.exit(1)
        output = run_pipeline(paths, model_path=args.model, workers=args.workers)
    else:
        output = run_pipeline(args.images, model_path=args.model, workers=args.workers)
    print(json.dumps(output, indent=2, ensure_ascii=False))