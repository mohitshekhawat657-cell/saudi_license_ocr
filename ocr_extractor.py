"""
Production-grade Together AI VLM Extractor — Saudi Arabian Driving License.
Extracts ONLY English Gregorian dates (DOB, Exp, ISS/ISSN).
No Hijri extraction, no fallback, no conversion.
Uses Together AI API with Qwen3.5-9B model.
"""

import base64
import hashlib
import json
import logging
import os
import re
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ocr_extractor")

class Config:
    VLM_BACKEND:       str   = os.getenv("VLM_BACKEND",      "together").lower()
    
    # Together AI Configuration
    TOGETHER_API_KEY:  str   = os.getenv("TOGETHER_API_KEY", "")
    TOGETHER_MODEL:    str   = os.getenv("TOGETHER_MODEL",   "Qwen/Qwen3.5-9B")
    TOGETHER_BASE_URL: str   = os.getenv("TOGETHER_BASE_URL","https://api.together.xyz/v1")
    
    # API Settings
    TIMEOUT:           int   = int(os.getenv("TOGETHER_TIMEOUT",   "120"))
    MAX_RETRIES:       int   = int(os.getenv("TOGETHER_RETRIES",   "3"))
    RETRY_BACKOFF:     float = 2.0
    
    # Model Parameters
    TEMPERATURE:       float = float(os.getenv("TOGETHER_TEMPERATURE", "0.0"))
    TOP_P:             float = float(os.getenv("TOGETHER_TOP_P",       "0.95"))
    TOP_K:             int   = int(os.getenv("TOGETHER_TOP_K",         "20"))
    MAX_TOKENS:        int   = int(os.getenv("TOGETHER_MAX_TOKENS",    "1500"))
    
    # Thinking Mode (Qwen3.5 specific)
    ENABLE_THINKING:   bool  = os.getenv("TOGETHER_ENABLE_THINKING", "false").lower() == "true"
    PRESENCE_PENALTY:  float = float(os.getenv("TOGETHER_PRESENCE_PENALTY", "1.5"))
    
    # Concurrency
    MAX_CONCURRENT:    int   = int(os.getenv("MAX_CONCURRENT", "4"))
    MAX_CONNECTIONS:   int   = int(os.getenv("TOGETHER_MAX_CONNECTIONS", "20"))
    
    # Image Settings
    MAX_IMAGE_MB:      int   = 50
    ALLOWED_EXTS:      tuple = (".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp", ".gif", ".heic", ".heif", ".pdf")
    ENABLE_CACHE:      bool  = os.getenv("ENABLE_CACHE", "true").lower() == "true"

    # Ultra preprocessing
    ENABLE_ULTRA_PREPROCESS: bool = os.getenv("ENABLE_ULTRA_PREPROCESS", "true").lower() == "true"
    PREPROCESS_TARGET_SIZE:  int  = int(os.getenv("PREPROCESS_TARGET_SIZE", "2800"))
    PREPROCESS_MIN_SIZE:     int  = int(os.getenv("PREPROCESS_MIN_SIZE",    "2000"))
    PREPROCESS_DEBUG:        bool = os.getenv("PREPROCESS_DEBUG", "false").lower() == "true"

cfg = Config()

# Validate API key
if not cfg.TOGETHER_API_KEY and cfg.VLM_BACKEND == "together":
    log.warning("TOGETHER_API_KEY not set! Please add it to your .env file")

# ── Prompt – English Gregorian dates only, no Hijri ──────────────────────────
EXTRACTION_PROMPT = """You are an expert OCR system for Saudi Arabian driving licenses.

Extract ALL visible fields from the license and return ONLY a valid JSON object.
No markdown, no explanation, no <think> block.

{
  "license_number": "",
  "full_name_english": "",
  "full_name_arabic": "",
  "date_of_birth": "",
  "date_of_issue": "",
  "date_of_expiry": "",
  "category": "",
  "nationality": "",
  "blood_group": "",
  "restrictions": "",
  "version": "",
  "issuing_country": "Saudi Arabia"
}

RULES FOR DATES (CRITICAL):
- Read ONLY the Gregorian (Western) dates that appear DIRECTLY NEXT TO an English label on the card.
- date_of_birth  → date immediately after the English label "DOB" or "Date of Birth:" ONLY (format DD/MM/YYYY)
- date_of_expiry → date immediately after the English label "EXP", "Exp", or "Expiry Date:" ONLY (format DD/MM/YYYY)
- date_of_issue  → date immediately after the English label "ISS", "ISSN", or "Issue Date:" ONLY (format DD/MM/YYYY)
- STRICT: If a field has NO English label on the card, set it to null. Do NOT infer it from Arabic dates.
- HIJRI DATES (right column): Arabic dates starting with ١٤ (e.g. ١٤١٠/٠٩/٠٤, ١٤٥١/٠٥/٢٥) are Hijri — IGNORE completely.
- DO NOT invent, convert, or guess any date. Copy the exact digits printed next to the English label.
- EXAMPLE: card shows only "No.", "DOB 30/03/1990", "Exp 04/10/2029" in English (no ISS label visible) → date_of_issue = null.
- VERIFIED EXAMPLE: Card row "DOB 03/11/1997  تاريخ الميلاد ١٤١٨/٠٧/٠٣" → date_of_birth = "03/11/1997"

Other fields:
- full_name_english: ALL CAPITAL ENGLISH letters on the card (e.g. "SUFYANI, AMAL MOHAMMED M")
- full_name_arabic: Arabic script name only.
- license_number: numeric ID next to "No.", "ID Number:", or "الرقم" label.
- category: Extract the VALUE only — never include the label prefix.
- nationality: Extract the VALUE only — never include the label prefix.
- blood_group: The ABO blood type ONLY — must be one of: A+, A-, B+, B-, O+, O-, AB+, AB-.
- restrictions: The numeric code printed IMMEDIATELY after the English "RESTR" label.
- version: The number printed after "VER" or "النسخة" label only.

If NOT a driving license return: {"error": "not_a_license"}
Only return {"error": "image_too_blurry"} if the image is so severely blurry that NO fields at all can be read.
"""

# ── Focused DOB-correction prompt ────────────────────────────────────────────
DOB_CORRECTION_PROMPT = """Look at this Saudi driving license image.
Find the English label "DOB" printed on the LEFT side of the card.
Read the date that appears IMMEDIATELY after "DOB " on the same line.
COPY the date EXACTLY character by character as it is printed on the card.
The card prints the date as DD/MM/YYYY. Copy it in that exact order.
Do NOT look at the Arabic/Hijri date on the RIGHT side of the card.
Return ONLY the 10-character date string exactly as printed. No label, no explanation.
If the card shows "DOB 03/11/1997" then output exactly: 03/11/1997"""

# ── Exceptions ────────────────────────────────────────────────────────────────
class OCRError(Exception): pass
class TogetherAPIError(OCRError): pass
class ImageValidationError(OCRError): pass
class ExtractionError(OCRError): pass

# ── Image utilities ───────────────────────────────────────────────────────────
def validate_image(image_path: str) -> Path:
    path = Path(image_path)
    if not path.exists():
        raise ImageValidationError(f"Image not found: {image_path}")
    if path.suffix.lower() not in cfg.ALLOWED_EXTS:
        raise ImageValidationError(f"Unsupported format '{path.suffix}'. Allowed: {cfg.ALLOWED_EXTS}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > cfg.MAX_IMAGE_MB:
        raise ImageValidationError(f"Image too large ({size_mb:.1f} MB). Max: {cfg.MAX_IMAGE_MB} MB")
    return path

def encode_image(image_path: str) -> str:
    import io
    path = Path(image_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(image_path, dpi=200, first_page=1, last_page=1)
            if not pages:
                raise OCRError("PDF has no pages")
            buf = io.BytesIO()
            pages[0].save(buf, format="JPEG", quality=95)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except ImportError:
            raise OCRError("pdf2image required for PDF support. pip install pdf2image")
    if ext in (".heic", ".heif"):
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except ImportError:
            raise OCRError("pillow-heif required for HEIC support")
    if ext in (".tiff", ".tif", ".bmp", ".gif"):
        from PIL import Image
        img = Image.open(image_path)
        if ext == ".gif":
            img.seek(0)
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ── Ultra preprocessing pipeline ──────────────────────────────────────────────
def _pil_open_oriented(image_path: str):
    """Load any supported format → EXIF-corrected RGB numpy array."""
    import io
    import numpy as np
    from PIL import Image, ImageOps

    ext = Path(image_path).suffix.lower()
    if ext == ".pdf":
        from pdf2image import convert_from_path
        pages = convert_from_path(image_path, dpi=300, first_page=1, last_page=1)
        if not pages:
            raise OCRError("PDF has no pages")
        img = pages[0]
    elif ext in (".heic", ".heif"):
        import pillow_heif
        pillow_heif.register_heif_opener()
        img = Image.open(image_path)
    else:
        img = Image.open(image_path)

    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img)

def _order_quad(pts):
    import numpy as np
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype="float32")

def _detect_and_warp_card(img_bgr):
    """Find largest convex 4-gon ≥25% of frame and warp to flat top-down view."""
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]
    frame_area = float(h * w)

    scale = 1000.0 / max(h, w)
    if scale < 1.0:
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr.copy()
        scale = 1.0
    inv_scale = 1.0 / scale

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    small_area = float(small.shape[0] * small.shape[1])

    quad = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if cv2.contourArea(approx) >= 0.25 * small_area:
                quad = approx
                break
    if quad is None:
        return img_bgr

    quad = (quad.reshape(4, 2) * inv_scale).astype("float32")
    quad = _order_quad(quad)
    tl, tr, br, bl = quad

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    target_w = int(max(widthA, widthB))
    target_h = int(max(heightA, heightB))
    if target_w < 200 or target_h < 200:
        return img_bgr
    if target_w * target_h < 0.20 * frame_area:
        return img_bgr

    dst = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img_bgr, M, (target_w, target_h))

def _deskew(img_bgr):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if coords.size == 0:
        return img_bgr
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.3 or abs(angle) > 15:
        return img_bgr
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _shadow_removal(img_bgr):
    import cv2
    import numpy as np

    out_channels = []
    for ch in cv2.split(img_bgr):
        dilated = cv2.dilate(ch, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(ch, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        out_channels.append(norm)
    return cv2.merge(out_channels)

def _white_balance(img_bgr):
    import numpy as np

    result = img_bgr.astype("float32")
    means = [result[:, :, i].mean() for i in range(3)]
    gray_mean = float(np.mean(means))
    for i in range(3):
        if means[i] > 1e-3:
            result[:, :, i] *= gray_mean / means[i]
    return np.clip(result, 0, 255).astype("uint8")

def _clahe_contrast(img_bgr):
    import cv2

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def _gamma_correct(img_bgr, gamma: float = 1.12):
    import cv2
    import numpy as np

    inv = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, lut)

def _unsharp_mask(img_bgr, amount: float = 1.0, radius: float = 1.2):
    import cv2
    import numpy as np

    blur = cv2.GaussianBlur(img_bgr, (0, 0), radius)
    out = cv2.addWeighted(img_bgr, 1 + amount, blur, -amount, 0)
    return np.clip(out, 0, 255).astype("uint8")

def _fine_digit_sharpen(img_bgr):
    """Sub-pixel sharpening on LAB L-channel (sigma=0.5).
    Targets the fine strokes that distinguish digit shapes —
    specifically the middle bar of '8' vs '3' and loop closures.
    AVX2-accelerated Gaussian on this server so cost is minimal.
    """
    import cv2
    import numpy as np
    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        blur = cv2.GaussianBlur(l, (0, 0), sigmaX=0.5)
        l_sharp = np.clip(
            l.astype(np.float32) * 1.9 - blur.astype(np.float32) * 0.9,
            0, 255
        ).astype("uint8")
        return cv2.cvtColor(cv2.merge([l_sharp, a, b]), cv2.COLOR_LAB2BGR)
    except Exception:
        return img_bgr

def _adaptive_resize(img_bgr, target: int, min_size: int):
    import cv2

    h, w = img_bgr.shape[:2]
    longest = max(h, w)
    if longest < min_size:
        scale = min_size / longest
        # LANCZOS4 for upscaling — best interpolation quality for small digit details
        return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
    if longest > target:
        scale = target / longest
        return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_bgr

def ultra_preprocess_image(image_path: str) -> str:
    """Run 9-stage preprocessing and return path to a temp _ultra.jpg.

    On any failure, returns the original image_path so extraction never breaks.
    """
    if not cfg.ENABLE_ULTRA_PREPROCESS:
        return image_path
    try:
        import cv2
        import numpy as np

        rgb = _pil_open_oriented(image_path)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        img = _detect_and_warp_card(img)
        img = _deskew(img)
        img = _shadow_removal(img)
        img = _white_balance(img)
        img = _clahe_contrast(img)
        img = _gamma_correct(img, gamma=1.12)
        img = _unsharp_mask(img, amount=1.0, radius=1.2)
        img = _fine_digit_sharpen(img)
        img = _adaptive_resize(img, target=cfg.PREPROCESS_TARGET_SIZE, min_size=cfg.PREPROCESS_MIN_SIZE)

        src = Path(image_path)
        out_path = str(src.with_name(f"{src.stem}_ultra.jpg"))
        cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 97])
        h, w = img.shape[:2]
        log.info(f"Ultra preprocess complete: {out_path} ({w}x{h})")
        return out_path
    except Exception as e:
        log.warning(f"Ultra preprocess failed ({e}); using original image")
        return image_path

def image_hash(image_path: str) -> str:
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ── Response cache ────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS: int = 24 * 60 * 60  # entries auto-expire after 24 hours
_response_cache: dict[str, tuple[float, dict]] = {}

def _cache_get(key: str) -> Optional[dict]:
    entry = _response_cache.get(key)
    if entry is None:
        return None
    ts, value = entry
    if time.time() - ts > CACHE_TTL_SECONDS:
        _response_cache.pop(key, None)
        return None
    return value

def _cache_set(key: str, value: dict) -> None:
    _response_cache[key] = (time.time(), value)

def clear_cache() -> None: _response_cache.clear(); log.info("Cache cleared")

# ── Together AI Health ────────────────────────────────────────────────────────
def check_together_health() -> dict:
    if not cfg.TOGETHER_API_KEY:
        return {"healthy": False, "model_loaded": False, "error": "TOGETHER_API_KEY not set"}
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{cfg.TOGETHER_BASE_URL}/models",
                headers={"Authorization": f"Bearer {cfg.TOGETHER_API_KEY}"},
            )
            resp.raise_for_status()
            body = resp.json()
            models = body if isinstance(body, list) else body.get("data", [])
            model_ids = [m.get("id", "") for m in models if isinstance(m, dict)]
            model_loaded = any(cfg.TOGETHER_MODEL in m for m in model_ids)
            return {"healthy": True, "model_loaded": model_loaded, "error": None}
    except Exception as e:
        return {"healthy": False, "model_loaded": False, "error": str(e)}

async def async_check_together_health() -> dict:
    """Async version of health check."""
    return check_together_health()

# ── JSON parsing ──────────────────────────────────────────────────────────────
_NULL_STR_VALUES = {"null", "none", "n/a", "na", "not available", "not present", "not visible", "-", ""}

def _parse_json_response(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ExtractionError(f"No JSON object found in response. Raw: {raw[:300]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ExtractionError(f"JSON parse failed: {e}. Raw: {raw[:300]!r}")
    if "error" in data:
        raise ExtractionError(data["error"])
    for k, v in data.items():
        if isinstance(v, str) and v.strip().lower() in _NULL_STR_VALUES:
            data[k] = None
    return data

# ── Together AI API Client ────────────────────────────────────────────────────
class TogetherAIClient:
    def __init__(self):
        self.api_key = cfg.TOGETHER_API_KEY
        self.base_url = cfg.TOGETHER_BASE_URL
        self.model = cfg.TOGETHER_MODEL
        self._client = None
        self._async_client = None
        self._lock = asyncio.Lock()
    
    def _get_sync_client(self) -> httpx.Client:
        if self._client is None:
            limits = httpx.Limits(max_connections=cfg.MAX_CONNECTIONS)
            self._client = httpx.Client(
                timeout=httpx.Timeout(cfg.TIMEOUT),
                limits=limits,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._client
    
    async def _get_async_client(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._async_client is None:
                limits = httpx.Limits(max_connections=cfg.MAX_CONNECTIONS)
                self._async_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(cfg.TIMEOUT),
                    limits=limits,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
        return self._async_client
    
    def call_sync(self, prompt: str, image_b64: str, max_tokens: int = None) -> str:
        if not self.api_key:
            raise TogetherAPIError("TOGETHER_API_KEY is not set")
        
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }],
            "max_tokens": max_tokens or cfg.MAX_TOKENS,
            "temperature": cfg.TEMPERATURE,
            "top_p": cfg.TOP_P,
            "top_k": cfg.TOP_K,
            "presence_penalty": cfg.PRESENCE_PENALTY,
            "stream": False,
        }
        
        if not cfg.ENABLE_THINKING:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        
        client = self._get_sync_client()
        last_error = None
        
        for attempt in range(1, cfg.MAX_RETRIES + 1):
            if attempt > 1:
                wait = cfg.RETRY_BACKOFF ** (attempt - 1)
                log.info(f"Retry {attempt}/{cfg.MAX_RETRIES} in {wait:.0f}s...")
                time.sleep(wait)
            try:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content:
                    raise TogetherAPIError("Empty response from Together AI")
                return content
            except Exception as e:
                last_error = e
                log.warning(f"Attempt {attempt} failed: {e}")
        
        raise TogetherAPIError(f"All {cfg.MAX_RETRIES} attempts failed. Last error: {last_error}")
    
    async def call_async(self, prompt: str, image_b64: str, max_tokens: int = None) -> str:
        if not self.api_key:
            raise TogetherAPIError("TOGETHER_API_KEY is not set")
        
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }],
            "max_tokens": max_tokens or cfg.MAX_TOKENS,
            "temperature": cfg.TEMPERATURE,
            "top_p": cfg.TOP_P,
            "top_k": cfg.TOP_K,
            "presence_penalty": cfg.PRESENCE_PENALTY,
            "stream": False,
        }
        
        if not cfg.ENABLE_THINKING:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        
        client = await self._get_async_client()
        last_error = None
        
        for attempt in range(1, cfg.MAX_RETRIES + 1):
            if attempt > 1:
                await asyncio.sleep(cfg.RETRY_BACKOFF ** (attempt - 1))
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content:
                    raise TogetherAPIError("Empty response from Together AI")
                return content
            except Exception as e:
                last_error = e
                log.warning(f"Attempt {attempt} failed: {e}")
        
        raise TogetherAPIError(f"All {cfg.MAX_RETRIES} attempts failed. Last error: {last_error}")

# Global client
_together_client = None
def get_client() -> TogetherAIClient:
    global _together_client
    if _together_client is None:
        _together_client = TogetherAIClient()
    return _together_client

# ── Async client management ───────────────────────────────────────────────────
_async_client = None
_async_client_lock = asyncio.Lock()

async def get_async_client() -> httpx.AsyncClient:
    """Get or create async HTTP client for Together AI."""
    global _async_client
    async with _async_client_lock:
        if _async_client is None:
            _async_client = httpx.AsyncClient(
                timeout=httpx.Timeout(cfg.TIMEOUT),
                limits=httpx.Limits(max_connections=cfg.MAX_CONNECTIONS),
                headers={
                    "Authorization": f"Bearer {cfg.TOGETHER_API_KEY}",
                    "Content-Type": "application/json"
                }
            )
    return _async_client

async def close_async_client() -> None:
    """Close async HTTP client."""
    global _async_client
    if _async_client:
        await _async_client.aclose()
        _async_client = None
        log.info("Async HTTP client closed")

# ── Extraction functions ─────────────────────────────────────────────────────
def _extract_via_together(image_path: str) -> dict:
    health = check_together_health()
    if not health["healthy"]:
        return {"success": False, "error": health["error"]}
    
    cache_key = image_hash(image_path) if cfg.ENABLE_CACHE else None
    if cache_key and (cached := _cache_get(cache_key)):
        return cached

    processed_path = ultra_preprocess_image(image_path)
    image_b64 = encode_image(processed_path)
    client = get_client()

    try:
        raw = client.call_sync(EXTRACTION_PROMPT, image_b64)
        data = _parse_json_response(raw)
        result = {"success": True, "data": data}
    except Exception as e:
        result = {"success": False, "error": str(e)}
    finally:
        _cleanup_preprocessed(image_path, processed_path)

    if cache_key and result["success"]:
        _cache_set(cache_key, result)
    return result

def _cleanup_preprocessed(orig_path: str, processed_path: str) -> None:
    if processed_path == orig_path or cfg.PREPROCESS_DEBUG:
        return
    try:
        p = Path(processed_path)
        if p.exists() and p.name.endswith("_ultra.jpg"):
            p.unlink()
    except Exception as e:
        log.debug(f"Failed to cleanup {processed_path}: {e}")

# ── Public synchronous API ───────────────────────────────────────────────────
def extract_license_data(image_path: str) -> dict:
    log.info(f"Extracting from: {image_path}")
    try:
        validate_image(image_path)
    except ImageValidationError as e:
        return {"success": False, "error": str(e)}
    
    if cfg.VLM_BACKEND == "together":
        return _extract_via_together(image_path)
    else:
        return {"success": False, "error": f"Unknown VLM_BACKEND '{cfg.VLM_BACKEND}'"}

# ── Corrective DOB extraction ────────────────────────────────────────────────
def extract_dob_corrective(image_path: str) -> Optional[str]:
    """Re-extract DOB using a simple focused prompt. Returns DD/MM/YYYY string or None."""
    processed_path = ultra_preprocess_image(image_path)
    try:
        image_b64 = encode_image(processed_path)
        client = get_client()
        raw = client.call_sync(DOB_CORRECTION_PROMPT, image_b64, max_tokens=32)
        m = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", raw)
        if m:
            candidate = m.group(1)
            parts = candidate.split("/")
            day, month, year = parts[0], parts[1], parts[2]
            if day == "14":
                log.warning(f"DOB corrective pass still returned day '14': {candidate!r}")
                return None
            if int(day) < 1 or int(day) > 31:
                log.warning(f"DOB corrective pass returned invalid day: {candidate!r}")
                return None
            if int(month) < 1 or int(month) > 12:
                log.warning(f"DOB corrective pass returned invalid month: {candidate!r}")
                return None
            log.info(f"DOB corrective pass returned: {candidate}")
            return candidate
        log.warning(f"DOB corrective pass failed to find a date: {raw!r}")
        return None
    except Exception as e:
        log.warning(f"DOB corrective pass exception: {e}")
        return None
    finally:
        _cleanup_preprocessed(image_path, processed_path)

async def async_extract_dob_corrective(image_path: str) -> Optional[str]:
    """Async version of extract_dob_corrective."""
    return await asyncio.to_thread(extract_dob_corrective, image_path)

# ── Async support ────────────────────────────────────────────────────────────
_concurrency_sem = None
def _get_semaphore():
    global _concurrency_sem
    if _concurrency_sem is None:
        _concurrency_sem = asyncio.Semaphore(cfg.MAX_CONCURRENT)
    return _concurrency_sem

async def _async_extract_via_together(image_path: str) -> dict:
    health = await async_check_together_health()
    if not health["healthy"]:
        return {"success": False, "error": health["error"]}
    
    cache_key = image_hash(image_path) if cfg.ENABLE_CACHE else None
    if cache_key and (cached := _cache_get(cache_key)):
        return cached
    
    processed_path = await asyncio.to_thread(ultra_preprocess_image, image_path)
    image_b64 = await asyncio.to_thread(encode_image, processed_path)
    client = get_client()

    try:
        raw = await client.call_async(EXTRACTION_PROMPT, image_b64)
        data = _parse_json_response(raw)
        result = {"success": True, "data": data}
    except Exception as e:
        result = {"success": False, "error": str(e)}
    finally:
        _cleanup_preprocessed(image_path, processed_path)

    if cache_key and result["success"]:
        _cache_set(cache_key, result)
    return result

async def async_extract_license_data(image_path: str) -> dict:
    log.info(f"Async extracting from: {image_path}")
    try:
        validate_image(image_path)
    except ImageValidationError as e:
        return {"success": False, "error": str(e)}
    
    sem = _get_semaphore()
    async with sem:
        if cfg.VLM_BACKEND == "together":
            return await _async_extract_via_together(image_path)
        else:
            return {"success": False, "error": f"Unknown backend {cfg.VLM_BACKEND}"}

# ── Validation – only English, no Hijri checks ────────────────────────────────
_DATE_RE_DDMMYYYY = re.compile(r"^\d{2}/\d{2}/\d{4}$")
def _parse_date(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except ValueError:
        return None

def validate_extracted_data(data: dict) -> dict:
    issues = []
    warnings = []

    if not data.get("license_number"):
        issues.append("Missing license number")
    if not data.get("full_name_english") and not data.get("full_name_arabic"):
        issues.append("Missing name (both English and Arabic absent)")

    dob = data.get("date_of_birth")
    if dob:
        if not _DATE_RE_DDMMYYYY.match(dob):
            issues.append(f"Invalid date format for date_of_birth: '{dob}' (expected DD/MM/YYYY Gregorian)")
    else:
        warnings.append("No English DOB found – date_of_birth is null")

    exp = data.get("date_of_expiry")
    if exp:
        if not _DATE_RE_DDMMYYYY.match(exp):
            issues.append(f"Invalid date format for date_of_expiry: '{exp}' (expected DD/MM/YYYY Gregorian)")
        else:
            try:
                exp_dt = datetime.strptime(exp, "%d/%m/%Y")
                if exp_dt < datetime.now():
                    issues.append(f"License EXPIRED on {exp}")
                elif (exp_dt - datetime.now()).days <= 30:
                    warnings.append(f"License expires soon: {exp}")
            except:
                pass
    else:
        warnings.append("No English Exp found – date_of_expiry is null")

    doi = data.get("date_of_issue")
    if doi:
        if not _DATE_RE_DDMMYYYY.match(doi):
            issues.append(f"Invalid date format for date_of_issue: '{doi}' (expected DD/MM/YYYY Gregorian)")
    else:
        warnings.append("No English ISS/ISSN found – date_of_issue is null")

    for field in ("blood_group", "nationality", "category"):
        if not data.get(field):
            warnings.append(f"Field not extracted: {field}")

    return {"valid": len(issues) == 0, "issues": issues, "warnings": warnings}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--health":
        print(json.dumps(check_together_health(), indent=2))
        sys.exit(0)
    path = sys.argv[1] if len(sys.argv) > 1 else "test_license.jpg"
    result = extract_license_data(path)
    if result["success"]:
        validation = validate_extracted_data(result["data"])
        print(json.dumps({**result, "validation": validation}, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(1)