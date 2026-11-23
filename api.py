"""
Saudi Driving License — Verification API
=========================================
Single-file FastAPI server. Lives in the same folder as pipeline.py
and ocr_extractor.py — no path tricks needed.

Endpoint:
    POST /verify
    Form fields:
        type  (str)        — document type, e.g. "driving_license"
        id    (str)        — your reference / user ID
        file  (UploadFile) — license image

Run:
    uvicorn api:app --host 0.0.0.0 --port 3000 --reload
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, List, Optional

import aiofiles
import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from api_key_auth import verify_api_key

# ── Load env from same directory ─────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env.example")

# ── Import pipeline — same folder, no sys.path needed ────────────────────────
from pipeline import (
    LicenseAuthPipeline,
    async_process_one,
    async_process_batch,
    find_model,
    PipelineResult,
)
from ocr_extractor import (
    Config as OCRConfig,
    async_check_together_health,
    clear_cache,
    close_async_client,
)
from webhook_api import router as webhook_router, send_webhook, close_client as close_webhook_client, _webhook_config
from database import init_db, insert_license_result, close_db, is_db_connected

# ── Config ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH: str | None = os.getenv("YOLO_MODEL_PATH", None)
ADMIN_WEBHOOK_URL = os.getenv("ADMIN_WEBHOOK_URL", "http://localhost:8001/api/ocr/webhook").strip()
ADMIN_WEBHOOK_SECRET = os.getenv("OCR_WEBHOOK_SECRET", "").strip()

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

JSON_RESPONSE_DIR = Path(__file__).parent / "json_response"
JSON_RESPONSE_DIR.mkdir(exist_ok=True)

BATCH_MAX_FILES: int = int(os.getenv("BATCH_MAX_FILES", "2000"))
BATCH_WORKERS:   int = int(os.getenv("BATCH_WORKERS",   "8"))   # maps to max_concurrent in async batch

ALLOWED_EXTS = set(OCRConfig.ALLOWED_EXTS)

_EMPTY_LICENSE_DATA = {
    "license_number":    None,
    "full_name_english": None,
    "full_name_arabic":  None,
    "date_of_birth":     None,
    "date_of_issue":     None,
    "date_of_expiry":    None,
    "category":          None,
    "nationality":       None,
    "blood_group":       None,
    "restrictions":      None,
    "version":           None,
    "issuing_country":   None,
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("license_api")

# Global semaphore to limit concurrent API requests (per instance)
_GLOBAL_API_SEMAPHORE = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_REQUESTS", "10")))

def _safe_name(value: str | None) -> str:
    """Sanitize a string for safe use as a filename (alnum, dash, underscore)."""
    if not value:
        return ""
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(value).strip())
    return cleaned.strip("_") or ""


def _compress_to_under_2mb(image_path: Path) -> None:
    """If the saved image is ≥ 2 MB, recompress it in-place until it is under 2 MB.
    Works by progressively lowering JPEG quality (95 → 85 → 75 → 65 → 55).
    If the file is not a JPEG-compatible format, converts to JPEG first.
    No-op when the file is already < 2 MB.
    """
    TARGET_BYTES = 2 * 1024 * 1024  # 2 MB

    if image_path.stat().st_size < TARGET_BYTES:
        return  # already small enough

    try:
        from PIL import Image, ImageOps
        import io

        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")

        out_path = image_path.with_suffix(".jpg")

        for quality in (85, 75, 65, 55, 45):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            size = buf.tell()
            if size < TARGET_BYTES:
                out_path.write_bytes(buf.getvalue())
                # Replace original with compressed version
                if out_path != image_path:
                    image_path.unlink(missing_ok=True)
                log.info(
                    f"Compressed {image_path.name} → {size/1024/1024:.2f} MB  (quality={quality})"
                )
                return

        # Last resort: resize to half dimensions then save
        h, w = img.height, img.width
        img_small = img.resize((w // 2, h // 2), Image.LANCZOS)
        buf = io.BytesIO()
        img_small.save(buf, format="JPEG", quality=60, optimize=True)
        out_path.write_bytes(buf.getvalue())
        if out_path != image_path:
            image_path.unlink(missing_ok=True)
        log.info(
            f"Compressed (resized) {image_path.name} → {buf.tell()/1024/1024:.2f} MB"
        )
    except Exception as e:
        log.warning(f"Image compression failed for {image_path.name}: {e}")


def _has_issues(data: dict) -> bool:
    """Return True if the payload (single or batch) contains any issue.

    Issue criteria (warnings are NOT counted as issues):
      - success is False
      - resultData.isValid is False
      - resultData.issue is non-empty
      - validation.valid is False, or validation.issues non-empty
      - top-level issue is non-empty
      - (batch) any item in results[] matches the above
    """
    if not isinstance(data, dict):
        return False

    def _check_single(d: dict) -> bool:
        if not isinstance(d, dict):
            return False
        if d.get("success") is False:
            return True
        if d.get("issue"):
            return True
        rd = d.get("resultData") or {}
        if isinstance(rd, dict):
            if rd.get("isValid") is False:
                return True
            if rd.get("issue"):
                return True
        val = d.get("validation") or {}
        if isinstance(val, dict):
            if val.get("valid") is False:
                return True
            if val.get("issues"):
                return True
        return False

    if _check_single(data):
        return True
    for item in data.get("results") or []:
        if _check_single(item):
            return True
    return False


async def _save_json(filename: str, payload: dict) -> None:
    """Persist the response payload as a JSON file in json_response/ (async).

    Only saves payloads that contain issues; clean responses are skipped.
    """
    if not _has_issues(payload):
        log.info(f"JSON skipped (clean) → {filename}")
        return
    try:
        import json as _json
        out = JSON_RESPONSE_DIR / filename
        async with aiofiles.open(out, "w", encoding="utf-8") as f:
            await f.write(_json.dumps(payload, indent=2, ensure_ascii=False))
        log.info(f"JSON saved → {out.name}")
    except Exception as e:
        log.warning(f"Could not save JSON response: {e}")


async def _send_admin_log(api_key: str, service: str, http_status: int, response_ms: int) -> None:
    """Fire-and-forget: log this call to the admin panel backend."""
    if not ADMIN_WEBHOOK_URL:
        return
    headers = {"Content-Type": "application/json"}
    if ADMIN_WEBHOOK_SECRET:
        headers["X-Webhook-Secret"] = ADMIN_WEBHOOK_SECRET
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(ADMIN_WEBHOOK_URL, json={
                "api_key":     api_key,
                "service":     service,
                "status_code": http_status,
                "response_ms": response_ms,
            }, headers=headers)
    except Exception as exc:
        log.warning(f"Admin log delivery failed ({ADMIN_WEBHOOK_URL}): {exc}")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Saudi Driving License Verification API",
    description=(
        "Upload a license image with `type`, `id`, `file`. "
        "Returns extracted fields as JSON after running "
        "YOLO detection → crop → VLM OCR → validation."
    ),
    version="1.0.0",
    docs_url=None,   # disable default light Swagger UI
    redoc_url=None,
)

app.include_router(webhook_router)

# ── Custom dark-theme Swagger UI ─────────────────────────────────────────────
from fastapi.responses import HTMLResponse, JSONResponse as _FRJSONResponse

_DARK_SWAGGER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>__TITLE__ — API Docs</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
  <link rel="icon" type="image/png" sizes="32x32" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/favicon-32x32.png" />
  <link rel="icon" type="image/png" sizes="16x16" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/favicon-16x16.png" />
  <style>
    /* ── Dark theme overrides for Swagger UI ──────────────────────────── */
    html, body { background:#0d1117 !important; color:#e6edf3 !important; }
    body { margin:0; }
    #swagger-ui { filter: none; }
    .swagger-ui, .swagger-ui .info, .swagger-ui .scheme-container,
    .swagger-ui .opblock-tag, .swagger-ui .opblock, .swagger-ui .opblock .opblock-summary,
    .swagger-ui .opblock-description-wrapper, .swagger-ui .opblock-section-header,
    .swagger-ui section.models, .swagger-ui .model-box, .swagger-ui table,
    .swagger-ui .responses-inner, .swagger-ui .response, .swagger-ui .parameters-col_description,
    .swagger-ui .parameter__name, .swagger-ui .parameter__type, .swagger-ui .parameter__in,
    .swagger-ui label, .swagger-ui select, .swagger-ui input, .swagger-ui textarea {
        color:#e6edf3 !important;
    }
    .swagger-ui .topbar { background:#010409; border-bottom:1px solid #30363d; }
    .swagger-ui .info .title, .swagger-ui .info .title small,
    .swagger-ui .info .title small pre { color:#58a6ff !important; }
    .swagger-ui .info, .swagger-ui .scheme-container { background:transparent !important; box-shadow:none !important; }
    .swagger-ui .opblock-tag { color:#58a6ff !important; border-bottom:1px solid #30363d; }
    .swagger-ui .opblock { background:#161b22 !important; border:1px solid #30363d !important; box-shadow:none !important; }
    .swagger-ui .opblock .opblock-summary { border-color:#30363d !important; }
    .swagger-ui .opblock-section-header { background:#0d1117 !important; box-shadow:none !important; }
    .swagger-ui .opblock.opblock-post   { border-color:#238636 !important; }
    .swagger-ui .opblock.opblock-post   .opblock-summary-method { background:#238636 !important; }
    .swagger-ui .opblock.opblock-get    { border-color:#1f6feb !important; }
    .swagger-ui .opblock.opblock-get    .opblock-summary-method { background:#1f6feb !important; }
    .swagger-ui .opblock.opblock-put    { border-color:#9e6a03 !important; }
    .swagger-ui .opblock.opblock-put    .opblock-summary-method { background:#9e6a03 !important; }
    .swagger-ui .opblock.opblock-delete { border-color:#da3633 !important; }
    .swagger-ui .opblock.opblock-delete .opblock-summary-method { background:#da3633 !important; }
    .swagger-ui section.models { background:#161b22 !important; border:1px solid #30363d !important; }
    .swagger-ui section.models h4, .swagger-ui .model-title, .swagger-ui .model { color:#e6edf3 !important; }
    .swagger-ui .model-box { background:#0d1117 !important; }
    .swagger-ui table thead tr th, .swagger-ui table tbody tr td { border-color:#30363d !important; }
    .swagger-ui input[type=text], .swagger-ui input[type=email], .swagger-ui input[type=password],
    .swagger-ui textarea, .swagger-ui select {
        background:#0d1117 !important; border:1px solid #30363d !important; color:#e6edf3 !important;
    }
    .swagger-ui .btn { background:#21262d !important; color:#e6edf3 !important; border:1px solid #30363d !important; }
    .swagger-ui .btn.execute { background:#1f6feb !important; border-color:#1f6feb !important; color:#fff !important; }
    .swagger-ui .btn.authorize { background:#238636 !important; border-color:#238636 !important; color:#fff !important; }
    .swagger-ui .highlight-code, .swagger-ui pre, .swagger-ui code,
    .swagger-ui .microlight, .swagger-ui .responses-inner pre {
        background:#010409 !important; color:#e6edf3 !important;
    }
    .swagger-ui .tab li { color:#e6edf3 !important; }
    .swagger-ui a { color:#58a6ff !important; }
    .swagger-ui .response-col_status { color:#7ee787 !important; }
    .swagger-ui .opblock-summary-path, .swagger-ui .opblock-summary-description { color:#e6edf3 !important; }
    .swagger-ui svg { fill:#e6edf3 !important; }
    /* File input — match dark theme */
    .swagger-ui input[type=file] {
        background:#0d1117 !important;
        color:#e6edf3 !important;
        border:1px solid #30363d !important;
        border-radius:4px;
        padding:4px;
    }
    .swagger-ui input[type=file]::file-selector-button {
        background:#21262d !important;
        color:#e6edf3 !important;
        border:1px solid #30363d !important;
        border-radius:4px;
        padding:4px 10px;
        margin-right:8px;
        cursor:pointer;
    }
    .swagger-ui input[type=file]::file-selector-button:hover {
        background:#30363d !important;
    }
    /* WebKit fallback */
    .swagger-ui input[type=file]::-webkit-file-upload-button {
        background:#21262d !important;
        color:#e6edf3 !important;
        border:1px solid #30363d !important;
        border-radius:4px;
        padding:4px 10px;
        margin-right:8px;
        cursor:pointer;
    }
    /* Schemas — kill the light pills around model names / Expand all */
    .swagger-ui .model-box,
    .swagger-ui .model-title,
    .swagger-ui .model-title__text,
    .swagger-ui section.models .model-toggle,
    .swagger-ui section.models button,
    .swagger-ui section.models .pointer,
    .swagger-ui section.models a {
        background:transparent !important;
        background-color:transparent !important;
        color:#e6edf3 !important;
        box-shadow:none !important;
    }
    .swagger-ui section.models .model-toggle,
    .swagger-ui section.models button.expand-operation,
    .swagger-ui section.models .models-control {
        border:1px solid #30363d !important;
        border-radius:4px !important;
        padding:2px 8px !important;
        color:#8b949e !important;
    }
    .swagger-ui .model .prop-type,
    .swagger-ui .model .prop-format { color:#d2a8ff !important; }
    .swagger-ui .model .prop-name { color:#7ee787 !important; }
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = () => {
      window.ui = SwaggerUIBundle({
        url: "__OPENAPI_URL__",
        dom_id: "#swagger-ui",
        deepLinking: true,
        layout: "BaseLayout",
        docExpansion: "list",
        defaultModelsExpandDepth: 1,
      });
    };
  </script>
</body>
</html>"""


@app.get("/docs", include_in_schema=False)
async def custom_swagger_dark() -> HTMLResponse:
    html = _DARK_SWAGGER_HTML.replace("__TITLE__", app.title).replace("__OPENAPI_URL__", app.openapi_url or "/openapi.json")
    return HTMLResponse(html)

# ── Singleton pipeline — YOLO model loaded once at startup ────────────────────
_pipeline: LicenseAuthPipeline | None = None

def get_pipeline() -> LicenseAuthPipeline:
    global _pipeline
    if _pipeline is None:
        model_path = YOLO_MODEL_PATH or find_model()
        if not model_path:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "success": False,
                    "error": (
                        "YOLO model not found. "
                        "Run 'python train.py' to train it, or set "
                        "YOLO_MODEL_PATH in .env.example"
                    ),
                },
            )
        log.info(f"Loading YOLO model: {model_path}")
        _pipeline = LicenseAuthPipeline(model_path=model_path)
    return _pipeline


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def on_startup():
    log.info("API starting up…")
    clear_cache()   # always flush on restart so prompt changes take effect

    model_path = YOLO_MODEL_PATH or find_model()
    if model_path:
        log.info(f"YOLO model found: {model_path} — loading into memory now…")
        try:
            get_pipeline()
            log.info("YOLO model loaded and ready.")
        except Exception as exc:
            log.warning(f"YOLO model load failed at startup: {exc}")
    else:
        log.warning("YOLO model not found — /verify will return 503 until model is ready.")

    # Check Together AI health
    if OCRConfig.VLM_BACKEND == "together":
        h = await async_check_together_health()
        if h["healthy"]:
            log.info(f"Together API OK — model={OCRConfig.TOGETHER_MODEL}")
        else:
            log.warning(f"Together API not reachable: {h['error']}")
    else:
        log.warning(f"Unknown VLM_BACKEND: {OCRConfig.VLM_BACKEND}")

    # Initialize database connection pool
    await init_db()


@app.on_event("shutdown")
async def on_shutdown():
    log.info("Shutting down API, closing HTTP clients...")
    await close_async_client()
    await close_webhook_client()
    await close_db()


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "Saudi Driving License Verification API"}


@app.get("/health", tags=["Health"])
async def health():
    """YOLO model + VLM backend status."""
    model_path = YOLO_MODEL_PATH or find_model()
    yolo_ok    = bool(model_path and Path(model_path).exists())

    raw_vlm: dict[str, Any] = {}
    if OCRConfig.VLM_BACKEND == "together":
        raw_vlm = await async_check_together_health()
    vlm: dict[str, Any] = {
        "backend":      OCRConfig.VLM_BACKEND,
        "healthy":      bool(raw_vlm.get("healthy")),
        "model_loaded": bool(raw_vlm.get("model_loaded")),
        "model":        OCRConfig.TOGETHER_MODEL,
        "error":        raw_vlm.get("error"),
    }

    db_connected = await is_db_connected()

    all_ok = yolo_ok and bool(vlm.get("healthy")) and db_connected

    return {
        "status":     "connected" if all_ok else "not_connected",
        "yolo_model": {"found": yolo_ok, "path": model_path},
        "vlm":        vlm,
        "database":   {"connected": db_connected},
    }


@app.post("/verify", tags=["Verification"])
async def verify_license(
    type: str               = Form(..., description="Document type, e.g. 'driving_license'"),
    driverId: str           = Form(..., description="Your reference or user ID"),
    file: UploadFile        = File(..., description="License image (jpg/png/webp/heic/tiff/bmp/pdf)"),
    webhook_url: Optional[str] = Form(None, description="Per-request callback URL (overrides global WEBHOOK_URL)"),
    _api: dict             = Depends(verify_api_key),
):
    """
    Verify a Saudi driving license.

    **Form parameters:**
    - `type` — document type string
    - `id`   — your reference ID
    - `file` — image file

    **Returns:** extracted license fields + validation.
    """
    request_id = str(uuid.uuid4())
    effective_webhook = webhook_url or _webhook_config.get("url", "")

    # ── ASYNC MODE: webhook_url provided → return instantly, process in background ──
    if effective_webhook:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"success": False, "request_id": request_id,
                        "issue": [f"Unsupported file type '{suffix}'."]},
            )
        temp_path = UPLOAD_DIR / f"{request_id}{suffix}"
        content = await file.read()
        async with aiofiles.open(temp_path, "wb") as out:
            await out.write(content)
        orig_filename = file.filename or ""

        async def _bg():
            async with _GLOBAL_API_SEMAPHORE:
                try:
                    await asyncio.to_thread(_compress_to_under_2mb, temp_path)
                    actual = temp_path if temp_path.exists() else temp_path.with_suffix(".jpg")
                    pipeline = get_pipeline()
                    result = await async_process_one(str(actual), pipeline)
                    validation = result.validation or {}
                    if result.success:
                        payload = {
                            "success": True, "type": type, "total": 1,
                            "succeeded": 1, "failed": 0, "index": 0,
                            "filename": orig_filename,
                            "detection_confidence": round(result.detection_confidence, 4),
                            "resultData": {
                                "license_data": result.data,
                                "isValid": validation.get("valid", False),
                                "sequence": "0",
                                "issue": validation.get("issues", []),
                                "driverId": driverId,
                                "warnings": validation.get("warnings", []),
                                "duration_ms": round(result.duration_ms, 1),
                            },
                        }
                    else:
                        payload = {
                            "success": False, "type": type, "total": 1,
                            "succeeded": 0, "failed": 1, "index": 0,
                            "filename": orig_filename,
                            "detection_confidence": None,
                            "resultData": {
                                "license_data": _EMPTY_LICENSE_DATA,
                                "isValid": False, "sequence": "0",
                                "issue": [result.error] if result.error else [],
                                "driverId": driverId, "warnings": [],
                                "duration_ms": round(result.duration_ms, 1),
                            },
                        }
                    await _save_json(f"{_safe_name(driverId) or request_id}.json", payload)
                    await send_webhook({
                        "event": "verify.completed",
                        "request_id": request_id,
                        "status": "success" if result.success else "failed",
                        "result": payload,
                    }, url=effective_webhook)
                    http_status = 200 if result.success else 422
                    await _send_admin_log(_api["api_key"], "license", http_status, round(result.duration_ms))

                    # Database insert (mirrors the sync path below)
                    _rd = payload["resultData"]
                    await insert_license_result(
                        request_id=request_id,
                        document_type=type,
                        total=payload["total"],
                        succeeded=payload["succeeded"],
                        failed=payload["failed"],
                        index=payload["index"],
                        filename=orig_filename,
                        success=result.success,
                        detection_confidence=round(result.detection_confidence, 4) if result.success else None,
                        data=_rd["license_data"] or {},
                        is_valid=_rd["isValid"],
                        sequence=_rd["sequence"],
                        issue=_rd["issue"],
                        driver_id=driverId,
                        warnings=_rd["warnings"],
                        duration_ms=result.duration_ms,
                    )
                except Exception as e:
                    log.exception(f"[{request_id}] Background processing error")
                    await send_webhook({
                        "event": "verify.failed",
                        "request_id": request_id,
                        "status": "failed",
                        "error": str(e),
                    }, url=effective_webhook)
                    await _send_admin_log(_api["api_key"], "license", 500, 0)
                finally:
                    temp_path.unlink(missing_ok=True)
                    (UPLOAD_DIR / f"{request_id}_cropped.jpg").unlink(missing_ok=True)

        asyncio.create_task(_bg())
        return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content={
            "success": True,
            "request_id": request_id,
            "message": "Request received. Result will be sent to your webhook URL when processing is complete.",
            "webhook_url": effective_webhook,
        })

    # ── SYNC MODE: no webhook_url → wait for result (original behaviour) ──────
    async with _GLOBAL_API_SEMAPHORE:
        # 1. Validate extension
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "success":              False,
                    "type":                 type,
                    "total":                1,
                    "succeeded":            0,
                    "failed":               1,
                    "index":                0,
                    "filename":             file.filename or "",
                    "detection_confidence": None,
                    "resultData": {
                        "license_data": _EMPTY_LICENSE_DATA,
                        "isValid":      False,
                        "sequence":     "0",
                        "issue":        [f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTS)}"],
                        "driverId":     driverId,
                        "warnings":     [],
                        "duration_ms":  0,
                    },
                },
            )

        # 2. Save to uploads/ asynchronously
        temp_path = UPLOAD_DIR / f"{request_id}{suffix}"
        try:
            async with aiofiles.open(temp_path, "wb") as out:
                content = await file.read()
                await out.write(content)
            log.info(f"[{request_id}] Saved upload → {temp_path.name}  (type={type}, driverId={driverId})")
        except Exception as e:
            log.error(f"[{request_id}] Save failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "success":              False,
                    "type":                 type,
                    "total":                1,
                    "succeeded":            0,
                    "failed":               1,
                    "index":                0,
                    "filename":             file.filename or "",
                    "detection_confidence": None,
                    "resultData": {
                        "license_data": _EMPTY_LICENSE_DATA,
                        "isValid":      False,
                        "sequence":     "0",
                        "issue":        ["Failed to save file."],
                        "driverId":     driverId,
                        "warnings":     [],
                        "duration_ms":  0,
                    },
                },
            )

        # 2b. Compress to under 2 MB if needed (protects server RAM)
        await asyncio.to_thread(_compress_to_under_2mb, temp_path)
        # Refresh temp_path in case extension changed to .jpg
        if not temp_path.exists():
            temp_path = temp_path.with_suffix(".jpg")

        # 3. Run async pipeline
        try:
            pipeline = get_pipeline()
            result = await async_process_one(str(temp_path), pipeline)
        except HTTPException:
            raise
        except Exception as e:
            log.exception(f"[{request_id}] Pipeline error")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "success":              False,
                    "type":                 type,
                    "total":                1,
                    "succeeded":            0,
                    "failed":               1,
                    "index":                0,
                    "filename":             file.filename or "",
                    "detection_confidence": None,
                    "resultData": {
                        "license_data": _EMPTY_LICENSE_DATA,
                        "isValid":      False,
                        "sequence":     "0",
                        "issue":        [f"Pipeline error: {e}"],
                        "driverId":     driverId,
                        "warnings":     [],
                        "duration_ms":  0,
                    },
                },
            )
        finally:
            # Clean up temp files
            temp_path.unlink(missing_ok=True)
            (UPLOAD_DIR / f"{request_id}_cropped.jpg").unlink(missing_ok=True)

        # 4. Return response
        if result.success:
            validation = result.validation or {}
            log.info(
                f"[{request_id}] ✓ success  "
                f"conf={result.detection_confidence:.2f}  "
                f"valid={validation.get('valid', 'n/a')}"
            )
            payload = {
                "success":              True,
                "type":                 type,
                "total":                1,
                "succeeded":            1,
                "failed":               0,
                "index":                0,
                "filename":             file.filename or "",
                "detection_confidence": round(result.detection_confidence, 4),
                "resultData": {
                    "license_data": result.data,
                    "isValid":      validation.get("valid", False),
                    "sequence":     "0",
                    "issue":        validation.get("issues", []),
                    "driverId":     driverId,
                    "warnings":     validation.get("warnings", []),
                    "duration_ms":  round(result.duration_ms, 1),
                },
            }
            await _save_json(f"{_safe_name(driverId) or request_id}.json", payload)

            # Insert into database (fire and forget)
            asyncio.create_task(
                insert_license_result(
                    request_id=request_id,
                    document_type=type,
                    total=1,
                    succeeded=1,
                    failed=0,
                    index=0,
                    filename=file.filename or "",
                    success=True,
                    detection_confidence=result.detection_confidence,
                    data=result.data or {},
                    is_valid=validation.get("valid", False),
                    sequence="0",
                    issue=validation.get("issues", []),
                    driver_id=driverId,
                    warnings=validation.get("warnings", []),
                    duration_ms=result.duration_ms,
                )
            )

            # Send webhook (fire and forget)
            asyncio.create_task(send_webhook({
                "event":      "verify.completed",
                "timestamp":  result.timestamp,
                "type":       type,
                "request_id": request_id,
                "status":     "success",
                "result":     payload,
            }, url=effective_webhook))
            asyncio.create_task(_send_admin_log(_api["api_key"], "license", 200, round(result.duration_ms)))

            return JSONResponse(status_code=status.HTTP_200_OK, content=payload)

        # Map stage → HTTP status
        http_code = {
            "input":     status.HTTP_422_UNPROCESSABLE_ENTITY,
            "detection": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "quality":   status.HTTP_422_UNPROCESSABLE_ENTITY,
            "ocr":       status.HTTP_500_INTERNAL_SERVER_ERROR,
        }.get(result.stage or "", status.HTTP_400_BAD_REQUEST)

        log.warning(f"[{request_id}] ✗ stage={result.stage}  error={result.error}")
        payload = {
            "success":              False,
            "type":                 type,
            "total":                1,
            "succeeded":            0,
            "failed":               1,
            "index":                0,
            "filename":             file.filename or "",
            "detection_confidence": None,
            "resultData": {
                "license_data": _EMPTY_LICENSE_DATA,
                "isValid":      False,
                "sequence":     "0",
                "issue":        [result.error] if result.error else [],
                "driverId":     driverId,
                "warnings":     [],
                "duration_ms":  round(result.duration_ms, 1),
            },
        }
        await _save_json(f"{_safe_name(driverId) or request_id}.json", payload)

        # Insert into database (fire and forget)
        asyncio.create_task(
            insert_license_result(
                request_id=request_id,
                document_type=type,
                total=1,
                succeeded=0,
                failed=1,
                index=0,
                filename=file.filename or "",
                success=False,
                detection_confidence=None,
                data={},
                is_valid=False,
                sequence="0",
                issue=[result.error] if result.error else [],
                driver_id=driverId,
                warnings=[],
                duration_ms=result.duration_ms,
            )
        )

        # Send webhook for failure
        asyncio.create_task(send_webhook({
            "event":      "verify.completed",
            "timestamp":  result.timestamp,
            "type":       type,
            "request_id": request_id,
            "status":     "failed",
            "result":     payload,
        }, url=effective_webhook))
        asyncio.create_task(_send_admin_log(_api["api_key"], "license", http_code, round(result.duration_ms)))

        return JSONResponse(status_code=http_code, content=payload)


# =============================================================================
# BATCH ENDPOINT
# =============================================================================

def _process_batch_item(
    temp_path: Path,
    request_id: str,
    file_index: int,
    original_name: str,
) -> dict:
    """Run pipeline on one file synchronously (called from thread pool)."""
    # This function is kept for compatibility but not used directly in async batch.
    # Instead, we use async_process_batch which internally handles concurrency.
    # We keep it to preserve original line count and structure.
    pass  # Not used, but kept for original code structure


@app.post("/verify-batch", tags=["Verification"])
async def verify_license_batch(
    type:  str                 = Form(..., description="Document type, e.g. 'driving_license'"),
    driverId: str              = Form(..., description="Your batch reference / user ID"),
    files: List[UploadFile]    = File(..., description="License images (up to BATCH_MAX_FILES)"),
    webhook_url: Optional[str] = Form(None, description="Per-request callback URL (overrides global WEBHOOK_URL)"),
    _api: dict                 = Depends(verify_api_key),
):
    """
    Verify multiple Saudi driving license images in one request.

    **Form parameters:**
    - `type`  — document type string
    - `id`    — your reference ID for the batch
    - `files` — one or more image files (jpg/png/webp/heic/tiff/bmp/pdf)

    **Returns:** single JSON array of results, one entry per file, ordered by
    original upload index.
    """
    if len(files) > BATCH_MAX_FILES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "success":   False,
                "type":      type,
                "total":     len(files),
                "succeeded": 0,
                "failed":    len(files),
                "isValid":   False,
                "sequence":  "0",
                "driverId":  driverId,
                "issue":     [f"Too many files: {len(files)} > limit {BATCH_MAX_FILES}"],
                "warnings":  [],
                "results":   [],
            },
        )

    batch_id = str(uuid.uuid4())
    log.info(f"[batch:{batch_id}] Received {len(files)} files  (type={type}, driverId={driverId})")

    # ── Save all uploads to disk asynchronously ──────────────────────────────
    saved: List[tuple[int, Path | None, str]] = []   # (index, temp_path, original_name)
    for i, upload in enumerate(files):
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTS:
            saved.append((i, None, upload.filename or ""))
            continue
        temp_path = UPLOAD_DIR / f"{batch_id}_{i}{suffix}"
        try:
            async with aiofiles.open(temp_path, "wb") as out:
                content = await upload.read()
                await out.write(content)
            # Compress to under 2 MB if needed
            await asyncio.to_thread(_compress_to_under_2mb, temp_path)
            if not temp_path.exists():
                temp_path = temp_path.with_suffix(".jpg")
            saved.append((i, temp_path, upload.filename or ""))
        except Exception as e:
            log.error(f"[batch:{batch_id}] file[{i}] save failed: {e}")
            saved.append((i, None, upload.filename or ""))

    # ── Process valid files using async batch pipeline ──────────────────────
    pipeline = get_pipeline()
    valid_items = [(i, p, n) for i, p, n in saved if p is not None]
    if not valid_items:
        results = []
    else:
        indices, paths, names = zip(*valid_items)
        # Use BATCH_WORKERS to control concurrency (mapped to max_concurrent)
        max_conc = BATCH_WORKERS if BATCH_WORKERS > 0 else int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        results_list, _ = await async_process_batch(
            [str(p) for p in paths],
            pipeline,
            max_concurrent=max_conc,
        )
        # Map results back to original indices, preserving original names
        results_with_index = []
        for idx, path, name, res in zip(indices, paths, names, results_list):
            res.image_path = name   # keep original name for output
            results_with_index.append((idx, res))
        # Add failed entries for unsaved files
        for i, p, n in saved:
            if p is None:
                fake_res = PipelineResult(
                    image_path=n,
                    success=False,
                    error="Failed to save file",
                    stage="input",
                    duration_ms=0,
                )
                results_with_index.append((i, fake_res))
        # Sort by original index
        results = [r for _, r in sorted(results_with_index, key=lambda x: x[0])]

    succeeded = sum(1 for r in results if r.success)
    log.info(f"[batch:{batch_id}] Done — {succeeded}/{len(results)} succeeded")

    # Build response payload and send webhook per file
    response_results = []
    for idx, r in enumerate(results):
        validation = r.validation or {}
        if r.success:
            item = {
                "index":                idx,
                "filename":             r.image_path,
                "success":              True,
                "detection_confidence": round(r.detection_confidence, 4),
                "resultData": {
                    "license_data": r.data,
                    "isValid":      validation.get("valid", False),
                    "sequence":     "0",
                    "issue":        validation.get("issues", []),
                    "driverId":     driverId,
                    "warnings":     validation.get("warnings", []),
                    "duration_ms":  round(r.duration_ms, 1),
                },
            }
            status_str = "success"
        else:
            item = {
                "index":                idx,
                "filename":             r.image_path,
                "success":              False,
                "detection_confidence": None,
                "resultData": {
                    "license_data": _EMPTY_LICENSE_DATA,
                    "isValid":      False,
                    "sequence":     "0",
                    "issue":        [r.error] if r.error else [],
                    "driverId":     driverId,
                    "warnings":     [],
                    "duration_ms":  round(r.duration_ms, 1),
                },
            }
            status_str = "failed"
        response_results.append(item)

        # Insert into database (fire and forget)
        asyncio.create_task(
            insert_license_result(
                request_id=batch_id,
                document_type=type,
                total=len(results),
                succeeded=succeeded,
                failed=len(results) - succeeded,
                index=idx,
                filename=r.image_path,
                success=r.success,
                detection_confidence=r.detection_confidence if r.success else None,
                data=r.data or {},
                is_valid=validation.get("valid", False),
                sequence="0",
                issue=validation.get("issues", []) if r.success else ([r.error] if r.error else []),
                driver_id=driverId,
                warnings=validation.get("warnings", []),
                duration_ms=r.duration_ms,
            )
        )

        # Send webhook for each file (fire and forget)
        asyncio.create_task(send_webhook({
            "event":      "verify.batch.file",
            "timestamp":  r.timestamp,
            "type":       type,
            "request_id": batch_id,
            "file_index": idx,
            "status":     status_str,
            "result":     item,
        }, url=webhook_url))

    payload = {
        "success":   True,
        "type":      type,
        "total":     len(results),
        "succeeded": succeeded,
        "failed":    len(results) - succeeded,
        "results":   response_results,
    }
    await _save_json(f"{_safe_name(driverId) or batch_id}.json", payload)
    return JSONResponse(status_code=status.HTTP_200_OK, content=payload)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=3000, reload=True)