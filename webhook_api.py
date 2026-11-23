"""
Webhook API
===========
Handles all webhook configuration, delivery, and management.
Exposes an APIRouter that is mounted into the main app.

Endpoints:
    GET  /webhook/config  — view current webhook configuration
    POST /webhook/config  — update webhook settings at runtime (no restart needed)
    POST /webhook/test    — send a test payload to the configured URL

Environment variables (initial defaults):
    WEBHOOK_URL      — destination URL (empty = webhooks disabled)
    WEBHOOK_SECRET   — value sent as X-Webhook-Secret header
    WEBHOOK_TIMEOUT  — seconds to wait for delivery (default 5)
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

log = logging.getLogger("license_api.webhook")

# ── Runtime-mutable config ─────────────────────────────────────────────────────
_webhook_config: dict = {
    "url":     os.getenv("WEBHOOK_URL", ""),
    "secret":  os.getenv("WEBHOOK_SECRET", ""),
    "timeout": int(os.getenv("WEBHOOK_TIMEOUT", "5")),
}

# ── Shared async HTTP client ───────────────────────────────────────────────────
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    global _client
    async with _client_lock:
        if _client is None:
            _client = httpx.AsyncClient(
                timeout=httpx.Timeout(_webhook_config["timeout"]),
                limits=httpx.Limits(max_keepalive_connections=10),
            )
    return _client


async def _reset_client() -> None:
    """Recreate the client (called when timeout changes)."""
    global _client
    async with _client_lock:
        if _client is not None:
            await _client.aclose()
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(_webhook_config["timeout"]),
            limits=httpx.Limits(max_keepalive_connections=10),
        )


async def close_client() -> None:
    """Call on app shutdown to release the connection pool."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ── Core delivery ──────────────────────────────────────────────────────────────

async def send_webhook(payload: dict, url: Optional[str] = None) -> None:
    """
    Send *payload* as JSON to a webhook URL.

    *url* — per-request callback URL (takes priority).
    Falls back to the global WEBHOOK_URL config when not provided.
    Silently logs and returns on any failure — never raises.
    """
    url = url or _webhook_config["url"]
    if not url:
        return
    try:
        client = await _get_client()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if _webhook_config["secret"]:
            headers["X-Webhook-Secret"] = _webhook_config["secret"]
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        log.info(f"Webhook delivered  url={url}  status={resp.status_code}")
    except httpx.RemoteProtocolError as exc:
        # Server closed the connection before sending the full response body.
        # This is common with ngrok / some proxies that send 200 OK but close
        # early. The webhook was delivered — treat as success if we got a 2xx.
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code is not None and 200 <= status_code < 300:
            log.info(f"Webhook delivered (truncated response)  url={url}  status={status_code}")
        else:
            log.warning(f"Webhook delivery failed (protocol error): {exc}")
    except Exception as exc:
        log.warning(f"Webhook delivery failed: {exc}")


# ── Router ─────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/webhook", tags=["Webhook"])


class WebhookConfigUpdate(BaseModel):
    url:     Optional[str] = None
    secret:  Optional[str] = None
    timeout: Optional[int] = None


@router.get("/config")
async def get_webhook_config():
    """
    View current webhook configuration.
    The URL is partially masked and only secret presence is shown.
    """
    url = _webhook_config["url"]
    masked_url = (
        url[:12] + "..." + url[-10:]
        if url and len(url) > 24
        else url
    )
    return {
        "enabled":         bool(url),
        "url":             masked_url or None,
        "secret_set":      bool(_webhook_config["secret"]),
        "timeout_seconds": _webhook_config["timeout"],
    }


@router.post("/config")
async def update_webhook_config(body: WebhookConfigUpdate):
    """
    Update webhook settings at runtime — no server restart needed.

    Only the fields you provide are updated. Example:

        { "url": "https://your-server.com/hook", "secret": "my-secret", "timeout": 10 }
    """
    timeout_changed = body.timeout is not None and body.timeout != _webhook_config["timeout"]

    if body.url is not None:
        _webhook_config["url"] = body.url
    if body.secret is not None:
        _webhook_config["secret"] = body.secret
    if body.timeout is not None:
        _webhook_config["timeout"] = body.timeout

    if timeout_changed:
        await _reset_client()

    url = _webhook_config["url"]
    masked_url = (
        url[:12] + "..." + url[-10:]
        if url and len(url) > 24
        else url
    )
    return {
        "success":         True,
        "enabled":         bool(url),
        "url":             masked_url or None,
        "secret_set":      bool(_webhook_config["secret"]),
        "timeout_seconds": _webhook_config["timeout"],
    }


@router.post("/test")
async def webhook_test():
    """
    Send a test payload to the configured webhook URL.
    Returns delivery result including the HTTP status code received.
    """
    url = _webhook_config["url"]
    if not url:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error":   "Webhook URL is not configured. Use POST /webhook/config to set it.",
            },
        )

    test_payload = {
        "event":     "webhook.test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message":   "This is a test webhook from the Saudi License Verification API.",
    }

    try:
        client = await _get_client()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if _webhook_config["secret"]:
            headers["X-Webhook-Secret"] = _webhook_config["secret"]
        resp = await client.post(url, json=test_payload, headers=headers)
        resp.raise_for_status()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success":     True,
                "url":         url,
                "status_code": resp.status_code,
                "message":     "Test payload delivered.",
            },
        )
    except httpx.HTTPStatusError as exc:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "success":     False,
                "url":         url,
                "status_code": exc.response.status_code,
                "error":       str(exc),
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "success": False,
                "url":     url,
                "error":   str(exc),
            },
        )
