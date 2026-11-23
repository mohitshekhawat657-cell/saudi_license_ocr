"""
API-key verification dependency, backed by the shared `api_keys` table
maintained by the admin panel.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from urllib.parse import quote_plus

from fastapi import Header, HTTPException, status
from sqlalchemy import create_engine, text


def _db_url() -> str:
    user = os.getenv("DB_USER", "kalp_ocr")
    pwd = quote_plus(os.getenv("DB_PASSWORD", "Kalp@1234"))
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "ocr_results")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{name}"


_engine = create_engine(_db_url(), pool_pre_ping=True)

# This service is License OCR — accept keys scoped to 'license' or 'both'.
_ALLOWED_SERVICES = {"license", "both", "license_plate", "all"}


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> dict:
    """FastAPI dependency: validate the X-API-Key header against `api_keys`."""
    with _engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT k.api_key, k.expiry_date, k.is_active,
                       COALESCE(k.service, 'license') AS service,
                       a.app_name
                FROM ocr_admin.api_keys k
                JOIN ocr_admin.applications a ON a.id = k.app_id
                WHERE k.api_key = :k
                """
            ),
            {"k": x_api_key},
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

    if not row.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key revoked",
        )

    # expiry_date may be timezone-aware (TIMESTAMPTZ) or naive depending on
    # the DB column type. Normalise both sides to UTC before comparing.
    if row.expiry_date:
        expiry = row.expiry_date
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
    if row.expiry_date and expiry < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key expired",
        )

    # Enforce service scope — a registration-only key cannot hit License OCR.
    if row.service not in _ALLOWED_SERVICES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"API Key not authorized for License OCR (scope: {row.service})",
        )

    # Enforce monthly call limit
    with _engine.connect() as conn:
        limit_row = conn.execute(
            text("""
                SELECT COALESCE(u.monthly_limit, 10) AS monthly_limit,
                       COALESCE(u.calls_this_month, 0) AS calls_this_month
                FROM ocr_admin.api_keys k
                JOIN ocr_admin.users u ON u.username = k.created_by
                WHERE k.api_key = :k
            """),
            {"k": x_api_key},
        ).fetchone()
    if limit_row and limit_row.calls_this_month >= limit_row.monthly_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Monthly call limit exhausted ({limit_row.monthly_limit} calls). Please upgrade your plan.",
        )

    return {"api_key": row.api_key, "app_name": row.app_name, "service": row.service}
