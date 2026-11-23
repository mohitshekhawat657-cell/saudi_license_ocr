"""
Database module
===============
Async PostgreSQL integration using asyncpg.

Reads connection settings from environment variables:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    DB_POOL_MIN_SIZE, DB_POOL_MAX_SIZE

Exposed functions:
    init_db()                  — create connection pool + ensure table exists
    insert_license_result(...) — insert one verification result (fire-and-forget safe)
    close_db()                 — close the connection pool on shutdown
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

log = logging.getLogger("license_api.database")

_pool = None  # asyncpg.Pool


async def init_db() -> None:
    """Create the asyncpg connection pool and ensure the results table exists."""
    global _pool
    try:
        import asyncpg  # type: ignore
    except ImportError:
        log.warning("asyncpg not installed — database persistence disabled.")
        return

    host     = os.getenv("DB_HOST", "localhost")
    port     = int(os.getenv("DB_PORT", "5432"))
    database = os.getenv("DB_NAME", "ocr_results")
    user     = os.getenv("DB_USER", "kalp_ocr")
    password = os.getenv("DB_PASSWORD", "")
    min_size = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
    max_size = int(os.getenv("DB_POOL_MAX_SIZE", "10"))

    try:
        _pool = await asyncpg.create_pool(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            min_size=min_size,
            max_size=max_size,
            command_timeout=30,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
        )
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS license_results (
                    id                     SERIAL PRIMARY KEY,
                    request_id             TEXT        NOT NULL,
                    driver_id              TEXT,
                    document_type          TEXT,
                    total                  INTEGER,
                    succeeded              INTEGER,
                    failed                 INTEGER,
                    index                  INTEGER,
                    filename               TEXT,
                    success                BOOLEAN,
                    detection_confidence   FLOAT,
                    license_number         TEXT,
                    full_name_english      TEXT,
                    full_name_arabic       TEXT,
                    date_of_birth          DATE,
                    date_of_issue          DATE,
                    date_of_expiry         DATE,
                    category               TEXT,
                    nationality            TEXT,
                    blood_group            TEXT,
                    restrictions           TEXT,
                    version                TEXT,
                    issuing_country        TEXT,
                    is_valid               BOOLEAN,
                    sequence               TEXT,
                    issue                  TEXT[],
                    warnings               TEXT[],
                    duration_ms            FLOAT,
                    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        log.info(f"PostgreSQL pool ready: {host}:{port}/{database} (min={min_size}, max={max_size})")
    except Exception as exc:
        log.error(f"Failed to connect to database {host}:{port}/{database} as '{user}': {exc}")
        _pool = None


def _parse_date(value: Any):
    """Try to parse a date string. Returns a date object or None."""
    if not value:
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except (ValueError, TypeError):
            continue
    return None


async def insert_license_result(
    *,
    request_id: str,
    document_type: str,
    total: int,
    succeeded: int,
    failed: int,
    index: int,
    filename: str,
    success: bool,
    detection_confidence: Optional[float],
    data: dict[str, Any],
    is_valid: bool,
    sequence: str,
    issue: list,
    driver_id: str,
    warnings: list,
    duration_ms: float,
) -> None:
    """
    Insert a single verification result row.
    Silently logs and returns on any failure — never raises.
    """
    if _pool is None:
        log.warning(f"[{request_id}] Database pool not initialised — skipping insert")
        return

    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO license_results (
                    request_id, driver_id, document_type, total, succeeded, failed,
                    index, filename, success, detection_confidence,
                    license_number, full_name_english, full_name_arabic,
                    date_of_birth, date_of_issue, date_of_expiry,
                    category, nationality, blood_group, restrictions, version,
                    issuing_country, is_valid, sequence, issue,
                    warnings, duration_ms
                ) VALUES (
                    $1,  $2,  $3,  $4,  $5,  $6,
                    $7,  $8,  $9,  $10,
                    $11, $12, $13,
                    $14, $15, $16,
                    $17, $18, $19, $20, $21,
                    $22, $23, $24, $25,
                    $26, $27
                )
                """,
                request_id,
                driver_id,
                document_type,
                total,
                succeeded,
                failed,
                index,
                filename,
                success,
                detection_confidence,
                data.get("license_number"),
                data.get("full_name_english"),
                data.get("full_name_arabic"),
                _parse_date(data.get("date_of_birth")),
                _parse_date(data.get("date_of_issue")),
                _parse_date(data.get("date_of_expiry")),
                data.get("category"),
                data.get("nationality"),
                data.get("blood_group"),
                data.get("restrictions"),
                data.get("version"),
                data.get("issuing_country"),
                is_valid,
                sequence,
                issue or [],
                warnings or [],
                duration_ms,
            )
        log.info(f"[{request_id}] Database insert successful")
    except Exception as exc:
        log.error(f"[{request_id}] DB insert failed: {exc}")


async def is_db_connected() -> bool:
    """Return True if the pool is initialised and a simple query succeeds."""
    if _pool is None:
        return False
    try:
        async with _pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as exc:
        log.warning(f"DB health check failed: {exc}")
        return False


async def close_db() -> None:
    """Close the connection pool. Call on app shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        log.info("PostgreSQL pool closed.")
