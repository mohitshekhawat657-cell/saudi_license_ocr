"""
Bulk License Extractor
======================
Process 1000–2000+ license images from a folder and save all results
as a single flat JSON array.

Usage:
    # Process all images in a folder
    python bulk_extract.py /path/to/images/

    # Custom output path
    python bulk_extract.py /path/to/images/ --output results.json

    # Tune concurrency & chunk size
    python bulk_extract.py /path/to/images/ --workers 4 --chunk-size 200

    # Save only successful extractions
    python bulk_extract.py /path/to/images/ --only-success

    # Use a specific YOLO model
    python bulk_extract.py /path/to/images/ --model runs/train/.../best.pt

Output JSON format (single flat array):
    [
      {
        "image_path": "...",
        "success": true,
        "data": { "license_number": "...", ... },
        "validation": { "valid": true, "issues": [], "warnings": [] },
        "detection_confidence": 0.97,
        "duration_ms": 3120.5,
        ...
      },
      ...
    ]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ── Import pipeline components ─────────────────────────────────────────────────
from pipeline import LicenseAuthPipeline, PipelineResult, _process_one, find_model
from ocr_extractor import Config as OCRConfig, check_together_health
from ocr_extractor import clear_cache
clear_cache()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bulk_extract")

SUPPORTED_EXTS = set(OCRConfig.ALLOWED_EXTS)


# =============================================================================
# IMAGE COLLECTION
# =============================================================================

def collect_images(folder: str, recursive: bool = True) -> list[str]:
    """Return sorted list of all supported image paths inside *folder*."""
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {folder}")

    glob_fn = root.rglob if recursive else root.glob
    paths = sorted(
        str(p)
        for p in glob_fn("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    return paths


# =============================================================================
# CHUNK PROCESSING
# =============================================================================

def _process_chunk(
    chunk: list[str],
    pipeline: LicenseAuthPipeline,
    workers: int,
    chunk_num: int,
    total_chunks: int,
    offset: int,          # how many images were processed before this chunk
    total_images: int,
) -> list[dict]:
    """
    Process a single chunk of image paths concurrently.
    Returns a list of result dicts in chunk order.
    """
    results: list[Optional[PipelineResult]] = [None] * len(chunk)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_one, path, pipeline): i
            for i, path in enumerate(chunk)
        }

        for future in as_completed(futures):
            idx        = futures[future]
            global_num = offset + idx + 1   # 1-based position across all images

            try:
                res = future.result()
            except Exception as exc:
                res = PipelineResult(
                    image_path=chunk[idx],
                    success=False,
                    error=f"Unhandled exception: {exc}",
                )

            results[idx] = res
            icon = "✓" if res.success else "✗"
            log.info(
                f"[{global_num:>5}/{total_images}] "
                f"chunk {chunk_num}/{total_chunks}  "
                f"{icon}  {Path(res.image_path).name}  ({res.duration_ms:.0f} ms)"
            )

    return [r.to_dict() for r in results if r is not None]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def bulk_extract(
    folder: str,
    output: str          = "bulk_results.json",
    workers: int         = 2,
    chunk_size: int      = 100,
    model_path: Optional[str] = None,
    only_success: bool   = False,
    recursive: bool      = True,
) -> None:
    """
    Scan *folder* for images, process every image through the YOLO + VLM
    pipeline, and write all results to *output* as a single flat JSON array.

    Writes the array to disk after every chunk so progress is never lost
    if the run is interrupted.
    """

    # ── Collect images ────────────────────────────────────────────────────────
    log.info(f"Scanning {'recursively ' if recursive else ''}in: {folder}")
    try:
        image_paths = collect_images(folder, recursive=recursive)
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    if not image_paths:
        log.error(
            f"No supported images found in: {folder}\n"
            f"Supported formats: {sorted(SUPPORTED_EXTS)}"
        )
        sys.exit(1)

    log.info(f"Found {len(image_paths):,} image(s)")

    # ── VLM / Ollama health ───────────────────────────────────────────────────
    if OCRConfig.VLM_BACKEND == "together":
        health = check_together_health()
        if not health["healthy"]:
            log.error(f"Together API is not reachable: {health['error']}")
            log.error("Check your TOGETHER_API_KEY environment variable.")
            sys.exit(1)
        log.info(f"Together API OK  model_loaded={health['model_loaded']}")

    # ── Load YOLO pipeline (once, shared across all chunks) ───────────────────
    resolved_model = model_path or find_model()
    if not resolved_model:
        log.error(
            "YOLO model not found. "
            "Run 'python train.py' or pass --model /path/to/best.pt"
        )
        sys.exit(1)

    log.info(f"Loading YOLO model: {resolved_model}")
    pipeline = LicenseAuthPipeline(model_path=resolved_model)

    # ── Divide into chunks ────────────────────────────────────────────────────
    chunks = [
        image_paths[i: i + chunk_size]
        for i in range(0, len(image_paths), chunk_size)
    ]
    total_chunks = len(chunks)
    log.info(
        f"Processing {len(image_paths):,} images in {total_chunks} chunk(s) "
        f"of ≤{chunk_size}  |  workers={workers}"
    )

    # ── Prepare output file (start with empty array) ──────────────────────────
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("[]")

    # ── Process chunk by chunk ────────────────────────────────────────────────
    all_results: list[dict] = []
    succeeded  = 0
    failed     = 0
    t_start    = time.monotonic()

    for chunk_num, chunk in enumerate(chunks, start=1):
        offset = (chunk_num - 1) * chunk_size

        chunk_results = _process_chunk(
            chunk        = chunk,
            pipeline     = pipeline,
            workers      = workers,
            chunk_num    = chunk_num,
            total_chunks = total_chunks,
            offset       = offset,
            total_images = len(image_paths),
        )

        if only_success:
            chunk_results = [r for r in chunk_results if r.get("success")]

        all_results.extend(chunk_results)

        # ── Per-chunk stats ───────────────────────────────────────────────────
        chunk_ok   = sum(1 for r in chunk_results if r.get("success"))
        chunk_fail = len(chunk_results) - chunk_ok
        succeeded += chunk_ok
        failed    += chunk_fail

        done    = offset + len(chunk)
        elapsed = time.monotonic() - t_start
        avg_s   = elapsed / done if done else 0
        eta_s   = (len(image_paths) - done) * avg_s

        log.info(
            f"── Chunk {chunk_num}/{total_chunks} complete — "
            f"{chunk_ok} ok / {chunk_fail} failed — "
            f"running total: {succeeded}/{done} succeeded — "
            f"ETA ~{eta_s:.0f}s"
        )

        # ── Write full array to disk after every chunk (safe against crash) ───
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - t_start
    rate = succeeded / len(image_paths) * 100 if image_paths else 0
    avg_ms = total_elapsed * 1000 / len(image_paths) if image_paths else 0

    print()
    print("=" * 64)
    print("BULK EXTRACTION COMPLETE")
    print("=" * 64)
    print(f"  Total images  : {len(image_paths):,}")
    print(f"  Succeeded     : {succeeded:,}  ({rate:.1f}%)")
    print(f"  Failed        : {failed:,}")
    print(f"  Total time    : {total_elapsed:.1f}s")
    print(f"  Avg per image : {avg_ms:.0f}ms")
    print(f"  Output file   : {out_path.resolve()}")
    print(f"  Records saved : {len(all_results):,}")
    print("=" * 64)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Bulk license extractor — processes 1000–2000+ images and\n"
            "writes all results to a single flat JSON array."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "folder",
        help="Folder containing license images.\nSearched recursively by default.",
    )
    parser.add_argument(
        "--output", "-o",
        default="bulk_results.json",
        metavar="FILE",
        help="Output JSON file path.\n(default: bulk_results.json)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Concurrent worker threads per chunk.\n"
            "Keep ≤ 2 for Ollama (single-GPU queue).\n"
            "(default: 2)"
        ),
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=100,
        metavar="N",
        help=(
            "Images processed per chunk before writing to disk.\n"
            "Larger = fewer writes, more RAM usage.\n"
            "(default: 100)"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="PATH",
        help="Path to YOLO best.pt. Auto-searched if omitted.",
    )
    parser.add_argument(
        "--only-success",
        action="store_true",
        help="Exclude failed extractions from the output JSON.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search sub-folders (top-level files only).",
    )

    args = parser.parse_args()

    bulk_extract(
        folder       = args.folder,
        output       = args.output,
        workers      = args.workers,
        chunk_size   = args.chunk_size,
        model_path   = args.model,
        only_success = args.only_success,
        recursive    = not args.no_recursive,
    )
