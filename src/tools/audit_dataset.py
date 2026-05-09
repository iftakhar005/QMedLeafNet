#!/usr/bin/env python3
"""
Audit the raw medicinal plant dataset.

Outputs:
- reports/data_audit.json
- reports/raw_checksums.txt
- reports/checksum_verification.txt

The script verifies:
- folder/file counts
- image readability
- corruption rate
- basic size and format distribution
"""

from __future__ import annotations

import argparse
import json
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass
class AuditSummary:
    raw_root: str
    total_files: int
    total_images: int
    valid_images: int
    corrupted_images: int
    corruption_rate: float
    folder_counts: dict
    format_counts: dict
    size_counts: dict
    corrupted_files: list


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(raw_root: Path, reports_dir: Path) -> AuditSummary:
    reports_dir.mkdir(parents=True, exist_ok=True)

    total_files = sum(1 for _ in raw_root.rglob("*"))
    image_files = list(iter_image_files(raw_root))

    folder_counts = Counter()
    format_counts = Counter()
    size_counts = Counter()
    corrupted_files = []
    valid_images = 0

    checksum_lines = []

    for image_path in tqdm(image_files, desc="Auditing images"):
        folder_counts[str(image_path.parent.relative_to(raw_root))] += 1
        checksum_lines.append(f"{sha256_file(image_path)}  {image_path.as_posix()}")

        try:
            with Image.open(image_path) as img:
                img.verify()
            with Image.open(image_path) as img:
                format_counts[str(img.format)] += 1
                size_counts[f"{img.width}x{img.height}"] += 1
            valid_images += 1
        except Exception as exc:  # pragma: no cover - runtime safety
            corrupted_files.append({"file": image_path.as_posix(), "error": str(exc)})

    corrupted_images = len(corrupted_files)
    corruption_rate = (corrupted_images / len(image_files)) if image_files else 0.0

    raw_checksums_path = reports_dir / "raw_checksums.txt"
    raw_checksums_path.write_text("\n".join(checksum_lines) + ("\n" if checksum_lines else ""), encoding="utf-8")

    checksum_verification_path = reports_dir / "checksum_verification.txt"
    checksum_verification_path.write_text(
        "Checksums created by audit_dataset.py.\n"
        "To verify on Windows PowerShell, rerun a SHA256 comparison if needed.\n",
        encoding="utf-8",
    )

    summary = AuditSummary(
        raw_root=str(raw_root),
        total_files=total_files,
        total_images=len(image_files),
        valid_images=valid_images,
        corrupted_images=corrupted_images,
        corruption_rate=corruption_rate,
        folder_counts=dict(sorted(folder_counts.items())),
        format_counts=dict(sorted(format_counts.items())),
        size_counts=dict(sorted(size_counts.items())),
        corrupted_files=corrupted_files[:50],
    )

    audit_path = reports_dir / "data_audit.json"
    audit_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the Phase 1 raw dataset")
    parser.add_argument("--raw-root", default="data/raw", help="Path to raw dataset root")
    parser.add_argument("--reports-dir", default="reports", help="Path to reports directory")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    reports_dir = Path(args.reports_dir)

    if not raw_root.exists():
        print(f"ERROR: raw dataset folder not found: {raw_root}")
        return 1

    summary = audit_dataset(raw_root, reports_dir)

    print("\nDATASET AUDIT COMPLETE")
    print(f"Raw root: {summary.raw_root}")
    print(f"Total files: {summary.total_files}")
    print(f"Image files: {summary.total_images}")
    print(f"Valid images: {summary.valid_images}")
    print(f"Corrupted images: {summary.corrupted_images}")
    print(f"Corruption rate: {summary.corruption_rate:.4%}")
    print(f"Report: {reports_dir / 'data_audit.json'}")
    print(f"Checksums: {reports_dir / 'raw_checksums.txt'}")
    return 0 if summary.corruption_rate < 0.01 else 2


if __name__ == "__main__":
    raise SystemExit(main())