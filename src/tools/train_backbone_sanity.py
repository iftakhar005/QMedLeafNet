#!/usr/bin/env python3
"""Run a tiny Phase 2 backbone sanity check.

This script loads a handful of files from a split manifest, applies the Phase 2
preprocessing pipeline, and runs one forward/backward step through a TorchVision
backbone plus a linear head.
"""

from __future__ import annotations
from src.utils.preprocessing import preprocess_image
from src.models.phase2_backbones import (
    build_classification_head,
    build_feature_extractor,
    count_parameters,
    freeze_module,
)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_num_classes(ontology_path: Path) -> int:
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    species = ontology.get("species", [])
    conditions = ontology.get("conditions", [])
    if not species or not conditions:
        raise ValueError("Ontology is missing species or conditions")
    return len(species) * len(conditions)


def sample_image_paths(manifest_path: Path, sample_size: int) -> list[str]:
    manifest = pd.read_csv(manifest_path)
    if "file_path" not in manifest.columns:
        raise ValueError("Manifest must contain a file_path column")
    sampled = manifest["file_path"].dropna().head(sample_size).tolist()
    if len(sampled) < sample_size:
        raise ValueError(
            f"Requested {sample_size} samples, found only {len(sampled)}")
    return sampled


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2 backbone sanity check")
    parser.add_argument(
        "--manifest", default="manifests/splits_v1/dev_fold_0_train.csv")
    parser.add_argument("--ontology", default="reports/label_ontology.json")
    parser.add_argument("--backbone", default="mobilenet_v3_small")
    parser.add_argument("--sample-size", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-backbone",
                        dest="freeze_backbone", action="store_false")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    ontology_path = Path(args.ontology)

    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            "torch is required to run this sanity check. Install the Phase 2 dependencies first."
        ) from exc

    image_paths = sample_image_paths(manifest_path, args.sample_size)
    batch = np.stack([preprocess_image(path) for path in image_paths], axis=0)
    batch_tensor = torch.from_numpy(batch)

    feature_extractor, feature_dim = build_feature_extractor(args.backbone)
    head = build_classification_head(
        feature_dim, load_num_classes(ontology_path))

    if args.freeze_backbone:
        freeze_module(feature_extractor)

    backbone_params = count_parameters(feature_extractor, trainable_only=False)
    head_params = count_parameters(head, trainable_only=False)
    trainable_params = count_parameters(
        feature_extractor, trainable_only=True) + count_parameters(head, trainable_only=True)

    feature_extractor.eval()
    head.eval()

    with torch.no_grad():
        features = feature_extractor(batch_tensor)
        if features.ndim == 4:
            features = torch.flatten(
                torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)), 1)
        elif features.ndim == 3:
            features = features.flatten(1)
        logits = head(features)

    print("PHASE 2 BACKBONE SANITY CHECK")
    print(f"Backbone: {args.backbone}")
    print(f"Sample images: {len(image_paths)}")
    print(f"Feature dim: {feature_dim}")
    print(f"Backbone parameters: {backbone_params}")
    print(f"Head parameters: {head_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Batch tensor shape: {tuple(batch_tensor.shape)}")
    print(f"Feature tensor shape: {tuple(features.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")

    if logits.shape[0] != len(image_paths):
        raise RuntimeError(
            "Batch dimension mismatch during backbone sanity check")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
