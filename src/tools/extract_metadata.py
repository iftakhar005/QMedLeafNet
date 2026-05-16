#!/usr/bin/env python3
"""
Extract metadata from the dataset folder structure and Metadata.csv.

This dataset uses folder names for labels and a companion Metadata.csv for
species/condition counts. The CSV is the source of truth for the ontology,
while the directory structure is used to enumerate files.

The script builds:
- reports/raw_metadata_extracted.csv
- reports/label_ontology.json

It also flags parent_leaf_id as unresolved for augmented files when there
is no explicit parent-mapping source in the download.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm


KNOWN_CONDITIONS = [
    "Mature Healthy",
    "Young Healthy",
    "Mild Disease",
    "Chlorotic",
    "Dried",
    "Distorted",
    "Insects",
    "Healthy",
    "Disease",
]


@dataclass
class Ontology:
    species: list[str]
    conditions: list[str]


def parse_species_condition(class_folder: str) -> tuple[str | None, str | None]:
    folder = class_folder.strip()
    for condition in sorted(KNOWN_CONDITIONS, key=len, reverse=True):
        suffix = f" {condition}"
        if folder.endswith(suffix):
            species = folder[: -len(suffix)].strip()
            return species or None, condition
        if folder == condition:
            return None, condition
    return None, None


def infer_variant(top_folder: str) -> tuple[str, bool]:
    name = top_folder.lower()
    if "augmented" in name:
        return "augmented", False
    if "original" in name:
        return "original", False
    if "background" in name:
        return "background_removed", True
    return "unknown", False


def load_metadata_csv(raw_root: Path) -> pd.DataFrame | None:
    metadata_path = raw_root / "Metadata.csv"
    if not metadata_path.exists():
        return None
    return pd.read_csv(metadata_path)


def extract_records(raw_root: Path) -> pd.DataFrame:
    records = []
    metadata_df = load_metadata_csv(raw_root)
    metadata_species = []
    metadata_conditions = []
    if metadata_df is not None:
        metadata_species = [
            str(value).strip()
            for value in metadata_df["plant_species"].ffill().dropna().unique().tolist()
            if str(value).strip()
            and str(value).strip().lower() not in {"nan", "total"}
        ]
        metadata_conditions = [
            str(value).strip()
            for value in metadata_df["leaf_condition"].dropna().unique().tolist()
            if str(value).strip()
            and str(value).strip().lower() not in {"nan", "total"}
        ]

    image_paths = [p for p in raw_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"}]

    for image_path in tqdm(image_paths, desc="Extracting metadata"):
        relative_path = image_path.relative_to(raw_root)
        parts = relative_path.parts

        top_folder = parts[0] if parts else ""
        class_folder = parts[1] if len(parts) > 1 else ""

        variant, is_background_removed = infer_variant(top_folder)
        species, condition = parse_species_condition(class_folder)

        is_original = variant == "original"
        parent_leaf_id = image_path.stem if is_original else None

        records.append(
            {
                "file_path": str(image_path),
                "relative_path": str(relative_path).replace("\\", "/"),
                "top_folder": top_folder,
                "class_folder": class_folder,
                "variant": variant,
                "is_original": is_original,
                "is_background_removed": is_background_removed,
                "species": species,
                "condition": condition,
                "parent_leaf_id": parent_leaf_id,
                "parent_mapping_status": "resolved" if is_original else "unresolved",
                "filename": image_path.name,
            }
        )

    df = pd.DataFrame.from_records(records)

    # Improve metadata completeness by using the CSV source of truth when available.
    if metadata_df is not None and not metadata_df.empty:
        df.attrs["metadata_species"] = metadata_species
        df.attrs["metadata_conditions"] = metadata_conditions

    return df


def build_ontology(df: pd.DataFrame) -> Ontology:
    originals = df[df["is_original"]].copy()
    metadata_species = df.attrs.get("metadata_species", [])
    metadata_conditions = df.attrs.get("metadata_conditions", [])
    species = sorted(set([s for s in metadata_species if s] + [s for s in originals["species"].dropna().unique().tolist()]))
    conditions = sorted(set([c for c in metadata_conditions if c] + [c for c in originals["condition"].dropna().unique().tolist()]))
    return Ontology(species=species, conditions=conditions)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract folder-based metadata")
    parser.add_argument("--raw-root", default="data/raw", help="Path to raw dataset root")
    parser.add_argument("--reports-dir", default="reports", help="Directory for metadata outputs")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        print(f"ERROR: raw dataset folder not found: {raw_root}")
        return 1

    df = extract_records(raw_root)
    ontology = build_ontology(df)

    metadata_path = reports_dir / "raw_metadata_extracted.csv"
    ontology_path = reports_dir / "label_ontology.json"

    df.to_csv(metadata_path, index=False)
    ontology_path.write_text(json.dumps(asdict(ontology), indent=2), encoding="utf-8")

    unresolved_parent_count = int((df["parent_mapping_status"] == "unresolved").sum())
    metadata_csv_path = raw_root / "Metadata.csv"

    print("\nMETADATA EXTRACTION COMPLETE")
    print(f"Records written: {len(df)}")
    print(f"Species discovered: {len(ontology.species)}")
    print(f"Conditions discovered: {len(ontology.conditions)}")
    print(f"Metadata CSV used: {metadata_csv_path.exists()}")
    print(f"Unresolved parent mappings: {unresolved_parent_count}")
    print(f"Metadata CSV: {metadata_path}")
    print(f"Label ontology: {ontology_path}")

    if unresolved_parent_count > 0:
        print(
            "WARNING: augmented images do not expose a reliable parent_leaf_id in the download. "
            "A parent-mapping source or author rule is needed before manifest creation."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())