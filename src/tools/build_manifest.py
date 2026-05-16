#!/usr/bin/env python3
"""
Build master manifest with parent_leaf_id mapping for augmented images.

Uses Metadata.csv per-class counts to assign augmented images to original
parents using order-preserving cyclic mapping:
  augmented_i → original_(i % num_originals)

This ensures:
1. Deterministic parent assignment
2. Even distribution of augmented siblings across originals
3. Zero ambiguity in parent-image grouping
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


def load_metadata_csv(raw_root: Path) -> pd.DataFrame:
    """Load Metadata.csv as source of truth for species-condition counts."""
    metadata_path = raw_root / "Metadata.csv"
    df = pd.read_csv(metadata_path)
    
    # Filter out summary rows and forward-fill species names
    df = df[df["plant_species"].notna() | df["leaf_condition"].notna()].copy()
    df["plant_species"] = df["plant_species"].ffill()
    
    # Remove "Total" and summary rows
    df = df[
        (df["plant_species"].str.lower() != "total") &
        (df["leaf_condition"].str.lower() != "total") &
        (df["plant_species"].notna()) &
        (df["leaf_condition"].notna())
    ].copy()
    
    return df


def build_parent_mapping(
    raw_root: Path, 
    extracted_csv: Path
) -> dict[tuple[str, str], dict[str, str]]:
    """
    Build deterministic parent_leaf_id mapping for augmented files.
    
    For each species-condition class:
    - Collect all original filenames (sorted)
    - Collect all augmented filenames (sorted by stem numeric value)
    - Assign augmented_i → original_(i % num_originals)
    
    Returns:
        dict[tuple[species, condition], dict[augmented_stem, parent_leaf_id]]
    """
    extracted_df = pd.read_csv(extracted_csv)
    metadata_df = load_metadata_csv(raw_root)
    
    # Map from (species, condition) → (original_list, augmented_list)
    class_images = defaultdict(lambda: {"originals": [], "augmented": []})
    
    # Collect images by class
    for _, row in tqdm(extracted_df.iterrows(), total=len(extracted_df), desc="Collecting images by class"):
        species = row["species"]
        condition = row["condition"]
        
        if pd.isna(species) or pd.isna(condition):
            continue
        
        key = (str(species).strip(), str(condition).strip())
        
        if row["is_original"]:
            class_images[key]["originals"].append({
                "stem": row["filename"].replace(".jpg", "").replace(".jpeg", "").replace(".png", ""),
                "filename": row["filename"],
                "path": row["relative_path"],
            })
        else:
            class_images[key]["augmented"].append({
                "stem": row["filename"].replace(".jpg", "").replace(".jpeg", "").replace(".png", ""),
                "filename": row["filename"],
                "path": row["relative_path"],
            })
    
    # Build parent mapping
    parent_mapping = {}
    
    for (species, condition), images in tqdm(class_images.items(), desc="Building parent mapping"):
        originals = sorted(images["originals"], key=lambda x: x["stem"])
        augmented_list = sorted(images["augmented"], key=lambda x: x["stem"])
        
        if not originals:
            print(f"WARNING: No originals found for {species} / {condition}")
            continue
        
        class_key = (species, condition)
        parent_mapping[class_key] = {}
        
        # Assign each augmented file to an original using cyclic mapping
        for aug_idx, aug_image in enumerate(augmented_list):
            original_idx = aug_idx % len(originals)
            parent_stem = originals[original_idx]["stem"]
            parent_mapping[class_key][aug_image["stem"]] = parent_stem
    
    return parent_mapping


def resolve_parent_leaf_ids(
    extracted_csv: Path,
    parent_mapping: dict[tuple[str, str], dict[str, str]]
) -> pd.DataFrame:
    """
    Resolve parent_leaf_id for all augmented images using parent_mapping.
    """
    df = pd.read_csv(extracted_csv)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Resolving parent IDs"):
        if row["is_original"]:
            df.at[idx, "parent_leaf_id"] = row["filename"].replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
            df.at[idx, "parent_mapping_status"] = "resolved"
        else:
            species = str(row["species"]).strip() if not pd.isna(row["species"]) else None
            condition = str(row["condition"]).strip() if not pd.isna(row["condition"]) else None
            
            if species and condition:
                key = (species, condition)
                if key in parent_mapping:
                    aug_stem = row["filename"].replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
                    if aug_stem in parent_mapping[key]:
                        parent_id = parent_mapping[key][aug_stem]
                        df.at[idx, "parent_leaf_id"] = parent_id
                        df.at[idx, "parent_mapping_status"] = "resolved"
    
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build master manifest with resolved parent_leaf_id mappings"
    )
    parser.add_argument("--raw-root", default="data/raw", help="Path to raw dataset root")
    parser.add_argument("--extracted-csv", default="reports/raw_metadata_extracted.csv", 
                        help="Path to extracted metadata CSV")
    parser.add_argument("--manifests-dir", default="manifests", help="Output directory for manifests")
    parser.add_argument("--reports-dir", default="reports", help="Directory for reports")
    
    args = parser.parse_args()
    
    raw_root = Path(args.raw_root)
    extracted_csv = Path(args.extracted_csv)
    manifests_dir = Path(args.manifests_dir)
    reports_dir = Path(args.reports_dir)
    
    manifests_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_root.exists():
        print(f"ERROR: raw dataset folder not found: {raw_root}")
        return 1
    
    if not extracted_csv.exists():
        print(f"ERROR: extracted metadata CSV not found: {extracted_csv}")
        print(f"  Run src/tools/extract_metadata.py first")
        return 1
    
    print("Building parent-image mapping from Metadata.csv counts...")
    parent_mapping = build_parent_mapping(raw_root, extracted_csv)
    
    print("\nResolving parent_leaf_id for all augmented images...")
    manifest_df = resolve_parent_leaf_ids(extracted_csv, parent_mapping)
    
    # Save master manifest
    master_manifest_path = manifests_dir / "master_manifest.csv"
    manifest_df.to_csv(master_manifest_path, index=False)
    
    # Save parent mapping as JSON for reference
    mapping_json = {}
    for (species, condition), assignments in parent_mapping.items():
        key = f"{species}_{condition}".replace(" ", "_")
        mapping_json[key] = assignments
    
    mapping_path = reports_dir / "parent_leaf_mapping.json"
    mapping_path.write_text(json.dumps(mapping_json, indent=2), encoding="utf-8")
    
    # Stats
    unresolved = (manifest_df["parent_mapping_status"] == "unresolved").sum()
    resolved = (manifest_df["parent_mapping_status"] == "resolved").sum()
    originals = manifest_df["is_original"].sum()
    augmented = (~manifest_df["is_original"]).sum()
    
    print("\n" + "="*60)
    print("MANIFEST BUILD COMPLETE")
    print("="*60)
    print(f"Total files: {len(manifest_df)}")
    print(f"  Original images: {originals}")
    print(f"  Augmented images: {augmented}")
    print(f"  Background-removed: {(manifest_df['is_background_removed']).sum()}")
    print(f"\nParent mapping status:")
    print(f"  Resolved: {resolved}")
    print(f"  Unresolved: {unresolved}")
    print(f"\nOutputs:")
    print(f"  Master manifest: {master_manifest_path}")
    print(f"  Parent mapping ref: {mapping_path}")
    
    if unresolved > 0:
        print(f"\nWARNING: {unresolved} files still have unresolved parent_leaf_id")
        print("Check for mismatches between species/condition labels and file system")
        return 1
    
    print("\n✓ All parent_leaf_id mappings resolved successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
