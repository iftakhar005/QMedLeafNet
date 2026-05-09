#!/usr/bin/env python3
"""
Create leakage-safe grouped data splits.

Principle: Group by parent_leaf_id, not by individual files.
- Test: 15% of unique parents (~297 parents) 
- Dev: 85% of unique parents (~1,684 parents) with 5-fold CV

This ensures:
1. All augmented siblings of a parent go to same split
2. Zero parent-image leakage across train/val/test
3. Stratified by species-condition for balanced splits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load master manifest with resolved parent_leaf_id."""
    return pd.read_csv(manifest_path)


def build_parent_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Build parent-level grouping with stratification labels.
    
    Returns DataFrame with one row per unique parent:
    - parent_leaf_id
    - num_files (all augmented + background variants)
    - num_originals (always 1)
    - num_augmented
    - num_background_removed
    - species, condition (stratification key)
    """
    parents = []
    
    for parent_id in manifest["parent_leaf_id"].dropna().unique():
        parent_files = manifest[manifest["parent_leaf_id"] == parent_id]
        
        # Get stratification info from any file in the group
        first_file = parent_files.iloc[0]
        species = first_file["species"]
        condition = first_file["condition"]
        
        # Count file types
        num_originals = int(parent_files[parent_files["is_original"]].shape[0])
        num_augmented = int(parent_files[(~parent_files["is_original"]) & (~parent_files["is_background_removed"])].shape[0])
        num_background_removed = int(parent_files[parent_files["is_background_removed"]].shape[0])
        num_total = len(parent_files)
        
        parents.append({
            "parent_leaf_id": parent_id,
            "num_files": num_total,
            "num_originals": num_originals,
            "num_augmented": num_augmented,
            "num_background_removed": num_background_removed,
            "species": species,
            "condition": condition,
            "stratify_key": f"{species}_{condition}",
        })
    
    return pd.DataFrame(parents)


def stratified_split(
    parent_groups: pd.DataFrame,
    test_ratio: float = 0.15,
    n_folds: int = 5,
    random_seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    """
    Create stratified train/test split with k-fold CV on train set.
    
    Returns:
        (parent_splits_df, fold_assignments)
    """
    np.random.seed(random_seed)
    
    # Create stratification labels (numeric encoding of species-condition)
    unique_strata = parent_groups["stratify_key"].unique()
    strata_map = {s: i for i, s in enumerate(sorted(unique_strata))}
    parent_groups["strata_id"] = parent_groups["stratify_key"].map(strata_map)
    
    n_parents = len(parent_groups)
    n_test = int(np.ceil(n_parents * test_ratio))
    n_dev = n_parents - n_test
    
    # Stratified split: test vs dev
    strata_indices = defaultdict(list)
    for idx, strata_id in enumerate(parent_groups["strata_id"]):
        strata_indices[strata_id].append(idx)
    
    test_indices = []
    dev_indices = []
    
    for strata_id, indices in strata_indices.items():
        indices_arr = np.array(indices)
        n_strata = len(indices_arr)
        n_strata_test = max(1, int(np.ceil(n_strata * test_ratio)))
        
        shuffled = np.random.permutation(indices_arr)
        test_indices.extend(shuffled[:n_strata_test])
        dev_indices.extend(shuffled[n_strata_test:])
    
    # Assign splits
    parent_groups["split"] = "unknown"
    parent_groups.loc[test_indices, "split"] = "test"
    parent_groups.loc[dev_indices, "split"] = "dev"
    
    # K-fold on dev set
    dev_group = parent_groups[parent_groups["split"] == "dev"].copy().reset_index(drop=True)
    fold_assignment = np.zeros(len(dev_group), dtype=int) - 1
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    fold_dict = defaultdict(list)
    for fold_id, (train_idx, val_idx) in enumerate(
        skf.split(dev_group, dev_group["strata_id"])
    ):
        fold_dict[fold_id] = {
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
        }
        fold_assignment[val_idx] = fold_id
    
    # Assign fold IDs back to parent_groups
    parent_groups.loc[dev_indices, "fold"] = fold_assignment
    parent_groups.loc[test_indices, "fold"] = -1  # Test set has no fold
    
    return parent_groups, fold_dict


def assign_files_to_splits(
    manifest: pd.DataFrame,
    parent_splits: pd.DataFrame
) -> pd.DataFrame:
    """
    Assign all files (including augmented and background-removed) to splits
    based on their parent_leaf_id.
    """
    manifest = manifest.copy()
    manifest["split"] = "unknown"
    manifest["fold"] = -1
    
    for _, parent_row in parent_splits.iterrows():
        parent_id = parent_row["parent_leaf_id"]
        split = parent_row["split"]
        fold = parent_row["fold"]
        
        mask = manifest["parent_leaf_id"] == parent_id
        manifest.loc[mask, "split"] = split
        manifest.loc[mask, "fold"] = fold
    
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe grouped data splits"
    )
    parser.add_argument("--manifest", default="manifests/master_manifest.csv",
                        help="Path to master manifest")
    parser.add_argument("--output-dir", default="manifests/splits_v1",
                        help="Output directory for splits")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="Fraction of parents for test set")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        return 1
    
    print("Loading manifest...")
    manifest = load_manifest(manifest_path)
    
    print("Building parent groups...")
    parent_groups = build_parent_groups(manifest)
    
    print(f"Total unique parents: {len(parent_groups)}")
    print(f"Total files: {manifest.shape[0]}")
    
    print("\nCreating stratified splits...")
    parent_splits, fold_dict = stratified_split(
        parent_groups,
        test_ratio=args.test_ratio,
        n_folds=args.n_folds,
        random_seed=args.seed
    )
    
    # Assign all files to splits
    print("Assigning files to splits...")
    manifest_splits = assign_files_to_splits(manifest, parent_splits)
    
    # Statistics
    print("\n" + "="*60)
    print("SPLIT SUMMARY")
    print("="*60)
    
    test_parents = parent_splits[parent_splits["split"] == "test"]
    dev_parents = parent_splits[parent_splits["split"] == "dev"]
    
    print(f"\nParent-level splits:")
    print(f"  Test parents: {len(test_parents)} ({100*len(test_parents)/len(parent_splits):.1f}%)")
    print(f"  Dev parents: {len(dev_parents)} ({100*len(dev_parents)/len(parent_splits):.1f}%)")
    
    print(f"\nFile-level splits:")
    test_files = manifest_splits[manifest_splits["split"] == "test"]
    dev_files = manifest_splits[manifest_splits["split"] == "dev"]
    
    print(f"  Test files: {len(test_files)} ({100*len(test_files)/len(manifest_splits):.1f}%)")
    print(f"  Dev files: {len(dev_files)} ({100*len(dev_files)/len(manifest_splits):.1f}%)")
    print(f"    - Original: {test_files['is_original'].sum()}")
    print(f"    - Augmented: {(~test_files['is_original'] & ~test_files['is_background_removed']).sum()}")
    print(f"    - Background-removed: {test_files['is_background_removed'].sum()}")
    
    print(f"\nK-fold CV ({args.n_folds} folds) on dev set:")
    for fold_id in range(args.n_folds):
        fold_files = manifest_splits[
            (manifest_splits["split"] == "dev") & (manifest_splits["fold"] == fold_id)
        ]
        print(f"  Fold {fold_id}: {len(fold_files)} files")
    
    # Save splits
    print(f"\nSaving splits...")
    
    # Save master splits CSV
    splits_csv = output_dir / "all_splits.csv"
    manifest_splits.to_csv(splits_csv, index=False)
    print(f"  {splits_csv}")
    
    # Save test set
    test_csv = output_dir / "test_set.csv"
    test_files.to_csv(test_csv, index=False)
    print(f"  {test_csv}")
    
    # Save fold assignments
    for fold_id in range(args.n_folds):
        fold_data = manifest_splits[
            (manifest_splits["split"] == "dev") & (manifest_splits["fold"] == fold_id)
        ]
        
        train_data = fold_data[fold_data["fold"] != fold_id]
        val_data = fold_data[fold_data["fold"] == fold_id]
        
        # Actually, need to recalculate properly - all dev files with this fold_id are val
        dev_mask = manifest_splits["split"] == "dev"
        dev_data = manifest_splits[dev_mask].copy()
        
        val_mask = dev_data["fold"] == fold_id
        train_mask = dev_data["fold"] != fold_id
        
        # Re-index within dev
        fold_train = manifest_splits[
            (manifest_splits["split"] == "dev") & (manifest_splits["fold"] != fold_id)
        ]
        fold_val = manifest_splits[
            (manifest_splits["split"] == "dev") & (manifest_splits["fold"] == fold_id)
        ]
        
        train_csv = output_dir / f"fold_{fold_id}_train.csv"
        val_csv = output_dir / f"fold_{fold_id}_val.csv"
        
        fold_train.to_csv(train_csv, index=False)
        fold_val.to_csv(val_csv, index=False)
    
    print(f"\n✓ Splits created in {output_dir}")
    print(f"\nKey files:")
    print(f"  - all_splits.csv: Master split assignments")
    print(f"  - test_set.csv: Test set ({len(test_files)} files)")
    print(f"  - fold_N_train.csv, fold_N_val.csv: K-fold splits")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
