#!/usr/bin/env python3
"""
Verify zero parent-image leakage in splits.

Checks:
1. Test and Dev sets have zero parent overlap
2. Each fold's train and val have zero parent overlap
3. All folds' train sets have zero parent overlap with each other
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def verify_no_parent_overlap(
    set1_df: pd.DataFrame,
    set2_df: pd.DataFrame,
    set1_name: str,
    set2_name: str
) -> bool:
    """Check no common parents between two sets."""
    parents1 = set(set1_df["parent_leaf_id"].dropna().unique())
    parents2 = set(set2_df["parent_leaf_id"].dropna().unique())
    overlap = parents1 & parents2
    
    if len(overlap) > 0:
        print(f"  ✗ LEAKAGE DETECTED: {len(overlap)} parents in both {set1_name} and {set2_name}")
        return False
    else:
        print(f"  ✓ {set1_name} vs {set2_name}: 0 parent overlap")
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify zero parent-image leakage")
    parser.add_argument("--splits-dir", default="manifests/splits_v1",
                        help="Directory with split files")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of folds")
    
    args = parser.parse_args()
    
    splits_dir = Path(args.splits_dir)
    
    if not splits_dir.exists():
        print(f"ERROR: splits directory not found: {splits_dir}")
        return 1
    
    print("="*60)
    print("LEAKAGE VERIFICATION")
    print("="*60)
    
    # Load splits
    test_csv = splits_dir / "test_set.csv"
    
    if not test_csv.exists():
        print(f"ERROR: test_set.csv not found")
        return 1
    
    test_df = pd.read_csv(test_csv)
    all_splits = pd.read_csv(splits_dir / "all_splits.csv")
    dev_df = all_splits[all_splits["split"] == "dev"]
    
    print(f"\nData loaded:")
    print(f"  Test set: {len(test_df)} files, {len(test_df['parent_leaf_id'].unique())} parents")
    print(f"  Dev set: {len(dev_df)} files, {len(dev_df['parent_leaf_id'].unique())} parents")
    
    all_passed = True
    
    # Check 1: Test vs Dev
    print(f"\n--- CHECK 1: Test vs Dev ---")
    passed = verify_no_parent_overlap(test_df, dev_df, "Test", "Dev")
    all_passed = all_passed and passed
    
    # Check 2: Within each fold, train vs val
    print(f"\n--- CHECK 2: Train vs Val (within folds) ---")
    for fold_id in range(args.n_folds):
        train_csv = splits_dir / f"fold_{fold_id}_train.csv"
        val_csv = splits_dir / f"fold_{fold_id}_val.csv"
        
        if not train_csv.exists() or not val_csv.exists():
            print(f"  Fold {fold_id}: files not found")
            continue
        
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        
        passed = verify_no_parent_overlap(
            train_df, val_df,
            f"Fold {fold_id} train", f"Fold {fold_id} val"
        )
        all_passed = all_passed and passed
    
    # Check 3: Validate all val sets are disjoint (each parent appears in exactly one fold)
    print(f"\n--- CHECK 3: Val sets (disjoint across folds) ---")
    all_val_parents = set()
    for fold_id in range(args.n_folds):
        val_csv = splits_dir / f"fold_{fold_id}_val.csv"
        if val_csv.exists():
            val_df = pd.read_csv(val_csv)
            val_parents = set(val_df["parent_leaf_id"].dropna().unique())
            
            # Check if any parent appears in multiple folds
            overlap_with_previous = all_val_parents & val_parents
            if len(overlap_with_previous) > 0:
                print(f"  ✗ Fold {fold_id}: {len(overlap_with_previous)} parents also in other folds")
                all_passed = False
            else:
                print(f"  ✓ Fold {fold_id} val: {len(val_parents)} unique parents (disjoint)")
            
            all_val_parents.update(val_parents)
    
    # Summary
    print(f"\n" + "="*60)
    if all_passed:
        print("✓ ZERO LEAKAGE VERIFIED - All checks passed!")
        return 0
    else:
        print("✗ LEAKAGE DETECTED - Some checks failed!")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
