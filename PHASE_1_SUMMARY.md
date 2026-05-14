# Phase 1 Summary

## What Phase 1 Established

Phase 1 created the data foundation for QMedLeafNet and confirmed the dataset is ready for leakage-safe model development.

## Final Results

- Total indexed files: 25,962
- Original images: 1,981
- Augmented images: 23,981
- Background-removed images: 1,981
- Corruption rate: 0.0%
- Unique parents: 1,981
- Parent mapping resolution: 100%
- Zero parent leakage: verified 3 times

## Frozen Label Ontology

The approved ontology contains:

- 7 species
- 9 conditions

Species:

- Aloe Vera
- Azadirachta Indica
- Centella Asiatica
- Hibiscus Rosa Sinensis
- Kalanchoe Pinnata
- Mikania Micrantha
- Piper Betle

Conditions:

- Chlorotic
- Disease
- Distorted
- Dried
- Healthy
- Insects
- Mature Healthy
- Mild Disease
- Young Healthy

## Key Artifacts

- `reports/data_audit.json`
- `reports/raw_checksums.txt`
- `reports/label_ontology.json`
- `reports/raw_metadata_extracted.csv`
- `reports/parent_leaf_mapping.json`
- `manifests/master_manifest.csv`
- `manifests/splits_v1/all_splits.csv`
- `manifests/splits_v1/test_set.csv`
- `PHASE_1_FINAL_SIGN_OFF.md`

## Split Summary

- Test split: 307 parents, 4,032 files
- Dev split: 1,674 parents, 21,930 files
- CV setup: 5 folds
- Train/validation overlap: 0 parents in every fold
- Test/dev overlap: 0 parents

## Notes

- The repo uses `manifests/master_manifest.csv` as the master manifest.
- `data/raw/Metadata.csv` is the source of truth for label structure and counts.
- Large image folders are excluded from git via `.gitignore`.

## Status

Phase 1 is complete and approved. The project is ready to move to Phase 2.
