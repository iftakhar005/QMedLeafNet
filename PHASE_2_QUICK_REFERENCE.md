# Phase 2 Quick Reference - Classical Baseline Training

## Overview

Phase 2 trains 4 parameter-matched classical heads on frozen MobileNetV3-Small features from Phase 1 splits.

---

## 1. Prerequisites Check

✅ Phase 1 data ready:

```bash
ls manifests/splits_v1/fold_*.csv  # Should show 10 files (5 folds × train+val)
ls reports/label_ontology.json      # Label mapping for 63 classes
```

✅ Dependencies installed:

```bash
pip list | grep torch
pip list | grep scikit-learn
```

---

## 2. Validation Runs (Quick)

### 2.1 Backbone Sanity Check (2 min)

```bash
python src/tools/train_backbone_sanity.py \
  --backbone mobilenet_v3_small \
  --sample-size 10
```

**Expected output:**

```
PHASE 2 BACKBONE SANITY CHECK
Backbone: mobilenet_v3_small
Sample images: 10
Feature dim: 576
Batch tensor shape: (10, 3, 224, 224)
Feature tensor shape: (10, 576)
Logits shape: (10, 63)
```

### 2.2 Preprocessing Validation (1 min)

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.utils.preprocessing import preprocess_image
import numpy as np

img_path = 'data/raw/Original Dataset/Aloe Vera Disease/IMG20260204141253.jpg'
arr = preprocess_image(img_path)
print(f'Shape: {arr.shape}, dtype: {arr.dtype}, range: [{arr.min():.2f}, {arr.max():.2f}]')
"
```

**Expected output:**

```
Shape: (3, 224, 224), dtype: float32, range: [-2.13, 2.45]
```

---

## 3. Full Baseline Training

### 3.1 Train on Fold 0 (45-60 min, GPU recommended)

```bash
python src/tools/train_classical_baseline.py \
  --data-dir . \
  --backbone mobilenet_v3_small \
  --fold 0
```

**Expected console output:**

```
Device: cuda (or cpu)
Fold 0: Train=1674, Val=417

============================================================
Training logistic_regression
Train samples: 1674, Val samples: 417
============================================================
Extracting training features...
Extracting features: 100%|████| 1674/1674
...
Val Accuracy: 0.XXXX
Val F1 (weighted): 0.XXXX

============================================================
CLASSICAL BASELINE RESULTS SUMMARY
============================================================
Head Type                 Accuracy          F1 (weighted)
------------------------------------------------------------
logistic_regression            0.XXXX              0.XXXX
svm_rbf                        0.XXXX              0.XXXX
svm_linear                     0.XXXX              0.XXXX
dense_mlp                      0.XXXX              0.XXXX

Results saved to reports/classical_baseline_fold0.json
```

### 3.2 Train on All Folds (Sequential)

```bash
for fold in {0..4}; do
  echo "Training fold $fold..."
  python src/tools/train_classical_baseline.py --fold $fold
done
```

**Time estimate:** 4-5 hours total (45 min × 5 folds)

---

## 4. Results Analysis

### 4.1 View Results for One Fold

```bash
python -c "
import json
with open('reports/classical_baseline_fold0.json') as f:
    results = json.load(f)
    for head, metrics in results.items():
        print(f'{head:20s} Acc: {metrics[\"accuracy\"]:.4f} F1: {metrics[\"f1\"]:.4f}')
"
```

### 4.2 Aggregate Results Across All Folds

```bash
python << 'EOF'
import json
import numpy as np
from pathlib import Path

results_by_head = {}
for fold in range(5):
    path = f'reports/classical_baseline_fold{fold}.json'
    if Path(path).exists():
        with open(path) as f:
            fold_results = json.load(f)
            for head, metrics in fold_results.items():
                if head not in results_by_head:
                    results_by_head[head] = {'acc': [], 'f1': []}
                results_by_head[head]['acc'].append(metrics['accuracy'])
                results_by_head[head]['f1'].append(metrics['f1'])

print("Cross-fold results:")
print(f"{'Head':<20} {'Accuracy':<20} {'F1 (weighted)':<20}")
print("-" * 60)
for head in sorted(results_by_head.keys()):
    accs = results_by_head[head]['acc']
    f1s = results_by_head[head]['f1']
    print(f"{head:<20} {np.mean(accs):.4f} ± {np.std(accs):.4f}   {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
EOF
```

---

## 5. Command-Line Reference

### train_backbone_sanity.py

```
--manifest PATH          Master manifest (default: manifests/master_manifest.csv)
--ontology PATH          Label ontology (default: reports/label_ontology.json)
--backbone {mobilenet_v3_small, mobilenet_v2, efficientnet_b0, resnet18}
                         Backbone model (default: mobilenet_v3_small)
--sample-size N          Number of images to test (default: 10)
--freeze-backbone        Freeze backbone parameters (flag)
--no-freeze-backbone     Train backbone (flag)
```

### train_classical_baseline.py

```
--data-dir PATH          Root data directory (default: .)
--manifest PATH          Master manifest (default: manifests/master_manifest.csv)
--ontology PATH          Label ontology (default: reports/label_ontology.json)
--backbone {mobilenet_v3_small, mobilenet_v2, efficientnet_b0, resnet18}
                         Backbone model (default: mobilenet_v3_small)
--fold N                 Fold index 0-4 (default: 0)
```

---

## 6. Troubleshooting

### "cannot import name 'load_and_preprocess_image'"

→ Use `preprocess_image()` instead (returns numpy CHW array)

### "No such file: manifests/splits_v1/fold_0_train.csv"

→ Run Phase 1 manifest generation first:

```bash
python src/tools/build_manifest.py
python src/tools/make_grouped_splits.py
```

### Out of memory (OOM) during feature extraction

→ Process fewer images at once or use CPU:

- Modify extract_features() to use batch processing
- Or run on CPU: `device = 'cpu'`

### Very slow feature extraction on CPU

→ Expected. GPU recommended:

- NVIDIA GPU: ~2-5 min per 1000 images
- CPU: ~10-15 min per 1000 images

### "weights=None" error in backbone loading

→ Update torchvision:

```bash
pip install --upgrade torchvision
```

---

## 7. Expected Metrics

### Baseline Performance (9 conditions × 7 species = 63 classes)

- **Random guessing:** ~1.6% accuracy
- **Majority class baseline:** ~5-10% accuracy (depends on class imbalance)
- **Classical heads:** Expected 40-70% accuracy (depends on species/condition complexity)

### Head Comparison

All heads have ~36-41K trainable parameters (parameter-matched).

- **Logistic Regression:** Fast, interpretable, baseline
- **SVM-RBF:** Nonlinear, typically higher accuracy
- **SVM-Linear:** Fast, lower capacity
- **Dense MLP:** Maximum flexibility, risk of overfitting

---

## 8. Next Steps After Classical Baseline

1. **Quantum Head (Phase 2.2)**
   - Replace classical head with quantum circuit
   - Compare performance vs classical baselines

2. **Ensemble (Phase 2.3)**
   - Combine predictions from multiple heads
   - Voting or learned weighting

3. **Fine-tuning (Phase 3)**
   - Unlock backbone layers gradually
   - Joint optimization of backbone + head

---

## 9. File Locations

| File            | Path                                           |
| --------------- | ---------------------------------------------- |
| Splits          | `manifests/splits_v1/fold_{N}_{train,val}.csv` |
| Ontology        | `reports/label_ontology.json`                  |
| Results         | `reports/classical_baseline_fold{N}.json`      |
| Preprocessing   | `src/utils/preprocessing.py`                   |
| Classical Heads | `src/models/classical_heads.py`                |
| Trainers        | `src/tools/train_classical_baseline.py`        |

---

## 10. Key Commands

```bash
# Check dependencies
python -c "import torch; import torchvision; import sklearn; print('✅ All deps installed')"

# Quick validation (5 min total)
python src/tools/train_backbone_sanity.py --sample-size 10
python src/tools/train_classical_baseline.py --fold 0 --sample-size 100  # ← Need to add this flag

# Full training (choose one)
python src/tools/train_classical_baseline.py --fold 0      # Fold 0 only (45 min)
for i in {0..4}; do python src/tools/train_classical_baseline.py --fold $i; done  # All folds (4-5 hr)

# View results
cat reports/classical_baseline_fold0.json
```

---

**Phase 2 is ready to run. Start with fold 0 proof-of-concept, then scale to all folds.**
