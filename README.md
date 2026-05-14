# QMedLeafNet

QMedLeafNet: A Parameter-Efficient Hybrid Quantum-Classical Neural Network for Medicinal Plant Health Detection

## Phase 2: Classical Baseline Training

Phase 2 builds parameter-matched classical baselines on frozen MobileNetV3-Small features extracted from Phase 1 splits.

### Quick Start

```bash
# Install Phase 2 dependencies
pip install -r requirements_phase2.txt

# Validate backbone sanity check (already passed ✅)
python src/tools/train_backbone_sanity.py --backbone mobilenet_v3_small --sample-size 10

# Train classical baselines on fold 0 (45-60 min)
python src/tools/train_classical_baseline.py --backbone mobilenet_v3_small --fold 0
```

### Phase 2 Architecture

**Frozen Backbone:** MobileNetV3-Small (927K params)  
**Classical Heads (parameter-matched):**

- Logistic Regression: 36,351 trainable params
- SVM-RBF: 36,351 trainable params
- SVM-Linear: 36,351 trainable params
- Dense MLP: 41,023 trainable params

**Input Pipeline:**

- Phase 1 splits (background-removed 1440×1080 images)
- Aspect-ratio preserving resize → 224×224 padding
- ImageNet normalization
- Frozen MobileNetV3-Small feature extraction (576-dim vectors)
- Classical head inference (63-class output: 9 conditions × 7 species)

### Phase 2 Deliverables

#### Completed ✅

- [x] Preprocessing pipeline (resize-with-padding + ImageNet normalization)
- [x] Feature extraction infrastructure (frozen backbone)
- [x] 4 parameter-matched classical heads (LR, SVM-RBF, SVM-Linear, MLP)
- [x] Backbone sanity check (validated on 10 samples)
- [x] Training runner for classical heads
- [x] All Phase 2 dependencies installed (PyTorch 2.3+, torchvision, scikit-learn)

#### In Progress 🔄

- [ ] Proof-of-concept training on fold 0 (ready to execute)
- [ ] Classical baseline results table (all folds)
- [ ] Quantum head implementation (Phase 2.2)

### Files Added/Modified

```
src/
  utils/
    ├── __init__.py
    └── preprocessing.py          # Image loading, resizing, normalization
  models/
    ├── __init__.py
    ├── classical_heads.py         # LR, SVM, MLP implementations
    └── phase2_backbones.py        # Backbone loaders (pre-existing)
  tools/
    ├── train_backbone_sanity.py   # Backbone validation
    └── train_classical_baseline.py # Classical head trainer

requirements_phase2.txt             # PyTorch + dependencies
PHASE_2_STATUS.md                   # Detailed status report
```

### Results Tracking

Trained models save results to:

```
reports/
  ├── classical_baseline_fold0.json
  ├── classical_baseline_fold1.json
  └── ...
```

Each JSON contains metrics (accuracy, F1, precision, recall) for all 4 heads.

### Next Steps

1. **Run proof-of-concept training** (fold 0)

   ```bash
   python src/tools/train_classical_baseline.py --fold 0
   ```

2. **Generate results table** across all folds (1-4)

3. **Implement quantum heads** (Phase 2.2)

4. **Compare classical vs. quantum** performance

---

### Reference

- Phase 1 Data: 25,962 indexed images (1,981 originals + 23,981 augmented)
- 7 species × 9 conditions = 63 classification targets
- 5-fold cross-validation splits with zero parent leakage
- See [PHASE_2_STATUS.md](PHASE_2_STATUS.md) for detailed architecture and validation checklist
