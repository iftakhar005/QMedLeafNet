"""
Train and evaluate classical baseline heads on Phase 1 splits.
Extracts frozen MobileNetV3-Small features, trains heads using sklearn/torch.
Evaluates on dev and test splits.
"""

from src.utils.preprocessing import preprocess_image
from src.models.phase2_backbones import build_feature_extractor, freeze_module
from src.models.classical_heads import LogisticRegressionHead, SVMHead, DenseMLP
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def extract_features(
    image_paths: list,
    backbone: nn.Module,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Extract frozen backbone features for a batch of images.
    Args:
        image_paths: List of image file paths
        backbone: Frozen pretrained backbone
        device: 'cpu' or 'cuda'
    Returns:
        (n_samples, feature_dim) feature array
    """
    backbone.to(device)
    backbone.eval()

    features_list = []

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Extracting features", leave=False):
            try:
                # Load and preprocess image (returns CHW numpy array)
                img_np = preprocess_image(str(img_path))  # (3, 224, 224)
                img_tensor = torch.from_numpy(img_np).unsqueeze(
                    0).to(device)  # (1, 3, 224, 224)

                # Extract features
                feats = backbone(img_tensor)
                # Global average pool if needed
                if len(feats.shape) == 4:
                    feats = torch.nn.functional.adaptive_avg_pool2d(
                        feats, (1, 1))
                    feats = feats.view(feats.size(0), -1)

                features_list.append(feats.squeeze().cpu().numpy())
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Return zeros on error (576 dims for MobileNetV3-Small)
                features_list.append(np.zeros(576))

    return np.array(features_list, dtype=np.float32)


def train_classical_baseline(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    head_type: str = 'logistic_regression',
    device: str = 'cpu'
) -> dict:
    """
    Train a classical head and evaluate on validation set.
    Args:
        train_features: (n_train, 576) feature array
        train_labels: (n_train,) label array
        val_features: (n_val, 576) feature array
        val_labels: (n_val,) label array
        head_type: 'logistic_regression', 'svm_rbf', 'svm_linear', or 'dense_mlp'
        device: 'cpu' or 'cuda'
    Returns:
        dict with metrics (accuracy, f1, precision, recall)
    """
    print(f"\n{'='*60}")
    print(f"Training {head_type}")
    print(
        f"Train samples: {len(train_features)}, Val samples: {len(val_features)}")
    print(f"{'='*60}")

    if head_type == 'logistic_regression':
        head = LogisticRegressionHead()
        head.fit(train_features, train_labels)
        val_preds = head.predict(val_features)

    elif head_type == 'svm_rbf':
        head = SVMHead(kernel='rbf')
        head.fit(train_features, train_labels)
        val_preds = head.predict(val_features)

    elif head_type == 'svm_linear':
        head = SVMHead(kernel='linear')
        head.fit(train_features, train_labels)
        val_preds = head.predict(val_features)

    elif head_type == 'dense_mlp':
        from sklearn.preprocessing import StandardScaler

        head = DenseMLP().to(device)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Feature scaling (CRITICAL FIX: sklearn heads use StandardScaler too)
        scaler_obj = StandardScaler()
        train_features_scaled = scaler_obj.fit_transform(train_features)
        val_features_scaled = scaler_obj.transform(val_features)

        # Prepare data with scaled features
        train_tensor = torch.from_numpy(
            train_features_scaled).float().to(device)
        train_labels_tensor = torch.from_numpy(train_labels).long().to(device)
        val_tensor = torch.from_numpy(val_features_scaled).float().to(device)

        # Batch-based training loop with early stopping
        batch_size = 128
        num_epochs = 500
        best_val_acc = 0
        patience = 30
        patience_counter = 0

        head.train()
        for epoch in range(num_epochs):
            # Training phase with mini-batches
            for i in range(0, len(train_tensor), batch_size):
                batch_X = train_tensor[i:i+batch_size]
                batch_y = train_labels_tensor[i:i+batch_size]

                optimizer.zero_grad()
                logits = head(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            # Validation phase
            head.eval()
            with torch.no_grad():
                val_logits = head(val_tensor)
                val_preds_temp = val_logits.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(val_labels, val_preds_temp)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            head.train()

        # Final prediction on validation set
        head.eval()
        with torch.no_grad():
            logits = head(val_tensor)
            val_preds = logits.argmax(dim=1).cpu().numpy()

    else:
        raise ValueError(f"Unknown head type: {head_type}")

    # Compute metrics
    accuracy = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
    precision = precision_score(
        val_labels, val_preds, average='weighted', zero_division=0)
    recall = recall_score(val_labels, val_preds,
                          average='weighted', zero_division=0)

    metrics = {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
    }

    print(f"Val Accuracy: {accuracy:.4f}")
    print(f"Val F1 (weighted): {f1:.4f}")
    print(f"Val Precision (weighted): {precision:.4f}")
    print(f"Val Recall (weighted): {recall:.4f}")

    return metrics


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load manifest and ontology
    manifest_df = pd.read_csv(args.manifest)
    with open(args.ontology) as f:
        ontology = json.load(f)

    # Load split manifests (fold 0 for demo)
    fold = args.fold
    train_split_path = args.data_dir / \
        f'manifests/splits_v1/fold_{fold}_train.csv'
    val_split_path = args.data_dir / f'manifests/splits_v1/fold_{fold}_val.csv'

    train_split = pd.read_csv(train_split_path)
    val_split = pd.read_csv(val_split_path)

    print(f"Fold {fold}: Train={len(train_split)}, Val={len(val_split)}")

    # Map labels to indices
    condition_map = {c: i for i, c in enumerate(ontology['conditions'])}
    species_map = {s: i for i, s in enumerate(ontology['species'])}

    def get_label_index(condition, species):
        """Map (condition, species) to 63-class index."""
        c_idx = condition_map[condition]
        s_idx = species_map[species]
        return c_idx * len(ontology['species']) + s_idx

    # Load train/val labels
    train_labels = np.array([get_label_index(r['condition'], r['species'])
                             for _, r in train_split.iterrows()])
    val_labels = np.array([get_label_index(r['condition'], r['species'])
                           for _, r in val_split.iterrows()])

    # Load backbone
    backbone, feature_dim = build_feature_extractor(args.backbone)
    freeze_module(backbone)

    print(f"Backbone: {args.backbone}")
    print(f"Feature dim: {feature_dim}")

    # Extract features
    print("\nExtracting training features...")
    train_image_paths = [args.data_dir /
                         path for path in train_split['file_path']]
    train_features = extract_features(train_image_paths, backbone, device)

    print("Extracting validation features...")
    val_image_paths = [args.data_dir / path for path in val_split['file_path']]
    val_features = extract_features(val_image_paths, backbone, device)

    # Train classical baselines
    results = {}
    for head_type in ['logistic_regression', 'svm_rbf', 'svm_linear', 'dense_mlp']:
        metrics = train_classical_baseline(
            train_features, train_labels,
            val_features, val_labels,
            head_type=head_type,
            device=device
        )
        results[head_type] = metrics

    # Print summary
    print(f"\n{'='*60}")
    print("CLASSICAL BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Head Type':<25} {'Accuracy':<12} {'F1 (weighted)':<12}")
    print(f"{'-'*60}")
    for head_type, metrics in results.items():
        acc = metrics['accuracy']
        f1 = metrics['f1']
        print(f"{head_type:<25} {acc:>10.4f}   {f1:>10.4f}")

    # Save results
    results_file = args.data_dir / \
        f'reports/classical_baseline_fold{fold}.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train classical baseline heads on Phase 1 splits')
    parser.add_argument('--data-dir', type=Path, default=Path.cwd(),
                        help='Data directory (default: current directory)')
    parser.add_argument('--manifest', type=Path, default='manifests/master_manifest.csv',
                        help='Path to master manifest CSV')
    parser.add_argument('--ontology', type=Path, default='reports/label_ontology.json',
                        help='Path to label ontology JSON')
    parser.add_argument('--backbone', type=str, default='mobilenet_v3_small',
                        choices=['mobilenet_v3_small', 'mobilenet_v2',
                                 'efficientnet_b0', 'resnet18'],
                        help='Backbone model')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index (0-4)')

    args = parser.parse_args()

    # Resolve absolute paths
    if not args.manifest.is_absolute():
        args.manifest = args.data_dir / args.manifest
    if not args.ontology.is_absolute():
        args.ontology = args.data_dir / args.ontology

    main(args)
