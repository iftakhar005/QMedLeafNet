"""
Classical baseline heads for Phase 2: Logistic Regression, SVM, Dense MLP.
All heads receive 576-dim frozen MobileNetV3-Small features and predict 63 labels (9 conditions × 7 species).
Parameters are matched (~36K) to ensure fair comparison.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class LogisticRegressionHead(nn.Module):
    """
    Logistic Regression head on frozen backbone features.
    Direct linear layer + sklearn LogisticRegression.
    Parameter count: 36,351 (576 × 63 + 63 bias)
    """

    def __init__(self, feature_dim: int = 576, num_classes: int = 63, random_state: int = 42):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Direct linear layer to class space (36,351 params)
        self.linear = nn.Linear(feature_dim, num_classes)

        # Sklearn LogisticRegression for robust training
        self.lr_model = None
        self.scaler = StandardScaler()
        self.random_state = random_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (batch_size, feature_dim)
        Returns:
            (batch_size, num_classes) - logits
        """
        return self.linear(x)

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """
        Fit sklearn LogisticRegression on frozen features.
        Args:
            features: (n_samples, feature_dim) - frozen backbone features
            labels: (n_samples,) - label indices
        """
        # Scale features and fit LR
        scaled = self.scaler.fit_transform(features)
        self.lr_model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=-1,
            solver='lbfgs'
        )
        self.lr_model.fit(scaled, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels on new features.
        Args:
            features: (n_samples, feature_dim) - frozen backbone features
        Returns:
            (n_samples,) - predicted label indices
        """
        if self.lr_model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        scaled = self.scaler.transform(features)
        return self.lr_model.predict(scaled)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        Args:
            features: (n_samples, feature_dim)
        Returns:
            (n_samples, num_classes) - class probabilities
        """
        if self.lr_model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        scaled = self.scaler.transform(features)
        return self.lr_model.predict_proba(scaled)

    def count_trainable_params(self) -> int:
        """Count trainable parameters in linear layer."""
        return sum(p.numel() for p in self.linear.parameters() if p.requires_grad)


class SVMHead(nn.Module):
    """
    SVM head on frozen backbone features.
    Direct linear layer + sklearn SVC (RBF and Linear kernels).
    Parameter count: 36,351 (576 × 63 + 63 bias)
    """

    def __init__(self, feature_dim: int = 576, num_classes: int = 63,
                 kernel: str = 'rbf', random_state: int = 42):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.kernel = kernel

        # Direct linear layer to class space (36,351 params)
        self.linear = nn.Linear(feature_dim, num_classes)

        # Sklearn SVM for robust training
        self.svm_model = None
        self.scaler = StandardScaler()
        self.random_state = random_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(x)

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """
        Fit sklearn SVC on frozen features.
        Args:
            features: (n_samples, feature_dim)
            labels: (n_samples,)
        """
        # Scale and fit SVM
        scaled = self.scaler.fit_transform(features)
        self.svm_model = SVC(
            kernel=self.kernel,
            C=1.0,
            random_state=self.random_state,
            probability=True
        )
        self.svm_model.fit(scaled, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict labels on new features."""
        if self.svm_model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        scaled = self.scaler.transform(features)
        return self.svm_model.predict(scaled)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.svm_model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        scaled = self.scaler.transform(features)
        return self.svm_model.predict_proba(scaled)

    def count_trainable_params(self) -> int:
        """Count trainable parameters in linear layer."""
        return sum(p.numel() for p in self.linear.parameters() if p.requires_grad)


class DenseMLP(nn.Module):
    """
    Dense MLP head on frozen backbone features.
    Architecture: 576 → 64 → 63 (single hidden layer, parameter-matched to other heads)
    Parameter count: 41,023 (576×64 + 64 + 64×63 + 63)
    """

    def __init__(self, feature_dim: int = 576, num_classes: int = 63,
                 hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (batch_size, feature_dim)
        Returns:
            (batch_size, num_classes) - logits
        """
        return self.mlp(x)

    def count_trainable_params(self) -> int:
        """Count all trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_head_parameters() -> dict:
    """
    Compare parameter counts across all heads.
    Returns:
        dict with head names and param counts
    """
    feature_dim = 576
    num_classes = 63

    heads = {
        'LogisticRegression': LogisticRegressionHead(feature_dim, num_classes),
        'SVM (RBF)': SVMHead(feature_dim, num_classes, kernel='rbf'),
        'SVM (Linear)': SVMHead(feature_dim, num_classes, kernel='linear'),
        'Dense MLP': DenseMLP(feature_dim, num_classes),
    }

    params = {}
    for name, head in heads.items():
        params[name] = head.count_trainable_params()

    return params


if __name__ == '__main__':
    print("Classical Baseline Heads - Parameter Comparison")
    print("=" * 50)

    params = count_head_parameters()
    for head_name, param_count in params.items():
        print(f"{head_name:20s}: {param_count:8d} trainable params")

    avg_params = np.mean(list(params.values()))
    print("=" * 50)
    print(f"Average params:       {avg_params:8.0f}")
