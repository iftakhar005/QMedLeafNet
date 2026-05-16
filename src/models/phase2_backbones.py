"""TorchVision backbone helpers for Phase 2 baseline checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    feature_dim: int


BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "mobilenet_v3_small": BackboneSpec("mobilenet_v3_small", 576),
    "mobilenet_v2": BackboneSpec("mobilenet_v2", 1280),
    "efficientnet_b0": BackboneSpec("efficientnet_b0", 1280),
    "resnet18": BackboneSpec("resnet18", 512),
}


def get_backbone_spec(name: str) -> BackboneSpec:
    try:
        return BACKBONE_SPECS[name]
    except KeyError as exc:
        available = ", ".join(sorted(BACKBONE_SPECS))
        raise ValueError(
            f"Unsupported backbone '{name}'. Available: {available}") from exc


def _require_torchvision():
    try:
        import torchvision.models as models
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "torchvision is required for Phase 2 backbone sanity checks"
        ) from exc
    return models


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "torch is required for Phase 2 backbone sanity checks") from exc
    return torch, nn


def build_feature_extractor(backbone_name: str):
    """Build a TorchVision model truncated before the classifier head."""
    models = _require_torchvision()
    _, nn = _require_torch()
    spec = get_backbone_spec(backbone_name)

    if spec.name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        return model.features, spec.feature_dim
    if spec.name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        return model.features, spec.feature_dim
    if spec.name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        return model.features, spec.feature_dim
    if spec.name == "resnet18":
        model = models.resnet18(weights=None)
        layers = list(model.children())[:-1]
        return nn.Sequential(*layers), spec.feature_dim

    raise AssertionError(f"Unhandled backbone: {backbone_name}")


def count_parameters(module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return sum(parameter.numel() for parameter in module.parameters())


def build_classification_head(input_dim: int, num_classes: int):
    """Build a small linear classification head for sanity checks."""
    _, nn = _require_torch()
    return nn.Linear(input_dim, num_classes)


def freeze_module(module) -> None:
    """Freeze all parameters in a module."""
    for parameter in module.parameters():
        parameter.requires_grad = False
