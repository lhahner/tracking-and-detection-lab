from __future__ import annotations

from pathlib import Path

import torch

from detector.pointnet.models.pointnet_cls import PointNetClassifier


def build_model(num_classes: int, input_channels: int = 4) -> PointNetClassifier:
    return PointNetClassifier(num_classes=num_classes, input_channels=input_channels)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_crops(model: torch.nn.Module, batch: torch.Tensor, device: torch.device):
    batch = batch.to(device)
    logits, _, _ = model(batch)
    probabilities = torch.softmax(logits, dim=1)
    scores, labels = probabilities.max(dim=1)
    return labels.cpu(), scores.cpu(), probabilities.cpu()
