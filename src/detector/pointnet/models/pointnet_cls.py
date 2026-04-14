from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2).values
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity
        return x.view(-1, self.k, self.k)


class PointNetEncoder(nn.Module):
    def __init__(self, input_channels: int = 4, feature_transform: bool = True):
        super().__init__()
        self.feature_transform = feature_transform
        self.input_transform = TNet(input_channels)
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.feature_transform_net = TNet(64) if feature_transform else None

    def forward(self, x: torch.Tensor):
        input_transform = self.input_transform(x)
        x = torch.bmm(input_transform, x)
        x = F.relu(self.bn1(self.conv1(x)))

        feature_transform = None
        if self.feature_transform_net is not None:
            feature_transform = self.feature_transform_net(x)
            x = torch.bmm(feature_transform, x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, dim=2).values
        return x, input_transform, feature_transform


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 4, feature_transform: bool = True):
        super().__init__()
        self.encoder = PointNetEncoder(input_channels=input_channels, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor):
        features, input_transform, feature_transform = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(features)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        logits = self.fc3(x)
        return logits, input_transform, feature_transform
