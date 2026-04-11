from util.trainer import Trainer
from datasets.dataloader import get_dataloader

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.kitti3D import Kitti3D
from detector.pointnet.infer import build_model

class PointnetTrainer(Trainer):
    def __init__(self, train_dataset, val_dataset, output_checkpoint, epochs, batch_size, num_points, learning_rate,
                 use_intensity):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_checkpoint = output_checkpoint
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_points = num_points
        self.learning_rate = learning_rate
        self.use_intensity = use_intensity
    
    def evaluate(self, model, loader, device):
        model.eval()
        total_examples = 0
        total_correct = 0
        with torch.no_grad():
            for batch in loader:
                points = batch["points"].to(device)
                labels = batch["labels"].to(device)
                logits, _, _ = model(points)
                predictions = logits.argmax(dim=1)
                total_examples += labels.numel()
                total_correct += (predictions == labels).sum().item()
        return 0.0 if total_examples == 0 else total_correct / total_examples

    def train(self):
        print(
            "Starting PointNet training "
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_channels = 4 if self.use_intensity else 3
        model = build_model(num_classes=4, input_channels=input_channels).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        train_loader = get_dataloader(
            self.train_dataset,
            self.batch_size,
            num_points=self.num_points,
            use_intensity=self.use_intensity,
            shuffle=True,
        )

        val_loader = get_dataloader(
          self.val_dataset,
          self.batch_size,
          num_points=self.num_points,
          use_intensity=self.use_intensity,
          shuffle=False,
        )
    
        best_val_accuracy = -1.0
        checkpoint_path = Path(self.output_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch} runninng")
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                points = batch["points"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                logits, _, _ = model(points)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            val_accuracy = self.evaluate(model, val_loader, device)
            average_loss = running_loss / max(len(train_loader), 1)
            print(f"epoch={epoch} loss={average_loss:.6f} val_accuracy={val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(
                    {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    },
                checkpoint_path,
                )