import csv
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import pytest

from entities.detection import Detection, DetectionSequence, FrameDetection
from evaluation import Evaluation


class TestEvaluationIoUAndAnalysis:
    def build_detection_sequence(self):
        frame = FrameDetection(
            frame="sample-1",
            highest_score_index=0,
            dets=[
                Detection(
                    score=torch.tensor(0.91),
                    label=4,
                    box=torch.tensor([1.0, 2.0, 3.0, 4.0, 1.8, 1.6, 0.1]),
                ),
                Detection(
                    score=torch.tensor(0.42),
                    label=7,
                    box=torch.tensor([6.0, 2.5, 1.0, 0.8, 0.7, 1.7, 0.0]),
                ),
            ],
            targets=[
                {
                    "type": "Cyclist",
                    "label": 2,
                    "box": np.asarray([1.0, 2.0, 3.0, 4.0, 1.8, 1.6, 0.1], dtype=np.float32),
                },
                {
                    "type": "Car",
                    "label": 3,
                    "box": np.asarray([10.0, 1.0, 1.0, 4.5, 1.9, 1.5, 0.0], dtype=np.float32),
                },
            ],
        )
        return DetectionSequence(frames=[frame])

    def install_fake_pytorch3d(self, iou_matrix):
        fake_root = types.ModuleType("pytorch3d")
        fake_ops = types.ModuleType("pytorch3d.ops")

        def fake_box3d_overlap(prediction_corner_boxes, ground_truth_corner_boxes):
            volume = torch.zeros((prediction_corner_boxes.shape[0], ground_truth_corner_boxes.shape[0]))
            return volume, iou_matrix.clone()

        fake_ops.box3d_overlap = fake_box3d_overlap
        return patch.dict(sys.modules, {"pytorch3d": fake_root, "pytorch3d.ops": fake_ops})
