import os
import sys
import torch
import unittest
import torch
from pathlib import Path

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
MMDET3D_SRC_ROOT = os.path.join(PROJECT_ROOT, "external", "mmdetection3d-cpu-only")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if MMDET3D_SRC_ROOT not in sys.path:
    sys.path.insert(0, MMDET3D_SRC_ROOT)

from util.evaluation import Evaluation


class TestEvaluationmAP3D(unittest.TestCase):
    def test_simple_precision_evaluation(self):
        predictions: list = [
            dict(
                boxes=torch.tensor([]),
                scores=torch.tensor([]),
                labels=torch.tensor([])
            )
        ]

        ground_truth: list = [
                dict(
                    boxes=torch.tensor([]),
                    scores=torch.tensor([]),
                    labels=torch.tensor([])
                )
        ]

        evaluation = Evaluation()
        results = evaluation.compute_mAP_3D(predicted_detections=predictions, ground_truth=ground_truth)
        self.assertTrue(len(results) > 0)  # Smoke test
