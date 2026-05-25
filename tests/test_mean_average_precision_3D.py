import os
import sys
import torch
import unittest
import torch

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
MMDET3D_SRC_ROOT = os.path.join(PROJECT_ROOT, "external", "mmdetection3d-cpu-only")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if MMDET3D_SRC_ROOT not in sys.path:
    sys.path.insert(0, MMDET3D_SRC_ROOT)

from util.metrics.mean_average_precision_3D import MeanAveragePrecision3D


class TestMeanAveragePrecision3D(unittest):
    def test___collect_class_values_correct_sorting_order(self):
        predictions = torch.tensor([[1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0.1, 2],
                                    [1.6, 2.2, 2.0, 2.1, 4.1, 1.0, 0.8, 0.8, 2],
                                    [1.6, 2.2, 2.0, 2.1, 4.1, 1.0, 0.8, 0.5, 3]])

        ground_truth = torch.tensor([[1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0, 2],
                                     [1.2, 2.1, 2.3, 2.5, 4.1, 1.0, 0.8, 0, 2]
                                     ])
        classes = torch.tensor([1, 2, 3])
        expected_prediction_order = torch.tensor([
            [1.6, 2.2, 2.0, 2.1, 4.1, 1.0, 0.8, 0.8, 2],
            [1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0.1, 2]])
        metric = MeanAveragePrecision3D(
                predictions=predictions,
                ground_truth=ground_truth,
                classes=classes)
        sorted_class_predictions, class_ground_truths = metric.__collect_class_values(
                predictions=predictions,
                ground_truths=ground_truth,
                req_class=2)
        self.assertTrue(torch.equal(sorted_class_predictions, expected_prediction_order))
        self.assertTrue(torch.equal(class_ground_truths, ground_truth))

    def test__collect_class_values_empty_predictions_and_ground_truths(self):
        predictions = torch.empty(3, 9)
        ground_truth = torch.empty(3, 9)
        classes = torch.tensor([1, 2, 3])
        metric = MeanAveragePrecision3D(
                predictions=predictions,
                ground_truth=ground_truth,
                classes=classes)
        with self.assertRaises(ValueError):
            metric.__collect_class_values(
                predictions=predictions,
                ground_truths=ground_truth,
                req_class=2)

    def test__compute_average_precision_result_correct(self):
        # TODO test expected AP value for this case
        predictions = torch.tensor([
                                    [1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0.1, 2],
                                    [1.6, 2.2, 2.0, 2.1, 4.1, 1.0, 0.8, 0.8, 2],
                                    [1.6, 2.2, 2.0, 2.1, 4.1, 1.0, 0.8, 0.5, 3]])

        ground_truth = torch.tensor([[1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0, 2],
                                     [1.2, 2.1, 2.3, 2.5, 4.1, 1.0, 0.8, 0, 2]
                                     ])
        classes = torch.tensor([1, 2, 3])
        metric = MeanAveragePrecision3D(
                predictions=predictions,
                ground_truth=ground_truth,
                classes=classes)

        precisions = torch.tensor([])
        recall = torch.tensor([])
        ap = metric.__compute_average_precision(
                precision=precisions,
                recall=recall
                )
        expected_ap = torch.tensor([])
        self.assertEqual(expected_ap, ap)
