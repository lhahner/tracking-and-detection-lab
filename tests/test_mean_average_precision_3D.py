import importlib
import torch
import unittest

class TestMeanAveragePrecision3D(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec(name="pytorch3d") is None:
            self.skipTest("Pytorch3D is required.")

    def test__collect_class_values_correct_sorting_order(self):
        from util.metrics.mean_average_precision_3D import MeanAveragePrecision3D

        # Input
        predictions = torch.tensor([
                                [1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0.1, 2],
                                [1.6, 2.2, 2.0, 2.1, 4.1, 1.0, 0.8, 0.8, 2],
                                [1.6, 2.2, 2.0, 2.1, 4.1, 1.0, 0.8, 0.5, 3]])
        ground_truth = torch.tensor([
            [1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0, 2],
            [1.2, 2.1, 2.3, 2.5, 4.1, 1.0, 0.8, 0, 2]])
        # Output
        expected_prediction_order = torch.tensor([
            [1.6, 2.2, 2.0, 2.1, 4.1, 1.0, 0.8, 0.8, 2],
            [1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0.1, 2]])
        expected_ground_truth = torch.tensor([
            [1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0, 2],
            [1.2, 2.1, 2.3, 2.5, 4.1, 1.0, 0.8, 0, 2]
            ])
        metric = MeanAveragePrecision3D()
        sorted_class_predictions, class_ground_truths = metric._MeanAveragePrecision3D__collect_class_values(
                predictions=predictions,
                ground_truths=ground_truth,
                req_class=2)
        self.assertTrue(torch.equal(sorted_class_predictions, expected_prediction_order))
        self.assertTrue(torch.equal(class_ground_truths, expected_ground_truth))

    def test__collect_class_values_empty_predictions_and_ground_truths(self):
        from util.metrics.mean_average_precision_3D import MeanAveragePrecision3D

        predictions = torch.tensor([])
        ground_truth = torch.tensor([])
        metric = MeanAveragePrecision3D()
        with self.assertRaises(ValueError):
            metric._MeanAveragePrecision3D__collect_class_values(
                    predictions=predictions,
                    ground_truths=ground_truth,
                    req_class=2)

    def test__compute_average_precision_result_correct(self):
        precisions = torch.tensor([1.0, 1.0, 0.6667])
        recalls = torch.tensor([0.5, 1.0, 1.0])
        metric = MeanAveragePrecision3D()
        ap = metric._MeanAveragePrecision3D__compute_average_precision(
                precision=precisions,
                recall=recalls
                )
        expected_ap = torch.tensor([1.0])
        self.assertEqual(expected_ap, ap)
