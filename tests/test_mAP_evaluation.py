import torch
import unittest
from util.evaluation import Evaluation
from util.metrics.mean_average_precision_3D import MeanAveragePrecision3D


class TestmAPEvaluation(unittest.TestCase):
    def test_simple_mAP_evaluation(self):
        # Each list item is one frame with all predictions.
        # x,y,z,dx,dy,dz,yaw,score,laebl
        predictions = torch.tensor([[1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0.8, 2],
                                    [1.6, 2.2, 2.0, 2.1, 4.1, 1.0, 0.8, 0.7, 2],
                                    [1.6, 2.2, 2.0, 2.1, 4.1, 1.0, 0.8, 0.1, 3]])

        ground_truth = torch.tensor([[1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0.8, 2],
                                     [1.2, 2.1, 2.3, 2.5, 4.1, 1.0, 0.8, 0.7, 2],
                                     [1.1, 2.1, 2.5, 2.8, 4.1, 1.0, 0.8, 0.1, 3]])
        classes = torch.tensor([1, 2, 3])
        evaluation = Evaluation()
        results = evaluation.compute_mAP_3D(predicted_detections=predictions,
                                            ground_truth=ground_truth,
                                            classes=classes)
        self.assertIsInstance(results, torch.Tensor)

    def test_perfect_detection_returns_one(self):
        predictions = torch.tensor([[1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0.8, 2]])
        ground_truth = torch.tensor([[1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 1.0, 2]])
        classes = torch.tensor([2])

        evaluation = Evaluation()
        results = evaluation.compute_mAP_3D(
                predicted_detections=predictions,
                ground_truth=ground_truth,
                classes=classes)

        self.assertTrue(torch.isclose(results, torch.tensor(1.0)))

    def test_prediction_order_does_not_change_map(self):
        sorted_predictions = torch.tensor([[1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0.9, 2],
                                           [8.0, 8.0, 8.0, 2.2, 4.3, 1.1, 0.8, 0.2, 2]])
        unsorted_predictions = torch.tensor([
            [8.0, 8.0, 8.0, 2.2, 4.3, 1.1, 0.8, 0.2, 2],
            [1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0.9, 2]])
        ground_truth = torch.tensor([[1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 1.0, 2]])
        classes = torch.tensor([2])

        evaluation = Evaluation()
        sorted_results = evaluation.compute_mAP_3D(
                predicted_detections=sorted_predictions,
                ground_truth=ground_truth,
                classes=classes)
        unsorted_results = evaluation.compute_mAP_3D(
                predicted_detections=unsorted_predictions,
                ground_truth=ground_truth,
                classes=classes)

        self.assertTrue(torch.isclose(sorted_results, unsorted_results))

    def test_duplicate_predictions_do_not_inflate_map_above_one(self):
        predictions = torch.tensor([
            [1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0.9, 2],
            [1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 0.8, 2]])
        ground_truth = torch.tensor([[1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 1.0, 2]])
        classes = torch.tensor([2])

        evaluation = Evaluation()
        results = evaluation.compute_mAP_3D(
                predicted_detections=predictions,
                ground_truth=ground_truth,
                classes=classes)

        self.assertTrue(torch.isclose(results, torch.tensor(1.0)))

    def test_empty_predictions_return_zero(self):
        predictions = torch.empty((0, 9))
        ground_truth = torch.tensor([[1.8, 2.7, 3.1, 2.2, 4.3, 1.1, 0.8, 1.0, 2]])
        classes = torch.tensor([2])

        evaluation = Evaluation()
        result = evaluation.compute_mAP_3D(
                    predicted_detections=predictions,
                    ground_truth=ground_truth,
                    classes=classes)
        self.assertEqual(result, torch.tensor([0]))

    def test_metric_accumulates_validation_batches(self):
        classes = torch.tensor([2])
        metric = MeanAveragePrecision3D(iou_threshold=0.5)

        metric.update(
                preds=torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.9, 2]]),
                target=torch.tensor([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2]]),
                classes=classes,
                )
        metric.update(
                preds=torch.empty((0, 9)),
                target=torch.tensor([[10.0, 10.0, 10.0, 2.0, 2.0, 2.0, 0.0, 1.0, 2]]),
                classes=classes,
                )

        result = metric.compute()

        self.assertTrue(torch.isclose(result, torch.tensor(0.5)), result)
