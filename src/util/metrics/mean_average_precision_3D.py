from torchmetrics import Metric, Precision, Recall
import torch
from pytorch3d.ops import box3d_overlap
from util.coordinate_converter import CoordinateConverter
from util.metrics.average_precision_3D import AveragePrecision3D

class MeanAveragePrecision3D(Metric):
    r"""Compute the `Mean-Average-Precision (mAP) for 3D object detection predictions.

    Args:
        predictions:
            Input to be evaluated, the Inpute requires to a tensor where each row represents
            one detections and the values correspond to exactly nine values.

        ground_truths:
            Ground truth values where the predictions are evaulated on, it requires to be a
            tensor of the size of the number of required predictions with the values.

        classes:
            Number of classes in that given dataset

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``predictions`` or ``ground_truth`` is not of shape (N, 9)
        ValueError:
            If ``predictions`` or ``ground_truth`` is not empty

    Example::
        Basic example on how to use this metric internally

        >>> predictions = torch.randn(3, 9)
        >>> ground_truth = torch.randn(2, 9)
        >>> classes = torch.tenosr([1, 2, 3])
        >>> metric = MeanAveragePrecision3D()
        >>> metric.update(
        >>>         predictions=predictions,
        >>>         ground_truths=ground_truths,
        >>>         classes=classes)
        >>> metric.compute()
    """
    def __init__(self, box_mode="camera", iou_threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.add_state("predictions", default=torch.tensor(0), dist_reduce_fx="cat")
        self.add_state("ground_truths", default=torch.tensor(0), dist_reduce_fx="cat")
        self.add_state("classes", default=torch.tensor(0), dist_reduce_fx="cat")
        self.iou_threshold = iou_threshold
        self.coordinate_converter = CoordinateConverter()
        self.box_mode = box_mode

    def update(self, preds: torch.tensor, target: torch.tensor, classes: torch.tensor) -> None:
        self.predictions = preds
        self.ground_truths = target
        self.classes = classes

    def compute(self):
        if self.predictions.numel() == 0:
            return torch.tensor(0.0)

        if self.classes.numel() == 0:
            return torch.tensor(0.0)
        class_wise_average_precision = []

        for class_ in self.classes:
            class_wise_true_positives = []
            class_wise_false_negatives = []
            # Filter from all predictions to lists containing predictions and gt values for this class only
            class_predictions, class_ground_truths = self.__collect_class_values(self.predictions,
                                                                                 self.ground_truths,
                                                                                 class_.item())
            if class_predictions.numel() == 0 or class_ground_truths.numel() == 0:
                class_wise_average_precision.append(torch.tensor(0.0).cpu())
                continue
            # Coordinate convertion need since pytorch3d iou computation needs corner boxes
            prediction_corner_boxes = self.coordinate_converter.boxes_3d_to_corners(
                    class_predictions[:, :7],
                    self.box_mode)
            ground_truth_cornder_boxes = self.coordinate_converter.boxes_3d_to_corners(
                    class_ground_truths[:, :7],
                    self.box_mode)

            _, iou_3d_class = box3d_overlap(prediction_corner_boxes, ground_truth_cornder_boxes)
            # Considering that we only consider the preds with the gt that has has the highest IoU
            best_iou, best_gt_idx = torch.max(iou_3d_class, dim=1)
            tp_tensor = torch.zeros(best_gt_idx.shape[0])
            fp_tensor = torch.zeros(best_gt_idx.shape[0])
            matched_ground_truths = torch.zeros(class_ground_truths.shape[0], dtype=torch.bool)

            # Now looping over each ious for each predictions; 3 preds = 3 best ious = 3 iterations
            for pred_idx, (iou, gt_idx) in enumerate(zip(best_iou, best_gt_idx)):
                if iou >= self.iou_threshold and not matched_ground_truths[gt_idx]:
                    class_wise_true_positives.append(class_predictions[pred_idx])
                    tp_tensor[pred_idx] = 1
                    matched_ground_truths[gt_idx] = True
                else:
                    fp_tensor[pred_idx] = 1
                    class_wise_false_negatives.append(class_predictions[pred_idx])

            # Cumlative sum to prevent divison by 0  and to reflect the descending recall whenever we match too much.
            cumulative_tp = torch.cumsum(tp_tensor, dim=0)
            cumulative_fp = torch.cumsum(fp_tensor, dim=0)
            precision = cumulative_tp / torch.clamp(cumulative_tp + cumulative_fp, min=1e-8)
            recall = cumulative_tp / len(class_ground_truths)  # total number of ground_truth_classes
            # Compute AP, initally the integration of the trade-off curve between recall and precision
            class_wise_average_precision.append(self.__compute_average_precision(precision, recall))
        if len(class_wise_average_precision) == 0:
            return torch.tensor(0.0)
        return torch.stack(class_wise_average_precision).cpu().mean()

    def __collect_class_values(self, predictions, ground_truths, req_class):
        """
        This method generates two lists that filters the given predictions tensor
        and the given ground_truth tensor to two tensors only consiting of the 
        predictions and the groundtruths from the given req_class.
        """
        if predictions.numel() == 0 or ground_truths.numel() == 0 or req_class is None:
            raise ValueError("Given arguments are empty.")
        if predictions.shape[1] != 9 or ground_truths.shape[1] != 9:
            raise ValueError("The given tensor dos not meet the required shape of (N, 9)")

        class_predictions: list = []
        class_ground_truths: list = []
        for prediction in predictions:
            if req_class == prediction[8].item():
                class_predictions.append(prediction)

        for ground_truth in ground_truths:
            if req_class == ground_truth[8].item():
                class_ground_truths.append(ground_truth)

        if len(class_predictions) == 0 or len(class_ground_truths) == 0:
            return torch.tensor([]), torch.tensor([])
        class_predictions_tensor = torch.stack(class_predictions)
        score_order = torch.argsort(class_predictions_tensor[:, 7], descending=True)
        sorted_class_predictions = class_predictions_tensor[score_order]
        return sorted_class_predictions, torch.stack(class_ground_truths)

    def __compute_average_precision(self, precision, recall: torch.Tensor):
        metric = AveragePrecision3D()
        metric.update(
                precision=precision,
                recall=recall)
        return metric.compute()
