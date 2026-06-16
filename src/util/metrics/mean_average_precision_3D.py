from torchmetrics import Metric
import torch
try:
    from pytorch3d.ops import box3d_overlap
except (ImportError, OSError):
    box3d_overlap = None
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
        # Keep per-update tensors instead of concatenating them immediately.  A
        # detection metric must only match predictions with ground-truth boxes
        # from the same sample/frame; flattening all validation batches together
        # lets boxes from one frame consume ground truth from another.
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
        self.add_state("ground_truths", default=[], dist_reduce_fx="cat")
        self.add_state("classes", default=[], dist_reduce_fx="cat")
        self.iou_threshold = iou_threshold
        self.coordinate_converter = CoordinateConverter()
        self.box_mode = box_mode

    def update(self, preds: torch.Tensor, target: torch.Tensor, classes: torch.Tensor) -> None:
        self.__validate_detection_tensor(preds, "preds")
        self.__validate_detection_tensor(target, "target")
        if classes is None or classes.numel() == 0:
            return

        self.predictions.append(preds.detach())
        self.ground_truths.append(target.detach())
        self.classes.append(classes.detach().flatten())

    def compute(self):
        prediction_batches = self.__state_as_batches(self.predictions)
        ground_truth_batches = self.__state_as_batches(self.ground_truths)
        class_batches = self.__state_as_batches(self.classes)

        if len(prediction_batches) == 0 or len(ground_truth_batches) == 0 or len(class_batches) == 0:
            return torch.tensor(0.0)

        device = self.__first_state_device(prediction_batches, ground_truth_batches, class_batches)
        if not any(predictions.numel() > 0 for predictions in prediction_batches):
            return torch.tensor(0.0, device=device)
        if not any(ground_truths.numel() > 0 for ground_truths in ground_truth_batches):
            return torch.tensor(0.0, device=device)

        classes = torch.unique(torch.cat([class_values.to(device) for class_values in class_batches]))
        if classes.numel() == 0:
            return torch.tensor(0.0, device=device)

        class_wise_average_precision = []
        for class_ in classes:
            class_wise_average_precision.append(
                    self.__compute_class_average_precision(
                        prediction_batches,
                        ground_truth_batches,
                        class_.item(),
                        device,
                        )
                    )

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

            if box3d_overlap is not None:
                _, iou_3d_class = box3d_overlap(prediction_corner_boxes, ground_truth_cornder_boxes)
            else:
               raise EnvironmentError("pytroch3d is needed here") 
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
            return torch.tensor(0.0, device=device)
        return torch.stack(class_wise_average_precision).mean().cpu()

    def __compute_class_average_precision(self, prediction_batches, ground_truth_batches, class_, device):
        class_predictions_by_frame = []
        class_ground_truths_by_frame = []
        total_ground_truths = 0

        for predictions, ground_truths in zip(prediction_batches, ground_truth_batches):
            predictions = predictions.to(device)
            ground_truths = ground_truths.to(device)

            class_predictions = self.__filter_class_values(predictions, class_)
            if class_predictions.numel() > 0:
                class_predictions_by_frame.append(class_predictions)
            else:
                class_predictions_by_frame.append(torch.empty((0, 9), device=device, dtype=predictions.dtype))

            class_ground_truths = self.__filter_class_values(ground_truths, class_)
            if class_ground_truths.numel() > 0:
                class_ground_truths_by_frame.append(class_ground_truths)
                total_ground_truths += class_ground_truths.shape[0]
            else:
                class_ground_truths_by_frame.append(torch.empty((0, 9), device=device, dtype=ground_truths.dtype))

        if total_ground_truths == 0:
            return torch.tensor(0.0, device=device)

        ranked_predictions = []
        for frame_idx, frame_predictions in enumerate(class_predictions_by_frame):
            for prediction in frame_predictions:
                ranked_predictions.append((prediction[7], frame_idx, prediction))

        if len(ranked_predictions) == 0:
            return torch.tensor(0.0, device=device)

        ranked_predictions.sort(key=lambda item: item[0].item(), reverse=True)
        tp_tensor = torch.zeros(len(ranked_predictions), device=device)
        fp_tensor = torch.zeros(len(ranked_predictions), device=device)
        matched_ground_truths = [
                torch.zeros(frame_ground_truths.shape[0], dtype=torch.bool, device=device)
                for frame_ground_truths in class_ground_truths_by_frame
                ]

        for pred_idx, (_, frame_idx, prediction) in enumerate(ranked_predictions):
            frame_ground_truths = class_ground_truths_by_frame[frame_idx]
            if frame_ground_truths.numel() == 0:
                fp_tensor[pred_idx] = 1
                continue

            iou_3d_class = self.__compute_iou_3d(
                    prediction[:7].unsqueeze(0),
                    frame_ground_truths[:, :7],
                    )
            best_iou, best_gt_idx = torch.max(iou_3d_class.squeeze(0), dim=0)
            if best_iou >= self.iou_threshold and not matched_ground_truths[frame_idx][best_gt_idx]:
                tp_tensor[pred_idx] = 1
                matched_ground_truths[frame_idx][best_gt_idx] = True
            else:
                fp_tensor[pred_idx] = 1

        cumulative_tp = torch.cumsum(tp_tensor, dim=0)
        cumulative_fp = torch.cumsum(fp_tensor, dim=0)
        precision = cumulative_tp / torch.clamp(cumulative_tp + cumulative_fp, min=1e-8)
        recall = cumulative_tp / total_ground_truths
        return self.__compute_average_precision(precision, recall).to(device)

    def __collect_class_values(self, predictions, ground_truths, req_class):
        """
        This method generates two lists that filters the given predictions tensor
        and the given ground_truth tensor to two tensors only consiting of the
        predictions and the groundtruths from the given req_class.
        """
        if predictions.numel() == 0 or ground_truths.numel() == 0 or req_class is None:
            raise ValueError("Given arguments are empty.")
        self.__validate_detection_tensor(predictions, "predictions")
        self.__validate_detection_tensor(ground_truths, "ground_truths")

        class_predictions = self.__filter_class_values(predictions, req_class)
        class_ground_truths = self.__filter_class_values(ground_truths, req_class)

        if class_predictions.numel() == 0 or class_ground_truths.numel() == 0:
            return torch.empty((0, 9), device=predictions.device, dtype=predictions.dtype), torch.empty((0, 9), device=ground_truths.device, dtype=ground_truths.dtype)
        score_order = torch.argsort(class_predictions[:, 7], descending=True)
        sorted_class_predictions = class_predictions[score_order]
        return sorted_class_predictions, class_ground_truths

    def __filter_class_values(self, values, req_class):
        if values.numel() == 0:
            return torch.empty((0, 9), device=values.device, dtype=values.dtype)
        return values[values[:, 8] == req_class]

    def __validate_detection_tensor(self, values, name):
        if values is None:
            raise ValueError(f"{name} must not be None")
        if values.numel() == 0:
            return
        if values.ndim != 2 or values.shape[1] != 9:
            raise ValueError(f"{name} does not meet the required shape of (N, 9)")

    def __state_as_batches(self, state):
        if isinstance(state, list):
            return state
        if state.numel() == 0:
            return []
        return [state]

    def __first_state_device(self, *state_groups):
        for state_group in state_groups:
            for value in state_group:
                if isinstance(value, torch.Tensor):
                    return value.device
        return torch.device("cpu")

    def __compute_iou_3d(self, prediction_boxes, ground_truth_boxes):
        # Coordinate convertion needed since pytorch3d iou computation needs corner boxes.
        if box3d_overlap is not None:
            prediction_corner_boxes = self.coordinate_converter.boxes_3d_to_corners(
                    prediction_boxes,
                    self.box_mode)
            ground_truth_corner_boxes = self.coordinate_converter.boxes_3d_to_corners(
                    ground_truth_boxes,
                    self.box_mode)
            _, iou_3d_class = box3d_overlap(prediction_corner_boxes, ground_truth_corner_boxes)
            return iou_3d_class

        return self.__axis_aligned_iou_3d(prediction_boxes, ground_truth_boxes)

    def __compute_average_precision(self, precision, recall: torch.Tensor):
        metric = AveragePrecision3D()
        metric.update(
                precision=precision,
                recall=recall)
        return metric.compute()
