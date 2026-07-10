from __future__ import annotations
from pathlib import Path
import csv
import datetime

try:
    import numpy as np
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda values, dtype=float: np.asarray(values, dtype=dtype)
except ImportError:  # pragma: no cover
    np = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import motmetrics as mm
except ImportError:  # pragma: no cover
    mm = None

try:
    import torchmetrics
except ImportError:  # pragma: no cover
    torchmetrics = None

from config.logging_config import LoggingConfig

logging_config = LoggingConfig()
logger = logging_config.get_logger(__name__)


class Evaluation:
    """
    API for evaluation of various metrics based on
    torchmetrics or motmetrics package.
    """
    def __init__(self, iou_threshold=0.5):
        """Create an evaluation helper for MOT-style tracking metrics.

        Args:
            iou_threshold: Minimum IoU required for a valid match.
        """
        self.iou_threshold = iou_threshold
        self.metrics_handler = mm.metrics.create() if mm is not None else None

    def _resolve_mot_file_path(self, file_path):
        """Resolve a MOT text file path from either a file or directory input."""
        resolved_path = Path(file_path)
        if resolved_path.is_dir():
            candidate = resolved_path / "gt.txt"
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"No gt.txt found in directory: {resolved_path}")
        return resolved_path

    def read_mot_file(self,
                      file_path,
                      filter_ground_truth_by_confidence=False,
                      allowed_class_ids=None):
        """Read a MOT-format file and group detections by frame.

        Args:
            file_path: Path to a MOT-format text file.
            filter_ground_truth_by_confidence: Whether to discard rows with
                non-positive confidence values.
            allowed_class_ids: Optional set of class IDs to keep.

        Returns:
            dict[int, list[tuple[int, list[float]]]]: Mapping of frame number to
            object IDs and bounding boxes in `xywh` format.
        """
        resolved_path = self._resolve_mot_file_path(file_path)
        detections_per_frame = {}
        if np is not None:
            mot_rows = np.loadtxt(resolved_path, delimiter=",")
            if mot_rows.ndim == 1:
                mot_rows = mot_rows.reshape(1, -1)
        else:
            with open(resolved_path, "r", encoding="utf-8") as handle:
                mot_rows = [
                    [float(value) for value in row]
                    for row in csv.reader(handle)
                    if row
                ]
        for mot_row in mot_rows:
            if filter_ground_truth_by_confidence and len(mot_row) > 6 and mot_row[6] <= 0:
                continue

            if allowed_class_ids is not None and len(mot_row) > 7:
                class_id = int(mot_row[7])
                # MOT15-style files often store -1 placeholders instead of a semantic class.
                if class_id >= 0 and class_id not in allowed_class_ids:
                    continue
            frame_number = int(mot_row[0])
            object_id = int(mot_row[1])
            bounding_box_xywh = [mot_row[2], mot_row[3], mot_row[4], mot_row[5]]
            detections_per_frame.setdefault(frame_number, []).append((
                object_id, bounding_box_xywh))
        return detections_per_frame

    def should_filter_ground_truth_to_pedestrians(self, sequence_name):
        pedestrian_sequences = (
                "KITTI-",
                "MOT",
                "ETH-",
                "TUD-",
                "PETS",
                "ADL-",
                "VENICE-",
                )
        normalized_name = str(sequence_name).upper()
        return normalized_name.startswith(pedestrian_sequences)

    def create_mot_accumulator(self):
        if mm is None:
            return []
        return mm.MOTAccumulator(auto_id=False)

    def convert_trackers_to_mot_items(self, trackers):
        if np is not None:
            tracker_rows = np.asarray(trackers, dtype=float)
            if tracker_rows.size == 0:
                return []
            if tracker_rows.ndim == 1:
                tracker_rows = tracker_rows.reshape(1, -1)
        else:
            tracker_rows = trackers or []
            if tracker_rows and not isinstance(tracker_rows[0], (list, tuple)):
                tracker_rows = [tracker_rows]

        mot_items = []
        for tracker_row in tracker_rows:
            if tracker_row.size < 5:
                continue

            x1, y1, x2, y2, track_id = tracker_row[:5]
            mot_items.append((int(track_id), [x1, y1, x2 - x1, y2 - y1]))

        return mot_items

    def compute_cumulative_tracking_metrics(self,
                                            mot_accumulator,
                                            frame_number,
                                            ground_truth_by_frame,
                                            trackers,
                                            sequence_name="sequence"):
        ground_truth_items = ground_truth_by_frame.get(frame_number, [])
        predicted_items = self.convert_trackers_to_mot_items(trackers)

        ground_truth_ids = [item[0] for item in ground_truth_items]
        predicted_ids = [item[0] for item in predicted_items]

        ground_truth_boxes_xywh = [item[1] for item in ground_truth_items]
        predicted_boxes_xywh = [item[1] for item in predicted_items]

        maximum_iou_distance = 1.0 - self.iou_threshold
        iou_distance_matrix = mm.distances.iou_matrix(
                ground_truth_boxes_xywh,
                predicted_boxes_xywh,
                max_iou=maximum_iou_distance,
                )

        mot_accumulator.update(
                ground_truth_ids,
                predicted_ids,
                iou_distance_matrix,
                frameid=frame_number,
                )

        metrics = ["idf1", "mota", "motp"]
        summary = self.metrics_handler.compute(
                mot_accumulator,
                metrics=metrics,
                name=sequence_name,
                )

        metric_row = summary.loc[sequence_name]
        return {metric: metric_row[metric] for metric in metrics}

    def get_metric_value(self, evaluation_summary, metric_name, sequence_name="sequence"):
        if hasattr(evaluation_summary, "loc"):
            return float(evaluation_summary.loc[sequence_name][metric_name])
        if isinstance(evaluation_summary, list):
            for row in evaluation_summary:
                if row.get("sequence", sequence_name) == sequence_name:
                    return float(row[metric_name])
        raise KeyError(f"Metric '{metric_name}' not found in evaluation summary")
    
    def evaluate_simpletrack_nuscenes_result_file(
            self,
            result_path,
            dataroot,
            version="v1.0-mini",
            eval_set="mini_val",
            output_dir="output/nuscenes_tracking_eval"
        ):
        from nuscenes.eval.tracking.evaluate import TrackingEval

        try:
            from nuscenes.eval.common.config import config_factory
        except ImportError:
            from nuscenes.eval.tracking.config import config_factory

        cfg = config_factory("tracking_nips_2019")
        evaluator = TrackingEval(
                config=cfg,
                result_path=str(result_path),
                eval_set=eval_set,
                output_dir=str(output_dir),
                nusc_dataroot=str(dataroot),
                nusc_version=version,
                verbose=True
                )
        return evaluator.main(render_curves=False)

    def evaluate_sequence(self, ground_truth_file_path,
                          predicted_tracking_file_path,
                          sequence_name="sequence",
                          metrics=None):
        if metrics is None:
            metrics = ["idf1", "mota", "motp", "precision", "recall"]

        allowed_ground_truth_class_ids = {1} if self.should_filter_ground_truth_to_pedestrians(sequence_name) else None
        ground_truth_by_frame = self.read_mot_file(
                ground_truth_file_path,
                filter_ground_truth_by_confidence=True,
                allowed_class_ids=allowed_ground_truth_class_ids,
                )
        predicted_tracks_by_frame = self.read_mot_file(predicted_tracking_file_path, filter_ground_truth_by_confidence=False)

        if mm is None or self.metrics_handler is None:
            total_gt = sum(len(items) for items in ground_truth_by_frame.values())
            total_pred = sum(len(items) for items in predicted_tracks_by_frame.values())
            matched = 0
            for frame_number in sorted(set(ground_truth_by_frame) | set(predicted_tracks_by_frame)):
                gt_items = {(item[0], tuple(item[1])) for item in ground_truth_by_frame.get(frame_number, [])}
                pred_items = {(item[0], tuple(item[1])) for item in predicted_tracks_by_frame.get(frame_number, [])}
                matched += len(gt_items & pred_items)
            precision = matched / total_pred if total_pred else 0.0
            recall = matched / total_gt if total_gt else 0.0
            mota = matched / total_gt if total_gt else 0.0
            return [{
                "sequence": sequence_name,
                "idf1": precision,
                "mota": mota,
                "motp": 0.0,
                "precision": precision,
                "recall": recall,
            }]

        mot_accumulator = mm.MOTAccumulator(auto_id=False)  # MotMetric setup
        maximum_iou_distance = 1.0 - self.iou_threshold  # default to 0.5
        for frame_number in sorted(set(ground_truth_by_frame) | set(predicted_tracks_by_frame)):
            ground_truth_items = ground_truth_by_frame.get(frame_number, [])
            predicted_items = predicted_tracks_by_frame.get(frame_number, [])
            ground_truth_ids = [item[0] for item in ground_truth_items]
            predicted_ids = [item[0] for item in predicted_items]
            ground_truth_boxes_xywh = [item[1] for item in ground_truth_items]
            predicted_boxes_xywh = [item[1] for item in predicted_items]
            iou_distance_matrix = mm.distances.iou_matrix(ground_truth_boxes_xywh,
                                                          predicted_boxes_xywh,
                                                          max_iou=maximum_iou_distance)
            mot_accumulator.update(
                    ground_truth_ids,
                    predicted_ids,
                    iou_distance_matrix,
                    frameid=frame_number)
        return self.metrics_handler.compute(mot_accumulator,
                                            metrics,
                                            name=sequence_name
                                            )

    def presist_evaluation(self, evaluation_summary, dataset, detector_name, tracking_name):
        """Persist an evaluation summary to a timestamped benchmark file.

        Args:
            evaluation_summary: Evaluation result object or string summary.
            dataset: Dataset name included in the output filename and content.
            detector_name: Detector name included in the output metadata.
            tracking_name: Tracker name included in the output metadata.

        Returns:
            Path: Path to the written benchmark file.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_dataset = str(dataset).replace("/", "_")
        safe_detector = str(detector_name).replace("/", "_")
        safe_tracker = str(tracking_name).replace("/", "_")

        benchmark_dir = Path(__file__).resolve().parents[2] / "data" / "benchmark"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        benchmark_filename = f"{timestamp}-{safe_dataset}-{safe_detector}-{safe_tracker}.txt"
        benchmark_file_path = benchmark_dir / benchmark_filename

        with open(benchmark_file_path, "w", encoding="utf-8") as benchmark_file:
            benchmark_file.write(f"timestamp: {timestamp}\n")
            benchmark_file.write(f"dataset: {dataset}\n")
            benchmark_file.write(f"detector: {detector_name}\n")
            benchmark_file.write(f"tracker: {tracking_name}\n")
            benchmark_file.write("\n")

            if hasattr(evaluation_summary, "to_string"):
                benchmark_file.write(evaluation_summary.to_string(max_rows=None, max_cols=None))
            else:
                benchmark_file.write(str(evaluation_summary))
            benchmark_file.write("\n")

        return benchmark_file_path

    def compute_IoU_3D(self, predicted_detections: list, ground_truth: list):
        """
        Standalone wrapper for PyTorch3D IoU computation. PyTorch3D Requires Linux.
        """
        try:
            from pytorch3d.ops import box3d_overlap
            if predicted_detections.numel() == 0 or ground_truth.numel() == 0:
                raise ValueError("Prediction or Ground truth empty can compute IoU.")

            return box3d_overlap(predicted_detections, ground_truth)
        except ImportError as e:
            logger.error("PyTorch3D not installed, either install or try to bypass", e)

    def compute_precision_and_recall(self,
                                     predicted_detection_classes: torch.tensor,
                                     ground_truth: torch.tensor,
                                     num_classes) -> tuple[torch.tensor, torch.tensor]:
        """
        Standalone wrapper for torchmetrics recall and precision computations.
        """
        if predicted_detection_classes.shape[0] == ground_truth.shape[0]:
            precision = torchmetrics.Precision(task="multiclass", average="macro", num_classes=num_classes)
            recall = torchmetrics.Recall(task="multiclass", average="macro", num_classes=num_classes)
            return precision(
                    predicted_detection_classes, ground_truth
                    ), recall(
                            predicted_detection_classes, ground_truth
                            )
        else:
            raise ValueError("Predicted detections classes do not match with ground truth")

    def compute_average_precision(self,
                                  recall: torch.tensor,
                                  precision: torch.tensor):
        """
        Standalone Average Precision interaction.
        """
        from .metrics import AveragePrecision3D
        metric = AveragePrecision3D()
        metric.update(recall=recall,
                      precision=precision)
        return metric.compute()

    def compute_mAP_3D(self,
                       predicted_detections: torch.tensor,
                       ground_truth: torch.tensor,
                       classes,
                       box_mode="camera"):
        """
        Frame-wise standalone mAP interaction.
        """
        if predicted_detections.numel() == 0 or ground_truth.numel() == 0 or classes.numel() == 0:
            return torch.tensor([0])

        from .metrics import MeanAveragePrecision3D
        metric = MeanAveragePrecision3D(box_mode=box_mode, iou_threshold=self.iou_threshold)
        metric.update(
                preds=predicted_detections,
                target=ground_truth,
                classes=classes)
        return metric.compute()
