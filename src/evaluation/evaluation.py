from __future__ import annotations
from pathlib import Path
import csv
import datetime
import numpy
import torch
import motmetrics as mm
import torchmetrics

try:
    import numpy as np
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda values, dtype=float: np.asarray(values, dtype=dtype)
except ImportError:  # pragma: no cover
    np = None

from config.logging_config import LoggingConfig

logging_config = LoggingConfig()
logger = logging_config.get_logger(__name__)


class Evaluation:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.metrics_handler = mm.metrics.create()

    def read_mot_file(self,
                      file_path,
                      filter_ground_truth_by_confidence=False,
                      allowed_class_ids=None):
        mot_rows = np.loadtxt(Path(file_path), delimiter=",")
        if mot_rows.ndim == 1:
            mot_rows = mot_rows.reshape(1, -1)
        detections_per_frame = {}
        if np is not None:
            mot_rows = np.loadtxt(Path(file_path), delimiter=",")
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

    def evaluate_simpletrack_nuscenes_sample_tokens(
            self,
            result_path,
            dataroot,
            sample_tokens,
            version="v1.0-mini",
            score_threshold=None,
        ):
        from nuscenes import NuScenes
        from nuscenes.eval.common.loaders import (
                load_gt_of_sample_tokens,
                load_prediction_of_sample_tokens,
                )
        from nuscenes.eval.tracking.data_classes import TrackingBox
        try:
            from nuscenes.eval.common.config import config_factory
        except ImportError:
            from nuscenes.eval.tracking.config import config_factory

        cfg = config_factory("tracking_nips_2019")
        nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
        sample_tokens = [str(sample_token) for sample_token in sample_tokens]
        pred_boxes, _ = load_prediction_of_sample_tokens(
                str(result_path),
                cfg.max_boxes_per_sample,
                TrackingBox,
                sample_tokens=sample_tokens,
                verbose=False,
                )
        gt_boxes = load_gt_of_sample_tokens(
                nusc,
                sample_tokens,
                TrackingBox,
                verbose=False,
                )

        if score_threshold is None:
            thresholds = {0.0}
            for sample_token in sample_tokens:
                thresholds.update(
                        box.tracking_score
                        for box in pred_boxes.boxes.get(sample_token, [])
                        )
            thresholds = sorted(thresholds, reverse=True)
        else:
            thresholds = [float(score_threshold)]

        metrics = [
                "mota",
                "motp",
                "recall",
                "num_matches",
                "num_false_positives",
                "num_misses",
                ]
        best_result = None
        for threshold in thresholds:
            accumulator = mm.MOTAccumulator(auto_id=False)
            ground_truth_id_map = {}
            prediction_id_map = {}
            frame_id = 0
            for sample_token in sample_tokens:
                for class_name in cfg.class_names:
                    ground_truth = [
                            box for box in gt_boxes.boxes.get(sample_token, [])
                            if box.tracking_name == class_name
                            ]
                    predictions = [
                            box for box in pred_boxes.boxes.get(sample_token, [])
                            if box.tracking_name == class_name
                            and box.tracking_score >= threshold
                            ]

                    ground_truth_ids = [
                            self.__numeric_tracking_id(box.tracking_id, ground_truth_id_map)
                            for box in ground_truth
                            ]
                    prediction_ids = [
                            self.__numeric_tracking_id(box.tracking_id, prediction_id_map)
                            for box in predictions
                            ]
                    if ground_truth and predictions:
                        ground_truth_centers = np.asarray(
                                [box.translation[:2] for box in ground_truth],
                                dtype=float,
                                )
                        prediction_centers = np.asarray(
                                [box.translation[:2] for box in predictions],
                                dtype=float,
                                )
                        distance_matrix = np.linalg.norm(
                                ground_truth_centers[:, None, :] - prediction_centers[None, :, :],
                                axis=2,
                                )
                        distance_matrix[distance_matrix >= cfg.dist_th_tp] = np.nan
                    else:
                        distance_matrix = np.full(
                                (len(ground_truth), len(predictions)),
                                np.nan,
                                )

                    accumulator.update(
                            ground_truth_ids,
                            prediction_ids,
                            distance_matrix,
                            frameid=frame_id,
                            )
                    frame_id += 1

            summary = self.metrics_handler.compute(
                    accumulator,
                    metrics=metrics,
                    name="sample_subset",
                    )
            metric_row = summary.loc["sample_subset"]
            result = {metric: metric_row[metric] for metric in metrics}
            result["score_threshold"] = threshold
            result["sample_count"] = len(sample_tokens)
            if best_result is None or result["mota"] > best_result["mota"]:
                best_result = result

        return best_result

    def __numeric_tracking_id(self, tracking_id, tracking_id_map):
        if tracking_id not in tracking_id_map:
            tracking_id_map[tracking_id] = len(tracking_id_map) + 1
        return tracking_id_map[tracking_id]

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

        mot_accumulator = mm.MOTAccumulator(auto_id=False)
        maximum_iou_distance = 1.0 - self.iou_threshold
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
        if predicted_detections.numel() == 0 or ground_truth.numel() == 0 or classes.numel() == 0:
            return torch.tensor([0])

        from .metrics import MeanAveragePrecision3D
        metric = MeanAveragePrecision3D(box_mode=box_mode, iou_threshold=self.iou_threshold)
        metric.update(
                preds=predicted_detections,
                target=ground_truth,
                classes=classes)
        return metric.compute()
