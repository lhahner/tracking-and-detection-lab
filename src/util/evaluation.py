from pathlib import Path
import csv
import numpy as np
import torch
from torch.autograd import grad
from entities.detection import DetectionSequence
import motmetrics as mm
import torchmetrics
import datetime
from util.logging_config import LoggingConfig
from util.coordinate_converter import CoordinateConverter
from util.file_handler import write_output
from definitions import ROOT_DIR
from nuscenes.nuscenes import NuScenes

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
        self.metrics_handler = mm.metrics.create()

    def read_mot_file(self, file_path, filter_ground_truth_by_confidence=False, allowed_class_ids=None):
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
        mot_rows = np.loadtxt(Path(file_path), delimiter=",")
        if mot_rows.ndim == 1:
            mot_rows = mot_rows.reshape(1, -1)
        detections_per_frame = {}
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
            detections_per_frame.setdefault(frame_number, []).append((object_id, bounding_box_xywh))
        return detections_per_frame

    def should_filter_ground_truth_to_pedestrians(self, sequence_name):
        """Determine whether a sequence should keep only pedestrian labels.

        Args:
            sequence_name: Sequence identifier used to infer dataset type.

        Returns:
            bool: `True` when the sequence belongs to a pedestrian benchmark.
        """
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
        """Create an accumulator for incremental MOT metric computation.

        Returns:
            motmetrics.MOTAccumulator: Empty accumulator that can be reused
            across frames with `compute_cumulative_tracking_metrics`.
        """
        return mm.MOTAccumulator(auto_id=False)

    def convert_trackers_to_mot_items(self, trackers):
        """Convert tracker output from `x1, y1, x2, y2, id` to MOT items.

        Args:
            trackers: One or more tracker rows in `x1, y1, x2, y2, id` form.

        Returns:
            list[tuple[int, list[float]]]: Object IDs with boxes in MOT `xywh`
            format.
        """
        tracker_rows = np.asarray(trackers, dtype=float)
        if tracker_rows.size == 0:
            return []
        if tracker_rows.ndim == 1:
            tracker_rows = tracker_rows.reshape(1, -1)

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
        """Update cumulative MOT metrics for one frame of tracker output.

        This method is intended for live visualization. Pass the same
        accumulator on every frame and it returns the cumulative IDF1, MOTA,
        and MOTP values after the current frame has been added.

        Args:
            mot_accumulator: Accumulator created by `create_mot_accumulator`.
            frame_number: Current frame number.
            ground_truth_by_frame: Mapping produced by `read_mot_file`.
            trackers: Tracker rows in `x1, y1, x2, y2, id` form.
            sequence_name: Name used in the metrics summary index.

        Returns:
            dict[str, float]: Cumulative `idf1`, `mota`, and `motp` values.
        """
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

    def evaluate_sequence(self, ground_truth_file_path,
                          predicted_tracking_file_path,
                          sequence_name="sequence",
                          metrics=None):
        """Evaluate one predicted tracking file against ground truth.

        Args:
            ground_truth_file_path: Path to the ground-truth MOT file.
            predicted_tracking_file_path: Path to the predicted tracking file.
            sequence_name: Name used in the resulting metrics table.
            metrics: Metrics to compute. Defaults to common MOT metrics.

        Returns:
            pandas.DataFrame: MOT metrics summary for the evaluated sequence.
        """
        if metrics is None:
            metrics = ["idf1", "mota", "motp", "precision", "recall"]

        allowed_ground_truth_class_ids = {1} if self.should_filter_ground_truth_to_pedestrians(sequence_name) else None
        ground_truth_by_frame = self.read_mot_file(
                ground_truth_file_path,
                filter_ground_truth_by_confidence=True,
                allowed_class_ids=allowed_ground_truth_class_ids,
                )
        predicted_tracks_by_frame = self.read_mot_file(predicted_tracking_file_path, filter_ground_truth_by_confidence=False)
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

    def __targets_to_tensor(self, targets):
        if isinstance(targets, torch.Tensor):
            if targets.numel() == 0:
                return torch.empty((0, 9), dtype=torch.float32)
            if targets.ndim == 1:
                return targets.reshape(1, -1).to(dtype=torch.float32)
            return targets.to(dtype=torch.float32)

        if targets is None or len(targets) == 0:
            return torch.empty((0, 9), dtype=torch.float32)

        rows = []
        for target in targets:
            if isinstance(target, torch.Tensor):
                row = target.to(dtype=torch.float32)
                if row.ndim == 1:
                    rows.append(row)
                elif row.ndim == 2:
                    rows.extend(individual_row.to(dtype=torch.float32) for individual_row in row)
                continue

            if not isinstance(target, dict):
                continue
            box = target.get("box")
            label = target.get("label")
            if box is None or label is None:
                continue
            rows.append(
                torch.tensor(
                    [*np.asarray(box, dtype=np.float32), 0.0, float(label)],
                    dtype=torch.float32,
                )
            )

        if not rows:
            return torch.empty((0, 9), dtype=torch.float32)
        return torch.stack(rows)

    def compute_IoU_3D(self, detection_sequence: DetectionSequence, box_mode="lidar"):
        """Compute per-frame 3D IoU matrices for a detection sequence."""
        try:
            from pytorch3d.ops import box3d_overlap
        except ImportError as exc:
            raise ImportError("PyTorch3D is required for 3D IoU computation.") from exc

        coordinate_converter = CoordinateConverter()
        iou_per_frame = []
        for frame_detection in detection_sequence.frames:
            prediction_rows = []
            for det in frame_detection.dets:
                prediction_rows.append(
                    torch.tensor(
                        [*det.box.detach().cpu().tolist(), float(det.score), float(det.label)],
                        dtype=torch.float32,
                    )
                )
            predicted_detections = (
                torch.stack(prediction_rows)
                if prediction_rows
                else torch.empty((0, 9), dtype=torch.float32)
            )
            ground_truth = self.__targets_to_tensor(frame_detection.targets)

            if predicted_detections.numel() == 0 or ground_truth.numel() == 0:
                iou_matrix = torch.empty(
                    (predicted_detections.shape[0], ground_truth.shape[0]),
                    dtype=torch.float32,
                )
            else:
                prediction_corner_boxes = coordinate_converter.boxes_3d_to_corners(
                    predicted_detections[:, :7],
                    box_mode,
                )
                ground_truth_corner_boxes = coordinate_converter.boxes_3d_to_corners(
                    ground_truth[:, :7],
                    box_mode,
                )
                _, iou_matrix = box3d_overlap(
                    prediction_corner_boxes,
                    ground_truth_corner_boxes,
                )

            iou_per_frame.append(
                {
                    "sample_id": frame_detection.frame,
                    "iou_matrix": iou_matrix.cpu(),
                }
            )
        return iou_per_frame

    def export_nuscenes_kitti3d_iou_analysis(
        self,
        detection_sequence: DetectionSequence,
        output_file_path,
        serializer=None,
        data_root=None,
        version="",
        serialize_predictions=False,
        box_mode="lidar",
    ):
        """Write a minimal per-prediction IoU analysis CSV for nuScenes-on-KITTI runs."""
        iou_results = self.compute_IoU_3D(detection_sequence, box_mode=box_mode)
        fieldnames = ["sample_id", "file_name", "IoU", "predicted_class", "ground_truth_class"]
        rows = []

        for frame_detection, frame_iou in zip(detection_sequence.frames, iou_results):
            iou_matrix = frame_iou["iou_matrix"]
            gt_labels = frame_detection.targets
            for prediction_index, det in enumerate(frame_detection.dets):
                predicted_class = det.label
                if iou_matrix.numel() == 0 or iou_matrix.shape[1] == 0:
                    rows.append(
                        {
                            "sample_id": str(frame_detection.frame),
                            "file_name": "",
                            "IoU": 0.0,
                            "predicted_class": predicted_class,
                            "ground_truth_class": "",
                        }
                    )
                    continue

                best_iou, best_gt_idx = torch.max(iou_matrix[prediction_index], dim=0)
                best_gt_idx_value = int(best_gt_idx.item())
                rows.append(
                    {
                        "sample_id": str(frame_detection.frame),
                        "file_name": str(self.__get_file_name_by_token_id(token_id=str(frame_detection.frame),
                                                                          version=version,
                                                                          data_root=data_root)),
                        "IoU": round(float(best_iou.item()), 6),
                        "predicted_class": predicted_class.item() if torch.is_tensor(predicted_class) else predicted_class,
                        "ground_truth_class": gt_labels[best_gt_idx_value]['label'] if best_gt_idx_value < len(gt_labels) else "",
                    }
                )
        output_path = Path(output_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        return rows

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
        from util.metrics.average_precision_3D import AveragePrecision3D
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
        from util.metrics.mean_average_precision_3D import MeanAveragePrecision3D

        metric = MeanAveragePrecision3D(box_mode=box_mode, iou_threshold=self.iou_threshold)
        metric.update(
                preds=predicted_detections,
                target=ground_truth,
                classes=classes)
        return metric.compute()

    def __get_file_name_by_token_id(self,
                                    token_id, 
                                    data_root, 
                                    version="v1.0-trainval", 
                                    verbose=True):
        nusc = NuScenes(version=version,
                        dataroot=data_root,
                        verbose=verbose)
        sample = nusc.get("sample", token_id)
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_sd = nusc.get("sample_data", lidar_token)
        return lidar_sd["filename"]
