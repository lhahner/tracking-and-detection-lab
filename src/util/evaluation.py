from pathlib import Path
import numpy as np
import motmetrics as mm
import datetime

class Evaluation:
    """Compute and persist MOT-style tracking benchmark metrics."""

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
           
            frame_number = int(mot_row[0]); object_id = int(mot_row[1])
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
        
        mot_accumulator = mm.MOTAccumulator(auto_id=False) # MotMetric setup
        maximum_iou_distance = 1.0 - self.iou_threshold # default to 0.5
        
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
