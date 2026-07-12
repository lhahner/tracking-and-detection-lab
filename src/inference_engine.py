import os
from pathlib import Path
import glob
import matplotlib
import importlib.util
import torch
import numpy as np
from skimage import io

from geometry.coordinate_converter import CoordinateConverter
from datasets.kitti_calib import extend_to_4x4
from evaluation import Evaluation
from visualization.visualizer import Visualizer

if importlib.util.find_spec("tkinter") is not None:
    matplotlib.use("Agg")

from tracker.DeepSORT.deepSort import DeepSort as DeepSortTracker
from tracker.SORT.sort import Sort
from entities.detection import convert_to_tensor, convert_classes_to_tensor

# Detection systems
from detector.detector_registry import MODELS

# Datasets
from datasets.nuScenes import NuScenesDataset
from datasets.kitti3D import Kitti3D

class InferenceEngine:
    def __init__(self, settings):
        self.settings = settings
        self.visualizer = Visualizer(
            self.settings.runtime.datatype
        )
        self.evaluation_runner = Evaluation(
            iou_threshold=settings.benchmark.iou_threshold
        )
        self.detection_path = os.path.join(settings.paths.detection_path, "det.txt")
        self.tracker = Sort(
            max_age=self.settings.tracker.max_age,
            min_hits=self.settings.tracker.min_hits,
            iou_threshold=self.settings.tracker.iou_threshold,
        )
        self.dataset = None

    def load(self, split, max_samples, labels=None):
        dataset_name = self.settings.runtime.dataset.lower()
        if dataset_name == "kitti3d":
            self.dataset = Kitti3D(data_root=self.settings.paths.dataset_path,
                                   split=split,
                                   max_samples=max_samples,
                                   labels=labels)
        elif dataset_name in {"nuscenes_openpcdet", "nuscenes-mini_openpcdet"}:
            from datasets.nuScenes_openpcdet_adapter import NuScenesOpenPCDetAdapter

            version = "v1.0-mini" if dataset_name == "nuscenes-mini_openpcdet" else "v1.0-trainval"
            split = "mini_val" if dataset_name == "nuscenes-mini_openpcdet" else "val"
            nuScenes = NuScenesDataset(data_root=self.settings.paths.dataset_path,
                                       version=version,
                                       split=split)
            self.dataset = NuScenesOpenPCDetAdapter(nuScenes=nuScenes,
                                                    root_path=self.settings.paths.dataset_path,
                                                    max_samples=max_samples,
                                                    class_names=["car",
                                                                 "truck",
                                                                 "construction_vehicle",
                                                                 "bus", 
                                                                 "trailer", 
                                                                 "barrier", 
                                                                 "motorcycle", 
                                                                 "bicycle", 
                                                                 "pedestrian", 
                                                                 "traffic_cone"])

        elif dataset_name in {"nuscenes", "nuscenes-mini"}:
            version = "v1.0-mini" if dataset_name == "nuscenes-mini" else "v1.0-trainval"
            split = "mini_val" if dataset_name == "nuscenes-mini" else "val"
            self.dataset = NuScenesDataset(data_root=self.settings.paths.dataset_path,
                                           version=version,
                                           split=split
                                           )
            if max_samples is not None:
                self.dataset.sample_records = self.dataset.sample_records[:max_samples]
        return self.dataset

    def predict(self, detector_name, dataset_path, detection_path, model_path):
        classes = getattr(self.dataset, "classes", None)
        if classes is None:
            classes = getattr(self.dataset, "labels", None)
        if classes is None:
            classes = getattr(self.settings.dataset, "classes", None)
        if classes is None:
            classes = getattr(self.settings.benchmark, "class_filter", None)
        detector = MODELS.create(detector_name,
                                 dataset=self.dataset,
                                 config_file=self.settings.paths.config_file,
                                 classes=classes,
                                 settings=self.settings,
                                 checkpoint_file=model_path)
        return detector.detect()

    def evaluate_detection(self,
                           detections,
                           classes,
                           box_mode="lidar"):
        """
        Run evaluation for detections.

        Args:
            detections: A tensor of detections

        Returns:
            List of mAP evaluations
        """
        results = []
        class_tensor = convert_classes_to_tensor(classes)
        for detection_frame in detections.frames:
            ground_truth = self.dataset.convert_ground_truth(detection_frame.targets, detection_frame.frame)
            detection_tensor = convert_to_tensor(detection_frame.dets)
            mAP = Evaluation(iou_threshold=self.settings.benchmark.iou_threshold).compute_mAP_3D(
                           predicted_detections=detection_tensor,
                           ground_truth=ground_truth,
                           classes=class_tensor,
                           box_mode=box_mode,
                           )
            results.append(
                {
                    "frame": detection_frame.frame,
                    "mAP": mAP
                }
            )
        return results

    def update_tracker(self):
        if self.settings.runtime.display:
            self.visualizer.setup_panel()

        for unparsed_detection in glob.glob(self.detection_path):
            self.__init_tracker()

            parsed_detection = np.loadtxt(unparsed_detection, delimiter=",")
            seq = os.path.basename(os.path.dirname(os.path.dirname(unparsed_detection)))

            sequence_output_path = Path(self.settings.paths.output_root) / f"{seq}.txt"
            sequence_output_path.parent.mkdir(parents=True, exist_ok=True)
            sequence_ground_truth_path = Path(self.settings.paths.ground_truth_path)
            ground_truth_by_frame = {}
            mot_accumulator = None
            metrics_history = {"idf1": [], "motp": [], "mota": []}

            if should_visualize_metrics and sequence_ground_truth_path.exists():
                ground_truth_by_frame = evaluation_runner.read_mot_file(
                    sequence_ground_truth_path,
                    filter_ground_truth_by_confidence=True,
                    allowed_class_ids=self.settings.benchmark.class_filter,
                )
                mot_accumulator = evaluation_runner.create_mot_accumulator()
            elif should_visualize_metrics:
                logger.warn(
                    f"Ground truth file not found for live metrics in {seq}: "
                    f"{sequence_ground_truth_path}"
                )

            with open(sequence_output_path, "w") as out_file:
                logger.info("Processing %s." % (seq))
                converter = CoordinateConverter()
                for frame in range(int(parsed_detections[:, 0].max())):
                    frame += 1
                    dets = converter.convert2DDetectionToBox(parsed_detections, frame)
                    frame_path = os.path.join(
                        settings.paths.dataset_path,
                        f"{frame:06d}.{settings.runtime.datatype}",
                    )
                    frame_img = io.imread(frame_path)
                    trackers = mot_tracker.update(dets, frame=frame_img)
                    for d in trackers:
                        print(
                            "%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1"
                            % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                            file=out_file,
                        )

                    if self.settings.runtime.display:
                        self.__update_visualization_frame(
                            self,
                            mot_accumulator=mot_accumulator,
                            frame=frame,
                            ground_truth_by_frame=ground_truth_by_frame,
                            trackers=trackers,
                            seq=seq,
                            cumulative_metrics=cumulative_metrics,
                            metrics_history=metrics_history,
                        )

            if sequence_ground_truth_path.exists():
                self.__save_evaluation_summary(
                    sequence_ground_truth_path=sequence_ground_truth_path,
                    sequence_output_path=sequence_output_path,
                    seq=seq,
                )
            else:
                logger.log(
                    f"Ground truth file not found for {seq}: "
                    f"{sequence_ground_truth_path}"
                )

    def __init_tracker(self):
        if self.settings.runtime.tracker.lower() == "deepsort":
            self.mot_tracker = DeepSortTracker(
                max_age=settings.tracker.max_age,
                min_hits=settings.tracker.min_hits,
                iou_threshold=settings.tracker.iou_threshold,
                bgr=False,  # skimage.io.imread returns RGB
            )

    def __save_evalulation_summary(self, sequence_ground_truth_path, sequence_output_path, seq):
        evaluation_summary = evaluation_runner.evaluate_sequence(
            ground_truth_file_path=sequence_ground_truth_path,
            predicted_tracking_file_path=sequence_output_path,
            sequence_name=seq,
        )
        print(evaluation_summary)
        benchmark_file_path = evaluation_runner.presist_evaluation(
            evaluation_summary=evaluation_summary,
            dataset=seq,
            detector_name=self.settings.runtime.detector,
            tracking_name=self.settings.runtime.tracker,
        )
        logger.info(f"Saved benchmark summary to {benchmark_file_path}")

    def __update_visualization_frame(
        self,
        mot_accumulator,
        frame,
        ground_truth_by_frame,
        trackers,
        seq,
        cumulative_metrics,
        metrics_history,
    ):
        if mot_accumulator is not None:
            cumulative_metrics = (
                self.evaluation_runner.compute_cumulative_tracking_metrics(
                    mot_accumulator=mot_accumulator,
                    frame_number=frame,
                    ground_truth_by_frame=ground_truth_by_frame,
                    trackers=trackers,
                    sequence_name=seq,
                )
            )
            for metric_name, metric_value in cumulative_metrics.items():
                metrics_history[metric_name].append(metric_value)

        visualizer.visualize_tracking_frame(
            dataset_path=self.settings.paths.dataset_path,
            frame=frame,
            filetype=self.settings.runtime.datatype,
            trackers=trackers,
            colours=self.settings.visualizer.colours, 
            metrics_history=metrics_history,
        )
