import os
import time
from pathlib import Path

import matplotlib
import importlib.util
import numpy as np
from skimage import io

from util.coordinate_converter import CoordinateConverter
from util.evaluation import Evaluation
from util.settings_loader import SettingsLoader
from util.visualizer import Visualizer

if importlib.util.find_spec("tkinter") is not None:
    matplotlib.use("Agg")

from tracker.DeepSORT.deepSort import DeepSort as DeepSortTracker
from tracker.SORT.sort import Sort
from torch.utils.data import Dataset
from entities.detection import convert_to_tensor, convert_classes_to_tensor

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
    
    def load(self):
        dataset_name = self.settings.runtime.dataset.lower()
        if dataset_name == "kitti3d":
            self.dataset = Kitti3D(
                    data_root=self.settings.paths.dataset_path) 
        return self.dataset

    def predict(self, detector_name, dataset_path, detection_path, model_path):
        """Instantiate and run the configured detector implementation.

        Args:
            detector_name: Short name of the detector to execute.
            dataset_path: Directory that contains the input frames.
            detection_path: Directory where `det.txt` should be written.
            model_path: Path to the detector model weights if required.
        """
        detector = None
        if detector_name == "frcnn":
            detector = FasterRCNNDetector(
                input_path=dataset_path, output_path=detection_path, threshold=0.9
            )
        if detector_name == "detr":
            detector = DetrHuggingFaceDetector(
                input_path=dataset_path,
                output_path=detection_path,
            )
        if detector_name == "yolo":
            detector = YoloDetector(
                input_path=dataset_path,
                output_path=detection_path,
                model_path=model_path,
            )
        if detector_name == "detectron2":
            detector = MaskFasterRCNNDetector(
                input_path=dataset_path, output_path=detection_path, threshold=0.9
            )
        if detector_name == "pointrcnn":
            detector = PointRCNNmmDetections3D(dataset=self.dataset,
                                               config_file=self.settings.paths.config_file,
                                               classes=self.settings.dataset.classes,
                                               settings=self.settings)
        return detector.detect()
    
    def evaluate_detection(self, detections, classes):
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
            results.append(
                {
                    "frame": detection_frame.frame,
                    "mAP":
                       Evaluation().compute_mAP_3D(
                           predicted_detections=convert_to_tensor(detection_frame.dets),
                           ground_truth=self.dataset.convert_ground_truth(detection_frame.targets),
                           classes=class_tensor,
                           )
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
