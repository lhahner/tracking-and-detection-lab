from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

from util.datatype import Datatype
from util.coordinate_converter import CoordinateConverter
from util.visualizer import Visualizer
from util.settings_loader import SettingsLoader
from util.evaluation import Evaluation

import glob
import time
import argparse
import os
from pathlib import Path

# Tracking systems
from tracker.SORT.sort import Sort
from tracker.SORT.kalmanBoxTracker import KalmanBoxTracker
from tracker.DeepSORT.deepSort import DeepSort as DeepSortTracker
from tracker.DeepSORT.deepSort import DeepSort as DeepSortTracker

# Detection systems
from detector.yolo.yolo_ultralytics import YoloUltralyticsDetector
from detector.detr.detr_huggingface import DetrHuggingFaceDetector
from detector.maskfrcnn.maskfrcnn_detectron2 import MaskFasterRCNNDetectron2Detector
from detector.frcnn.frcnn_detectron2 import FasterRCNNDetectron2Detector
from detector.pointnet.pointnet_trainer import PointnetTrainer

from datasets.kitti3D import Kitti3D
from inference import inference

class Application:
    """Coordinate detector execution, tracking, visualization, and evaluation."""

    def __init__(self, total_time, total_frames, colours):
        """Initialize the application state used during detection and tracking.

        Args:
            total_time: Accumulated processing time across all frames.
            total_frames: Number of processed frames.
            colours: Color table used to render tracked objects.
        """
        self.seed = np.random.seed(0)
        self.total_time = total_time
        self.total_frames = total_frames
        self.colours = colours
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.implemented_detectors = self.read_implemented_detectors() # TODO read from folders
        self.datatype = Datatype.RGB # Changes depending on used detector
    
    def get_implemented_detectors(self):
        """Return the detector package names available in the project.

        Returns:
            list[str]: Names of implemented detector directories.
        """
        return self.implemented_detectors
    
    def read_implemented_detectors(self):
        """Discover implemented detectors from the local detector package.

        Returns:
            list[str]: Detector directory names found under `src/detector`.

        Raises:
            ValueError: If the detector directory does not exist.
        """
        detector_dir = os.path.join(self.project_root, 'detector')
        if not os.path.isdir(detector_dir):
            raise ValueError(f'No detector folder found at {detector_dir}.')
        det_folders = [
            folder for folder in os.listdir(detector_dir)
            if os.path.isdir(os.path.join(detector_dir, folder)) and not folder.startswith(('.', '_'))
        ]
        return det_folders
    
    def run_detector_by_argument(self, detector_name, dataset_path, detection_path, model_path):
        """Instantiate and run the configured detector implementation.

        Args:
            detector_name: Short name of the detector to execute.
            dataset_path: Directory that contains the input frames.
            detection_path: Directory where `det.txt` should be written.
            model_path: Path to the detector model weights if required.
        """
        detector = None
        if (detector_name == 'frcnn'):
            detector = FasterRCNNDetector(
                input_path=dataset_path,
                output_path=detection_path,
                threshold=0.9
            )
        if (detector_name == 'detr'):
            detector = DetrDetector(
                input_path=dataset_path,
                output_path=detection_path,
            )
        if (detector_name == 'yolo'):
            detector = YoloDetector(
                input_path=dataset_path, 
                output_path=detection_path, 
                model_path=model_path
            )
        if (detector_name == 'detectron2'):
            detector = MaskFasterRCNNDetector(
                input_path=dataset_path, 
                output_path=detection_path, 
                threshold=0.9    
            ) 
        detector.detect()
    
    def parse_args(self):
        """Parse command-line arguments for the tracking benchmark.

        Returns:
            argparse.Namespace: Parsed runtime arguments.
        """
        parser = argparse.ArgumentParser(description='SORT Benchmark')
        parser.add_argument('--display', dest='display', 
                            help='Display online tracker output (slow) [False]',
                            action='store_true')
        parser.add_argument("--seq_path", 
                            help="Path to detections.", 
                            type=str, default='data')
        parser.add_argument("--phase", 
                            help="Subdirectory in seq_path.", 
                            type=str, default='train')
        parser.add_argument("--max_age", 
                            help="Maximum number of frames to keep alive a track without associated detections.", 
                            type=int, default=1)
        parser.add_argument("--detector", 
                            type=str, default='frcnn')
        parser.add_argument("--dataset",
                            help="Specify which dataset to use, the name should be equal to where the dataset is present",
                            type=str, default='*')
        parser.add_argument("--min_hits", 
                            help="Minimum number of associated detections before track is initialised.", 
                            type=int, default=3)
        parser.add_argument("--iou_threshold", 
                            help="Minimum IOU for match.", 
                            type=float, default=0.3)
        args = parser.parse_args()
        return args
    
if __name__ == "__main__":
    settings = SettingsLoader.load("settings.yaml")
    if settings.runtime.mode == "inference":
        inference(Application(0.0, 0, np.random.rand(32, 3)), settings)
        
    elif settings.runtime.mode == "train":
      train_dataset = Kitti3D(
          data_root=settings.paths.dataset_path,
          split="train",
          mode="object",
          num_points=1024,
          include_background=True
      )

      val_dataset = Kitti3D(
          data_root=settings.paths.dataset_path,
          split="val",
          mode="object",
          num_points=1024,
          include_background=True,
      )

      trainer = PointnetTrainer(
          train_dataset=train_dataset,
          val_dataset=val_dataset,
          output_checkpoint=settings.paths.models_root,
          epochs=20,
          batch_size=16,
          num_points=1024,
          learning_rate=1e-3,
          use_intensity=True,
      )

      trainer.train()

  
