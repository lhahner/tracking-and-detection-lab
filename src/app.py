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
from detector.yolo.yolo import YoloDetector
from detector.detr.detr import DetrDetector
from detector.maskfrcnn.maskfrcnn import MaskFasterRCNNDetector
from detector.frcnn.frcnn import FasterRCNNDetector

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
                            help="Specify which detection system you want to use, the directory which contains the detection inside the data folder should have the same name.", 
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
  app = Application(0.0, 0, np.random.rand(32, 3))
  
  args = app.parse_args()
  phase = args.phase
  path_detection = []
  
  settings = SettingsLoader.load("settings.yaml")
  evaluation_runner = Evaluation(iou_threshold=0.5)
  
  app.run_detector_by_argument(
      settings.runtime.detector, 
      dataset_path=settings.paths.mot_root, 
      detection_path=settings.paths.detection_path, 
      model_path=settings.paths.models_root
    )
        
  visualizer = Visualizer(app.datatype)
  if(settings.runtime.display): 
    visualizer.setup_panel()

  pattern = os.path.join(settings.paths.detection_path, "det.txt")
  for seq_dets_fn in glob.glob(pattern):
    if settings.runtime.tracker.lower() == "deepsort":
      mot_tracker = DeepSortTracker(
          max_age=args.max_age,
          min_hits=args.min_hits,
          iou_threshold=args.iou_threshold,
          bgr=False,  # skimage.io.imread returns RGB
      )
    else:
      mot_tracker = Sort(max_age=args.max_age, 
                         min_hits=args.min_hits,
                         iou_threshold=args.iou_threshold)
     
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',') 
    seq = os.path.basename(os.path.dirname(os.path.dirname(seq_dets_fn)))
    
    sequence_output_path = Path(settings.paths.output_root) / f"{seq}.txt"
    sequence_output_path.parent.mkdir(parents=True, exist_ok=True)
    sequence_ground_truth_path = Path(settings.paths.ground_truth_path)

    with open(sequence_output_path,'w') as out_file:
        
      print("Processing %s."%(seq))
      converter = CoordinateConverter()
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1
        dets = converter.convert2DDetectionToBox(seq_dets, frame)
        app.total_frames += 1

        if(settings.runtime.display):
           visualizer.visualize_data(
               dir_path=settings.paths.mot_root, 
               filetype=settings.runtime.datatype, 
               frame=frame)

        start_time = time.time()
        if settings.runtime.tracker.lower() == "deepsort":
          frame_path = os.path.join(
              settings.paths.mot_root,
              f"{frame:06d}.{settings.runtime.datatype}"
          )
          frame_img = io.imread(frame_path)
          trackers = mot_tracker.update(dets, frame=frame_img)
        else:
          trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        app.total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(settings.runtime.display):
              visualizer.visualize_boxes(d, app.colours)
        
        if(settings.runtime.display):
          visualizer.visualize_and_draw()
    
    if settings.runtime.benchmark:
      if sequence_ground_truth_path.exists():
        evaluation_summary = evaluation_runner.evaluate_sequence(
            ground_truth_file_path=sequence_ground_truth_path,
            predicted_tracking_file_path=sequence_output_path,
            sequence_name=seq)
        print(evaluation_summary)
        benchmark_file_path = evaluation_runner.presist_evaluation(
            evaluation_summary=evaluation_summary,
            dataset=seq,
            detector_name=settings.runtime.detector,
            tracking_name=settings.runtime.tracker,
        )
        print(f"Saved benchmark summary to {benchmark_file_path}")
      else:
        print(f"Ground truth file not found for {seq}: {sequence_ground_truth_path}")
        
  if app.total_time > 0:
    fps = app.total_frames / app.total_time
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (app.total_time, app.total_frames, fps))
  
  else:
    print("Total Tracking took: %.3f seconds for %d frames (FPS unavailable: no processing time recorded)" % (app.total_time, app.total_frames))
  
