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
import glob
import time
import argparse
import os

from tracker.SORT.sort import Sort
from tracker.SORT.kalmanBoxTracker import KalmanBoxTracker
from detector.yolo.yolo import YoloDetector

class Application:
    def __init__(self, total_time, total_frames, colours):
        self.seed = np.random.seed(0)
        self.total_time = total_time
        self.total_frames = total_frames
        self.colours = colours
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.implemented_detectors = self.read_implemented_detectors() # TODO read from folders
        self.datatype = Datatype.RGB # Changes depending on used detector
    
    def get_implemented_detectors(self):
        """
        basic getter function to retrieve the 
        global list of all implemented detection
        systems.
        """
        return self.implemented_detectors
    
    def read_implemented_detectors(self):
        """
        All implemented detection systems 
        are located at the detector package
        thus the following build every detection
        system known for the current project.
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
        """
        Depending on the argument read by the argument
        parse a different detection system is choosen
        and loaded. A detection system should be choosen
        based on the data used.
        """
        detector = None
        if (detector_name == 'frcnn'):
            return None
        if (detector_name == 'yolo'):
            detector = YoloDetector(
                input_path=dataset_path, 
                output_path=detection_path, 
                model_path=model_path
            )
        detector.detect()
    
    def parse_args(self):
        """
        Core argument parser that show all options
        available for the program.
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
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold)
     
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',') 
    seq = os.path.basename(os.path.dirname(os.path.dirname(seq_dets_fn)))
    
    with open(os.path.join('src/output', '%s.txt'%(seq)),'w') as out_file:
        
      print("Processing %s."%(seq))
      converter = CoordinateConverter()
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1
        # TODO in dets.txt confidence is missing and the transformation is not that simple
        if app.datatype is Datatype.LIDAR:
            dets = converter.convert3DLIDARDetectionToBEV(seq_dets, frame)
        
        elif app.datatype is Datatype.RGB:
            dets = converter.convert2DDetectionToBox(seq_dets, frame)
        
        else:
            raise ValueError("Detection datatype not detected.")    
        app.total_frames += 1

        if(settings.runtime.display):
           visualizer.visualize_data(
               dir_path=settings.paths.mot_root, 
               filetype=settings.runtime.datatype, 
               frame=frame)

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        app.total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(settings.runtime.display):
              visualizer.visualize_boxes(d, app.colours)

        if(settings.runtime.display):
          visualizer.visualize_and_draw()
          print("Visualized Box")

  if app.total_time > 0:
    fps = app.total_frames / app.total_time
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (app.total_time, app.total_frames, fps))
  else:
    print("Total Tracking took: %.3f seconds for %d frames (FPS unavailable: no processing time recorded)" % (app.total_time, app.total_frames))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
