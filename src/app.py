from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
import os

from tracker.SORT.sort import Sort
from tracker.SORT.kalmanBoxTracker import KalmanBoxTracker
from detector.yolo.yolo import YoloDetector
from detector.pointpillars.pointpillars import PointpillarsDetector

class Application:
    def __init__(self, total_time, total_frames, colours):
        self.seed = np.random.seed(0)
        self.total_time = total_time
        self.total_frames = total_frames
        self.colours = colours
        self.implemented_detectors = self.read_implemented_detectors() # TODO read from folders
    
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
        if not os.path.exists('data'):
            raise ValueError('No data Folder found.')
        det_folders = os.listdir('data')
        return det_folders
    
    def run_detector_by_argument(self, arg_detector, arg_dataset):
        """
        Depending on the argument read by the argument
        parse a different detection system is choosen
        and loaded. A detection system should be choosen
        based on the data used.
        """
        if (arg_detector == 'frcnn'):
            return None
        if (arg_detector == 'yolo'):
            detector = YoloDetector(
                os.path.join('mot_benchmark', 'train', arg_dataset, 'img1'),
                os.path.join('data', arg_detector, arg_dataset, 'det'),
                os.path.join('detector', arg_detector, 'model', 'yolo11n.pt') # maybe check that the .pt file is read
            )
            detector.detect()
        if (arg_detector == 'pointpillars'):
            detector = PointpillarsDetector(
                os.path.join('mot_benchmark', 'train', arg_dataset, 'img1/'), # TODO
                os.path.join('data', arg_detector, arg_dataset, 'det/'),
                os.path.join('detector', arg_detector, 'model', 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py'),
                os.path.join('detector', arg_detector, 'model', 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth')
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
 # all train
  app = Application(0.0, 0, np.random.rand(32, 3))
  args = app.parse_args()
  display = args.display
  phase = args.phase
  detector = args.detector
  dataset = args.dataset
  path_detection = []
  
  if (detector and not dataset):
      raise ValueError('Please specify a dataset to use for detection')
 
  if (dataset):
      if dataset not in os.listdir('mot_benchmark/train/'):
          raise ValueError(f'dataset {dataset} is not in input folder, please create a folder named {dataset}')
 
  if (detector):
     if detector not in app.get_implemented_detectors():
         print(f"detector {detector} not implemented detector {app.get_implemented_detectors()}")
     else:
        for implemented_detector in app.get_implemented_detectors():
            app.run_detector_by_argument(implemented_detector, dataset)
        
  if(display): 
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
 
  pattern = os.path.join('data', detector, dataset, 'det', 'det.txt') # path/filename matching
  
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold)
     
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',') 
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
        
      print("Processing %s."%(seq))
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        app.total_frames += 1

        if(display):
          fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        app.total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=app.colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (app.total_time, app.total_frames, app.total_frames / app.total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")