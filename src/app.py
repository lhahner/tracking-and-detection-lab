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

from tracker.SORT.sort import Sort
from tracker.SORT.kalmanBoxTracker import KalmanBoxTracker

class Application:
    def __init__(self, total_time, total_frames, colours):
        self.seed = np.random.seed(0)
        self.total_time = total_time
        self.total_frames = total_frames
        self.colours = colours

    def parse_args(self):
        """Parse input arguments."""
        parser = argparse.ArgumentParser(description='SORT demo')
        parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
        parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
        parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
        parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
        parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
        parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
        args = parser.parse_args()
        return args
 
if __name__ == "__main__":
 # all train
  app = Application(0.0, 
                     0, 
                     np.random.rand(32, 3)
                     )
  args = app.parse_args()
  display = args.display
  phase = args.phase
    
  if(display): # Initially used to demonstrate the performance on the MOT15 Data set.
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt') # path/filename matching
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) # create instance of the SORT tracker
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',') # TODO what do the det.txt files provide?
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