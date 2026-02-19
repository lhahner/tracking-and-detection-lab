import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import numpy as np
from util.datatype import Datatype

class Visualizer:
    def __init__(self, datatype):
        self.datatype = datatype
        self.fig = None
        self.axis_one = None
            
    def visualize_data(self, phase, seq, filetype, frame):
        filename = f"{frame:06d}.{filetype}"
        path = os.path.join('mot_benchmark', phase, seq, 'img1', filename)
        if self.datatype is Datatype.RGB: 
            # TODO Run the normal display approach
            im = io.imread(path)
            ax1.imshow(im)
            plt.title(seq + ' Tracked Targets')
            
        if self.datatype is Datatype.LIDAR:   
            # TODO Run the LIDAR visualizer 
            bev = self.lidar_bin_to_bev(path)
            im = self.axis_one.imshow(bev, origin='lower')
            plt.title(seq + ' Tracked Targets')
    
    def visualize_boxes(self, bounding_box, colours):
        d = np.asarray(bounding_box, dtype=np.int32)
        if d.ndim == 1:
            d = d.reshape(1, -1)
        for row in d:
            if row.size < 5:
                continue
            x1, y1, x2, y2, track_id = row[:5]
            self.axis_one.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, lw=3, ec=colours[track_id % 32, :]
                )
            )
    
    def visualize_and_draw(self):
        self.fig.canvas.flush_events()
        plt.draw()
        self.axis_one.cla()

    def setup_panel(self):
        """
        Setting the panel on which we want to display
        our read dead with the given rectangles
        """
        plt.ion()
        self.fig = plt.figure()
        self.axis_one = self.fig.add_subplot(111, aspect='equal')
    
    def lidar_bin_to_bev(
      self,
      bin_path,
      x_range=(0.0, 70.4),  # forward
      y_range=(-40.0, 40.0),  # left/right
      z_range=(-3.0, 1.0),  # up/down
      resolution=0.1):
      pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
      x, y, z, i = pts.T

      mask = (
          (x >= x_range[0]) & (x <= x_range[1]) & 
          (y >= y_range[0]) & (y <= y_range[1]) & 
          (z >= z_range[0]) & (z <= z_range[1])
      )
      x, y, z, i = x[mask], y[mask], z[mask], i[mask]

      W = int((x_range[1] - x_range[0]) / resolution)
      H = int((y_range[1] - y_range[0]) / resolution)

      x_idx = np.floor((x - x_range[0]) / resolution).astype(np.int32)
      y_idx = np.floor((y - y_range[0]) / resolution).astype(np.int32)

      x_idx = np.clip(x_idx, 0, W - 1)
      y_idx = np.clip(y_idx, 0, H - 1)

      max_height = np.full((H, W), -np.inf, dtype=np.float32)
      max_intensity = np.zeros((H, W), dtype=np.float32)
      counts = np.zeros((H, W), dtype=np.int32)

      for xi, yi, zi, ii in zip(x_idx, y_idx, z, i):
          if zi > max_height[yi, xi]:
              max_height[yi, xi] = zi
          if ii > max_intensity[yi, xi]:
              max_intensity[yi, xi] = ii
          counts[yi, xi] += 1

      max_height[max_height == -np.inf] = z_range[0]
      max_height = (max_height - z_range[0]) / (z_range[1] - z_range[0] + 1e-6)

      density = np.log(counts + 1) / np.log(64)  # cap at 64 hits per cell
      density = np.clip(density, 0, 1)

      bev = np.stack([max_height, max_intensity, density], axis=-1)
      return bev
