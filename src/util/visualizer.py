import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import numpy as np
from util.datatype import Datatype

class Visualizer:
    """Display image frames, tracking boxes, and LiDAR bird's-eye views."""

    def __init__(self, datatype):
        """Create a visualization helper for RGB or LiDAR data.

        Args:
            datatype: Data modality that determines the visualization mode.
        """
        self.datatype = datatype
        self.fig = None
        self.axis_one = None
            
    def visualize_data(self, dir_path, frame, filetype):
        """Load and display a frame image on the active matplotlib axis.

        Args:
            dir_path: Directory containing frame images.
            frame: Frame number to visualize.
            filetype: File extension of the frame images.
        """
        filename = f"{frame:06d}.{filetype}"
        path = os.path.join(dir_path, filename)
        im = io.imread(path)
        self.axis_one.imshow(im)
        plt.title('Tracked Targets')
            
    def visualize_boxes(self, bounding_box, colours):
        """Draw tracked bounding boxes on the active matplotlib axis.

        Args:
            bounding_box: One or more tracked boxes in `x1, y1, x2, y2, id` form.
            colours: Color lookup table indexed by track ID.
        """
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
        """Refresh the interactive plot and clear the current axis."""
        self.fig.canvas.flush_events()
        plt.draw()
        self.axis_one.cla()

    def setup_panel(self):
        """Create the interactive matplotlib panel used for visualization."""
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
      """Convert a KITTI LiDAR `.bin` file into a bird's-eye-view tensor.

      Args:
          bin_path: Path to the KITTI LiDAR binary file.
          x_range: Forward range in meters.
          y_range: Lateral range in meters.
          z_range: Vertical range in meters.
          resolution: Output grid resolution in meters per cell.

      Returns:
          numpy.ndarray: BEV tensor with height, intensity, and density channels.
      """
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
