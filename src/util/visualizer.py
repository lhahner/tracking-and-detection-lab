import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import numpy as np
from util.logging_config import LoggingConfig

logging_config = LoggingConfig()
logger = logging_config.get_logger(__name__)


class Visualizer:
    def __init__(self, datatype):
        self.datatype = datatype
        self.fig = None
        self.axis_one = None
        self.axis_idf1 = None
        self.axis_motp = None
        self.axis_mota = None

    def __visualize_boxes(self, bounding_box, colours):
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

    def __visualize_and_draw(self):
        self.fig.canvas.flush_events()
        plt.draw()
        self.__clear_dashboard_axes()

    def setup_panel(self):
        plt.ion()
        self.fig = plt.figure(figsize=(13, 7))
        grid = self.fig.add_gridspec(
            nrows=4,
            ncols=3,
            width_ratios=[4, 4, 4],
            height_ratios=[4, 4, 4, 4],
            hspace=0.35,
            wspace=0.18,
        )
        self.axis_one = self.fig.add_subplot(grid[0:3, 0:3], aspect="equal")
        self.axis_idf1 = self.fig.add_subplot(grid[3, 0])
        self.axis_motp = self.fig.add_subplot(grid[3, 1])
        self.axis_mota = self.fig.add_subplot(grid[3, 2])
        self.__clear_dashboard_axes()

    def visualize_tracking_frame(self,
                                 dataset_path,
                                 frame,
                                 filetype,
                                 trackers,
                                 colours,
                                 metrics_history=None):
        if self.fig is None:
            self.setup_panel()

        filename = f"{frame:06d}.{filetype}"
        image_path = os.path.join(dataset_path, filename)
        image = io.imread(image_path)

        self.axis_one.imshow(image)
        self.axis_one.title.set_text(f"Dataset: ...{str(dataset_path)[-20:]}\n"
                                     f"Frame: {frame:06d}.{filetype}\n"
                                     )
        self.axis_one.axis("off")

        self.__visualize_boxes(trackers, colours)
        self.__visualize_metric_plots(metrics_history or {})

        self.fig.tight_layout()
        self.__visualize_and_draw()

    def __clear_dashboard_axes(self):
        axes = [
            self.axis_one,
            self.axis_idf1,
            self.axis_motp,
            self.axis_mota,
        ]
        for axis in axes:
            if axis is not None:
                axis.cla()

    def __visualize_metric_plots(self, metrics_history):
        metric_axes = [
            ("idf1", "IDF1", self.axis_idf1),
            ("motp", "MOTP", self.axis_motp),
            ("mota", "MOTA", self.axis_mota),
        ]
        for metric_key, metric_label, axis in metric_axes:
            if axis is None:
                continue
            values = metrics_history.get(metric_key, [])
            axis.set_title(metric_label, fontsize=9)
            axis.tick_params(labelsize=7)
            axis.grid(True, alpha=0.25)
            if values:
                x_values = np.arange(1, len(values) + 1)
                axis.plot(x_values, values, color="tab:red", lw=1.4)
                axis.text(
                    0.5,
                    -0.28,
                    f"{metric_label}: {values[-1]:.3f}",
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
            else:
                axis.text(
                    0.5,
                    0.5,
                    f"{metric_label}: n/a",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    def lidar_bin_to_bev(
      self,
      bin_path,
      x_range=(0.0, 70.4),
      y_range=(-40.0, 40.0),
      z_range=(-3.0, 1.0),
      resolution=0.1):
        """
        Resolutions defines the scale of the image always consider,
        x_range / resolution and y_range /resoltion provide width
        and height of the image.
        """
        if not all(isinstance(i, tuple) for i in [x_range, y_range, z_range]):
            raise ValueError("The range should in the format of [0, 80]")

        points = np.fromfile(bin_path, dtype=np.float32)
        if points.shape[0] % 4 != 0 and len(points.shape) == 1:
            raise ValueError("The flat pointcloud should be divided "
                             "in x,y,z and intensity")

        if resolution <= 0:
            logger.warn("Division by zero will cause Programm interrupt" 
                        "continuing with 0.1 resolution.")
            resolution = 0.1

        points = points.reshape(-1, 4)
        x, y, z, i = points.T

        mask = (
          (x >= x_range[0]) & (x <= x_range[1]) &
          (y >= y_range[0]) & (y <= y_range[1]) &
          (z >= z_range[0]) & (z <= z_range[1])
        )

        # filtering away all points out of range
        x, y, z, i = x[mask], y[mask], z[mask], i[mask]

        # image size
        width = int((x_range[1] - x_range[0]) / resolution)
        heigth = int((y_range[1] - y_range[0]) / resolution)

        # scale and round
        x_idx = np.floor((x - x_range[0]) / resolution).astype(np.int32)
        y_idx = np.floor((y - y_range[0]) / resolution).astype(np.int32)

        # clipping values outside of the integral to the edges
        x_idx = np.clip(x_idx, 0, width - 1)
        y_idx = np.clip(y_idx, 0, heigth - 1)

        max_height = np.full((heigth, width), -np.inf, dtype=np.float32)
        max_intensity = np.zeros((heigth, width), dtype=np.float32)
        counts = np.zeros((heigth, width), dtype=np.int32)
        # Find the largest values in terms of height and intensity
        for xi, yi, zi, ii in zip(x_idx, y_idx, z, i):
            if zi > max_height[yi, xi]:
                max_height[yi, xi] = zi
            if ii > max_intensity[yi, xi]:
                max_intensity[yi, xi] = ii
            counts[yi, xi] += 1
        # All values not set by our loop
        max_height[max_height == -np.inf] = z_range[0]
        z_range_difference = (z_range[1] - z_range[0] + 1e-6)
        max_height = (max_height - z_range[0]) / z_range_difference
        # Density is defined by the counts, more points = higher density
        density = np.log(counts + 1) / np.log(64)
        density = np.clip(density, 0, 1)

        bev = np.stack([max_height, max_intensity, density], axis=-1)
        return bev

    def visualize_image_and_bev(self, dataset, dataitem, split="training"):
        frame = dataset[dataitem]
        normalized_sample_id = frame["sample_id"].lstrip('0')
        image = io.imread(frame["image_path"])
        bev = self.lidar_bin_to_bev(frame["points_path"])

        plt.ion()
        fig, (axis_image, axis_bev) = plt.subplots(1, 2, figsize=(14, 6))
        axis_image.imshow(image)
        axis_image.set_title(f"KITTI Image {normalized_sample_id}")
        axis_image.axis("off")

        axis_bev.imshow(np.flipud(bev))
        axis_bev.set_title(f"LiDAR BEV {normalized_sample_id}")
        axis_bev.set_xlabel("Forward")
        axis_bev.set_ylabel("Left / Right")

        fig.tight_layout()
        plt.draw()
