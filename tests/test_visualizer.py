import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import matplotlib.pyplot as plt

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from util.visualizer import Visualizer

if "skimage" not in sys.modules:
    skimage_module = types.ModuleType("skimage")
    skimage_module.io = types.SimpleNamespace(imread=MagicMock())
    sys.modules["skimage"] = skimage_module


class TestVisualizer(unittest.TestCase):
    @patch("util.visualizer.plt.draw")
    @patch("util.visualizer.plt.subplots")
    @patch("util.visualizer.plt.ion")
    @patch("util.visualizer.io.imread")
    def test_visualize_image_and_bev(self,
                                     mock_imread,
                                     mock_ion,
                                     mock_subplots,
                                     mock_draw):
        visualizer = Visualizer("bin")

        dataset = MagicMock()
        dataset.__getitem__.return_value = {
            "image_path": "/fake/path/image.png",
            "points_path": "/fake/path/points.bin",
            "sample_id": "000123",
        }

        fake_image = np.zeros((375, 1242, 3), dtype=np.uint8)
        fake_bev = np.ones((200, 200), dtype=np.float32)

        mock_imread.return_value = fake_image

        visualizer.lidar_bin_to_bev = MagicMock(return_value=fake_bev)

        mock_fig = MagicMock()
        mock_axis_image = MagicMock()
        mock_axis_bev = MagicMock()
        mock_image_bev_tupel = (mock_axis_image, mock_axis_bev)
        mock_subplots.return_value = (mock_fig, mock_image_bev_tupel)
        visualizer.visualize_image_and_bev(dataset, 0)

        dataset.__getitem__.assert_called_once_with(0)
        mock_imread.assert_called_once_with("/fake/path/image.png")
        fake_path = "/fake/path/points.bin"
        visualizer.lidar_bin_to_bev.assert_called_once_with(fake_path)

        mock_ion.assert_called_once()
        mock_subplots.assert_called_once_with(1, 2, figsize=(14, 6))

        mock_axis_image.imshow.assert_called_once_with(fake_image)
        mock_axis_image.set_title.assert_called_once_with("KITTI Image 123")
        mock_axis_image.axis.assert_called_once_with("off")

        mock_axis_bev.imshow.assert_called_once()
        bev_arg = mock_axis_bev.imshow.call_args[0][0]
        np.testing.assert_array_equal(bev_arg, np.flipud(fake_bev))

        mock_axis_bev.set_title.assert_called_once_with("LiDAR BEV 123")
        mock_axis_bev.set_xlabel.assert_called_once_with("Forward")
        mock_axis_bev.set_ylabel.assert_called_once_with("Left / Right")

        mock_fig.tight_layout.assert_called_once()
        mock_draw.assert_called_once()

    @patch("util.visualizer.io.imread")
    @patch("util.visualizer.setup_panel")
    @patch("util.visualizer.axis_one")
    def test_visualize_tracking_frame_render_image(self,
                                                   mock_imread,
                                                   mock_setup_panel,
                                                   mock_axis_one):
        visualizer = Visualizer("bin")

        dataset = MagicMock()
        dataset.__getitem__.return_value = {
            "image_path": "/fake/path/image.png",
            "points_path": "/fake/path/points.bin",
            "sample_id": "000123",
        }

        fake_image = np.zeros((375, 1242, 3), dtype=np.uint8)

        mock_imread.return_value = fake_image
        plt.ion()
        visualizer.fig = plt.figure(figsize=(13, 7))
        grid = visualizer.fig.add_gridspec(
            nrows=4,
            ncols=3,
            width_ratios=[4, 4, 4],
            height_ratios=[4, 4, 4, 4],
            hspace=0.35,
            wspace=0.18,
        )
        visualizer.axis_one = visualizer.fig.add_subplot(
                grid[0:3, 0:3],
                aspect="equal")
        visualizer.axis_idf1 = visualizer.fig.add_subplot(grid[3, 0])
        visualizer.axis_motp = visualizer.fig.add_subplot(grid[3, 1])
        visualizer.axis_mota = visualizer.fig.add_subplot(grid[3, 2])
        axes = [
            visualizer.axis_one,
            visualizer.axis_idf1,
            visualizer.axis_motp,
            visualizer.axis_mota,
        ]
        for axis in axes:
            if axis is not None:
                axis.cla()
        mock_axis_one.imshow.assert_called_once()
        visualizer.__visualize_boxes.assert_called_once()
        visualizer.__visualize_metric_plots.assert_called_once()
        visualizer.fig.tight_layout.assert_called_once()
        visualizer.__visualize_and_draw.assert_called_once()

    def test_lidar_bin_to_bev_default_range_and_resolution(self):
        path_to_test_bin = "tests/point_sample.bin"
        visualizer = Visualizer("bin")
        x_range = (0.0, 50.0)
        y_range = (-50.0, 50.0)
        z_range = (-3.0, 1.0)
        resolution = 0.1
        bev = visualizer.lidar_bin_to_bev(path_to_test_bin,
                                          x_range,
                                          y_range,
                                          z_range,
                                          resolution
                                          )
        expected_width = (x_range[1] - x_range[0]) / resolution
        expected_heigth = (y_range[1] - y_range[0]) / resolution
        self.assertEquals(bev.shape, (expected_heigth, expected_width, 3))
    # TODO Behavior Test for lidar_bin_to_bev
    def test_lidar_bin_to_bev_synthetic_point_cloud_transformed():
        points = np.array([0.1, 0.1, 0.1],
                          [0.1, 0.1, 0.1],
                          [0.1, 0.1, 0.1])
        raise NotImplementedError()

    # TODO negative Test for lidar_bin_to_bev

if __name__ == "__main__":
    unittest.main()
