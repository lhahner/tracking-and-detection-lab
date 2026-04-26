import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

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
    def test_visualize_tracking_frame_render_image(self,
                                                   mock_imread):
        visualizer = Visualizer("bin")
        fake_image = np.zeros((375, 1242, 3), dtype=np.uint8)
        mock_imread.return_value = fake_image

        visualizer.fig = MagicMock()
        visualizer.fig.canvas = MagicMock()
        visualizer.axis_one = MagicMock()
        visualizer.axis_one.title = MagicMock()
        visualizer.axis_idf1 = MagicMock()
        visualizer.axis_motp = MagicMock()
        visualizer.axis_mota = MagicMock()

        trackers = np.array([[10, 20, 30, 40, 5]])
        colours = np.ones((32, 3), dtype=np.float32)
        metrics_history = {"idf1": [0.5]}

        with patch.object(visualizer,
                          "_Visualizer__visualize_boxes") as mock_boxes, \
                patch.object(visualizer,
                             "_Visualizer__visualize_metric_plots") as \
                mock_metrics, \
                patch.object(visualizer,
                             "_Visualizer__visualize_and_draw") as mock_draw:
            visualizer.visualize_tracking_frame("/fake/dataset",
                                                123,
                                                "png",
                                                trackers,
                                                colours,
                                                metrics_history)

        mock_imread.assert_called_once_with("/fake/dataset/000123.png")
        visualizer.axis_one.imshow.assert_called_once_with(fake_image)
        visualizer.axis_one.title.set_text.assert_called_once()
        visualizer.axis_one.axis.assert_called_once_with("off")
        mock_boxes.assert_called_once_with(trackers, colours)
        mock_metrics.assert_called_once_with(metrics_history)
        visualizer.fig.tight_layout.assert_called_once()
        mock_draw.assert_called_once()

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

    @patch("util.visualizer.np.fromfile")
    def test_lidar_bin_to_bev_synthetic_point_cloud_transformed(self,
                                                                mock_fromfile):
        points = np.array([0.3, 0.5, 0.4, 0.7, 0.1, 0.1, 0.1, 0.1])
        mock_fromfile.return_value = points
        visualizer = Visualizer("bin")
        bev = visualizer.lidar_bin_to_bev("")

        max_height = bev[bev != 0][0]
        max_intensity = bev[bev != 0][1]
        max_density = bev[bev != 0][2]

        self.assertEquals(np.floor(max_height), 0)
        self.assertEquals(np.floor(max_intensity), 0)
        self.assertEquals(np.floor(max_density), 0)

    @patch("util.visualizer.np.fromfile")
    def test_lidar_bin_to_bev_negative_tests(self,
                                             mock_fromfile):
        points = np.array([0.3, 0.5, 0.4])
        mock_fromfile.return_value = points
        visualizer = Visualizer("bin")
        with self.assertRaises(ValueError):
            visualizer.lidar_bin_to_bev("")

        with self.assertRaises(ValueError):
            visualizer.lidar_bin_to_bev("",
                                        (40),
                                        (-40.0, 40.0),
                                        (-3.0),
                                        0.1)

        points = np.array([0.3, 0.5, 0.4, 0.7])
        x_range = (0.0, 50.0)
        y_range = (-50.0, 50.0)
        z_range = (-3.0, 1.0)
        resolution = 0.1
        mock_fromfile.return_value = points

        expected_width = (x_range[1] - x_range[0]) / resolution
        expected_heigth = (y_range[1] - y_range[0]) / resolution

        bev = visualizer.lidar_bin_to_bev("",
                                          x_range,
                                          y_range,
                                          z_range,
                                          0)
        self.assertEquals(bev.shape, (expected_heigth, expected_width, 3))


if __name__ == "__main__":
    unittest.main()
