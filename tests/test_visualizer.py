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

if "skimage" not in sys.modules:
    skimage_module = types.ModuleType("skimage")
    skimage_module.io = types.SimpleNamespace(imread=MagicMock())
    sys.modules["skimage"] = skimage_module

from util.visualizer import Visualizer

class TestVisualizer(unittest.TestCase):
    @patch("util.visualizer.plt.draw")
    @patch("util.visualizer.plt.subplots")
    @patch("util.visualizer.plt.ion")
    @patch("util.visualizer.io.imread")
    def test_visualize_image_and_bev(self, mock_imread, mock_ion, mock_subplots, mock_draw):
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
        mock_subplots.return_value = (mock_fig, (mock_axis_image, mock_axis_bev))

        visualizer.visualize_image_and_bev(dataset, 0)

        dataset.__getitem__.assert_called_once_with(0)
        mock_imread.assert_called_once_with("/fake/path/image.png")
        visualizer.lidar_bin_to_bev.assert_called_once_with("/fake/path/points.bin")

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


if __name__ == "__main__":
    unittest.main()
