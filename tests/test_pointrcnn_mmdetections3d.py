import os
import sys
import unittest
import torch
from unittest.mock import MagicMock, patch, create_autospec
import numpy as np

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class TestPointRCNNmmDetections3D(unittest.TestCase):
    # TODO move to integration test
    def test_detect_and_format(self):
        if not torch.cuda.is_available():
            self.skipTest("These tests need GPU support")
        
        from detector.pointrcnnmmdetections3D import PointRCNNmmDetections3D
        pointrcnnmmdetections3D = PointRCNNmmDetections3D(
                "./point_sample.bin",
                "./models/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth"
                )
        prev_results = []
        results = []
        detections = pointrcnnmmdetections3D.format_detections("00001", prev_results,
                                                               results)
        self.assertTrue(len(detections) > 0)  # Smoke test

    def test_format_detections(self):
        test_detect_and_format(self):
        if not torch.cuda.is_available():
            self.skipTest("These tests need GPU support")
        from detector.pointrcnnmmdetections3D import PointRCNNmmDetections3D
        pointrcnnmmdetections3D = PointRCNNmmDetections3D(
                "./point_sample.bin",
                "./models/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth"
                )
        # Each row is (x, y, z, x_size, y_size, z_size, yaw)
        bboxes = torch.tensor([
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            ])

        xyz_centroids = bboxes.numpy()[:, :3]
        lwh_box = bboxe.numpy()[:, 3:5]
        yaw = bboxes.numpy()[:, 6]
        det_scroes = np.array([0.78])

        formatted_detections = pointrcnnmmdetections3.format_detections(
                "00001",
                xyz_centroids,
                lwh_box,
                yaw,
                det_scroes
        )
        self.assertEquals(f"00001,0.2,0.2,0.2,0.2,0.2,1-1,-1,-1")
