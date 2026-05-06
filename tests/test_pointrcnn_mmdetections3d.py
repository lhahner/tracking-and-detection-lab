import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from detector.pointrcnnmmdetections3D import PointRCNNmmDetections3D


class TestPointRCNNmmDetections3D(unittest.TestCase):
    def test_detect_and_format(self):
        pointrcnnmmdetections3D = PointRCNNmmDetections3D(
                "./point_sample.bin",
                "./models/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth"
                )
        prev_results = []
        results = []
        detections = pointrcnnmmdetections3D.format_detections("00001", prev_results,
                                                               results)
        self.assertTrue(len(detections) > 0)  # Smoke test
