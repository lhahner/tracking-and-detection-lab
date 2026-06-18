import math
import os
import sys
import unittest

import torch

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from geometry.coordinate_converter import CoordinateConverter


class TestCoordinateConverter(unittest.TestCase):
    def test_boxes_3d_to_corners_uses_pytorch3d_order(self):
        self.skipTest("Deprecated")
        converter = CoordinateConverter()
        boxes = torch.tensor([[0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0]], dtype=torch.float32)

        corners = converter.boxes_3d_to_corners(boxes)

        expected_corners = torch.tensor([
            [-1.0, -2.0, -3.0],
            [1.0, -2.0, -3.0],
            [1.0, 2.0, -3.0],
            [-1.0, 2.0, -3.0],
            [-1.0, -2.0, 3.0],
            [1.0, -2.0, 3.0],
            [1.0, 2.0, 3.0],
            [-1.0, 2.0, 3.0],
        ], dtype=torch.float32)

        self.assertTrue(torch.allclose(corners[0], expected_corners, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
