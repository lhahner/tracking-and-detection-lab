import math
import unittest

import torch

from geometry.coordinate_converter import CoordinateConverter


class TestCoordinateConverter(unittest.TestCase):
    def setUp(self):
        self.converter = CoordinateConverter()

    def assert_corners_equal(self, actual, expected):
        torch.testing.assert_close(
            actual, torch.tensor(expected, dtype=actual.dtype), atol=1e-5, rtol=0
        )

    def test_lidar_corners_use_bottom_center_and_documented_order(self):
        boxes = torch.tensor([[10.0, 20.0, 30.0, 2.0, 4.0, 6.0, 0.0]])

        corners = self.converter.boxes_3d_to_corners(boxes, "lidar")

        self.assertEqual(corners.shape, (1, 8, 3))
        self.assert_corners_equal(corners[0], [
            [9, 18, 30], [9, 18, 36], [9, 22, 36], [9, 22, 30],
            [11, 18, 30], [11, 18, 36], [11, 22, 36], [11, 22, 30],
        ])

    def test_camera_corners_use_bottom_center_and_documented_order(self):
        boxes = torch.tensor([[10.0, 20.0, 30.0, 2.0, 4.0, 6.0, 0.0]])

        corners = self.converter.boxes_3d_to_corners(boxes, "camera")

        self.assertEqual(corners.shape, (1, 8, 3))
        self.assert_corners_equal(corners[0], [
            [9, 16, 27], [9, 16, 33], [9, 20, 33], [9, 20, 27],
            [11, 16, 27], [11, 16, 33], [11, 20, 33], [11, 20, 27],
        ])

    def test_lidar_yaw_rotates_corners_around_z_axis(self):
        boxes = torch.tensor([[10.0, 20.0, 30.0, 2.0, 4.0, 6.0, math.pi / 2]])

        corners = self.converter.boxes_3d_to_corners(boxes, "lidar")

        self.assert_corners_equal(corners[0], [
            [12, 19, 30], [12, 19, 36], [8, 19, 36], [8, 19, 30],
            [12, 21, 30], [12, 21, 36], [8, 21, 36], [8, 21, 30],
        ])

    def test_camera_yaw_rotates_corners_around_y_axis(self):
        boxes = torch.tensor([[10.0, 20.0, 30.0, 2.0, 4.0, 6.0, math.pi / 2]])

        corners = self.converter.boxes_3d_to_corners(boxes, "camera")

        self.assert_corners_equal(corners[0], [
            [7, 16, 31], [13, 16, 31], [13, 20, 31], [7, 20, 31],
            [7, 16, 29], [13, 16, 29], [13, 20, 29], [7, 20, 29],
        ])

    def test_multiple_boxes_keep_their_own_centers_and_dimensions(self):
        boxes = torch.tensor([
            [0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0],
            [5.0, -3.0, 2.0, 4.0, 2.0, 1.0, 0.0],
        ])

        corners = self.converter.boxes_3d_to_corners(boxes, "lidar")

        self.assertEqual(corners.shape, (2, 8, 3))
        self.assert_corners_equal(corners[0, 0], [-1, -2, 0])
        self.assert_corners_equal(corners[1, 0], [3, -4, 2])
        self.assert_corners_equal(corners[1, 6], [7, -2, 3])

    def test_empty_boxes_return_empty_corners(self):
        boxes = torch.empty((0, 7), dtype=torch.float64)

        for mode in ("camera", "lidar"):
            with self.subTest(mode=mode):
                corners = self.converter.boxes_3d_to_corners(boxes, mode)
                self.assertEqual(corners.shape, (0, 8, 3))
                self.assertEqual(corners.device, boxes.device)

    def test_unsupported_box_mode_raises_value_error(self):
        boxes = torch.empty((0, 7))

        with self.assertRaisesRegex(ValueError, "Unsupported box_mode invalid"):
            self.converter.boxes_3d_to_corners(boxes, "invalid")


if __name__ == "__main__":
    unittest.main()
