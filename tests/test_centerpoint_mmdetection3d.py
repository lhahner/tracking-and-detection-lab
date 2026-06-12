import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch


class TestCenterPointMMDetections3D(unittest.TestCase):
    def test_builds_class_map_from_mmdetection3d_class_order(self):
        if not torch.cuda.is_available():
            self.skipTest("This Test needs CUDA GPU support")
        try:
            from detector.centerpoint.centerpoint_mmdetection3d import CenterPointMMDetections3D
        except (ImportError, ModuleNotFoundError):
            self.skipTest("This test needs a working centerpoint enviornment")
        detector = CenterPointMMDetections3D(
            dataset=Mock(),
            classes={
                'car': 0,
                'truck': 1,
                'construction_vehicle': 2,
                'bus': 3,
                'trailer': 4,
                'barrier': 5,
                'motorcycle': 6,
                'bicycle': 7,
                'pedestrian': 8,
                'traffic_cone': 9
            },
            settings=SimpleNamespace()
        )
        self.assertTrue(torch.equal(detector.class_map, torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
        self.assertTrue(torch.equal(
            detector._CenterPointMMDetections3D__map_labels(torch.tensor([0, 2])),
            torch.tensor([0, 2]),
        ))

    def test_prepare_point_input_adds_time_lag_channel_required_by_centerpoint(self):
        if not torch.cuda.is_available():
            self.skipTest("This Test needs CUDA GPU support")
        try:
            from detector.centerpoint.centerpoint_mmdetection3d import CenterPointMMDetections3D
        except (ImportError, ModuleNotFoundError):
            self.skipTest("This test needs a working centerpoint enviornment")
        detector = CenterPointMMDetections3D(
            dataset=Mock(),
            classes={
                'car': 0,
                'truck': 1,
                'construction_vehicle': 2,
                'bus': 3,
                'trailer': 4,
                'barrier': 5,
                'motorcycle': 6,
                'bicycle': 7,
                'pedestrian': 8,
                'traffic_cone': 9
            },
            settings=SimpleNamespace()
        )

        prepared = detector._CenterPointMMDetections3D__prepare_point_input(
            torch.tensor([[1.0, 2.0, 3.0, 0.5]])
        )

        np.testing.assert_array_equal(
            prepared,
            np.array([[1.0, 2.0, 3.0, 0.5, 0.0]], dtype=np.float32),
        )

    def test_predict_instances_unwraps_nested_mmdetection3d_results(self):
        if not torch.cuda.is_available():
            self.skipTest("This Test needs CUDA GPU support")
        try:
            from detector.centerpoint.centerpoint_mmdetection3d import CenterPointMMDetections3D
        except (ImportError, ModuleNotFoundError):
            self.skipTest("This test needs a working centerpoint enviornment")
        detector = CenterPointMMDetections3D(
            dataset=Mock(),
            classes={
                'car': 0,
                'truck': 1,
                'construction_vehicle': 2,
                'bus': 3,
                'trailer': 4,
                'barrier': 5,
                'motorcycle': 6,
                'bicycle': 7,
                'pedestrian': 8,
                'traffic_cone': 9
            },
            settings=SimpleNamespace()
        )

        instances = detector._CenterPointMMDetections3D__predict_instances(torch.tensor([[1.0, 2.0, 3.0, 0.5]]))
        self.assertEqual(instances, "instances")

    def test_convert_instances_returns_empty_frame_when_model_predicts_no_boxes(self):
        if not torch.cuda.is_available():
            self.skipTest("This Test needs CUDA GPU support")
        try:
            from detector.centerpoint.centerpoint_mmdetection3d import CenterPointMMDetections3D
        except (ImportError, ModuleNotFoundError):
            self.skipTest("This test needs a working centerpoint enviornment")
        detector = CenterPointMMDetections3D(
            dataset=Mock(),
            classes={
                'car': 0,
                'truck': 1,
                'construction_vehicle': 2,
                'bus': 3,
                'trailer': 4,
                'barrier': 5,
                'motorcycle': 6,
                'bicycle': 7,
                'pedestrian': 8,
                'traffic_cone': 9
            },
            settings=SimpleNamespace(),
        )
        detections, highest_score_index = detector._CenterPointMMDetections3D__convert_instances(
            SimpleNamespace(
                scores_3d=torch.empty(0),
                bboxes_3d=SimpleNamespace(tensor=torch.empty((0, 7))),
                labels_3d=torch.empty(0, dtype=torch.long),
            )
        )

        self.assertEqual(detections, [])
        self.assertIsNone(highest_score_index)

    def test_map_labels_rejects_out_of_range_labels(self):
        if not torch.cuda.is_available():
            self.skipTest("This Test needs CUDA GPU support")
        try:
            from detector.centerpoint.centerpoint_mmdetection3d import CenterPointMMDetections3D
        except (ImportError, ModuleNotFoundError):
            self.skipTest("This test needs a working centerpoint enviornment")
        detector = CenterPointMMDetections3D(
            dataset=Mock(),
            classes={
                'car': 0,
                'truck': 1,
                'construction_vehicle': 2,
                'bus': 3,
                'trailer': 4,
                'barrier': 5,
                'motorcycle': 6,
                'bicycle': 7,
                'pedestrian': 8,
                'traffic_cone': 9
            },
            settings=SimpleNamespace(),
        )

        self.assertTrue(
            torch.equal(
                detector._CenterPointMMDetections3D__map_labels(torch.tensor([0, 1])),
                torch.tensor([0, 1]),
            )
        )
        with self.assertRaisesRegex(ValueError, "outside the configured label map"):
            detector._CenterPointMMDetections3D__map_labels(torch.tensor([12]))


if __name__ == "__main__":
    unittest.main()
