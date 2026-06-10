import unittest
import torch
from types import SimpleNamespace
from datasets.kitti3D import Kitti3D
from detector.pointpillars.pointpillars import Pointpillars
from unittest.mock import Mock, patch
from entities.detection import DetectionSequence, FrameDetection, Detection

class TetsPointPillars(unittest.TestCase):
    def build_settings(self):
        return SimpleNamespace(
            paths=SimpleNamespace(
                detection_path="output/",
                dataset_path="tests/data/kitti3d_dummy",
                config_file="",
            ),
            runtime=SimpleNamespace(
                datatype="png",
                dataset="kitti3d",
                display=False,
            ),
            benchmark=SimpleNamespace(iou_threshold=0.4, class_filter=[1, 2, 3]),
            tracker=SimpleNamespace(max_age=3, min_hits=2, iou_threshold=0.2),
            dataset=SimpleNamespace(classes=["Pedestrian", "Cyclist", "Car"]),
        )

    def test_filter_points_for_inference_filters_points_correctly(self):
        points = [torch.randn(256, 4)]
        pointpillars = Pointpillars(dataset=Mock(),
                                    config_file = "",
                                    classes=[1, 2, 3],
                                    settings=self.build_settings()
                                    )
        filtered_points = pointpillars._Pointpillars__filter_points_for_inference(points)
        breakpoint()
        self.assertTrue(points[0].shape[0] > len(filtered_points[0]))


