import unittest
from types import SimpleNamespace

import numpy as np
import torch

from entities.detection import Detection, DetectionSequence, FrameDetection
from inference_engine import InferenceEngine


class DummyNuScenesDataset:
    def __init__(self):
        self.targets = [{
            "type": "car",
            "label": 4,
            "box": np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5], dtype=np.float32),
        }]

    def convert_ground_truth(self, ground_truth_dicts):
        rows = []
        for target in ground_truth_dicts:
            rows.append(torch.tensor([*target["box"], 0.0, target["label"]], dtype=torch.float32))
        return torch.stack(rows) if rows else torch.empty((0, 9), dtype=torch.float32)


class TestDETR3DMMDetections3DNuScenesPipeline(unittest.TestCase):
    def build_settings(self):
        return SimpleNamespace(
            paths=SimpleNamespace(detection_path="output/", dataset_path="tests/data/nuscenes_dummy", config_file=""),
            runtime=SimpleNamespace(datatype="png", dataset="nuscenes-mini", display=False),
            benchmark=SimpleNamespace(iou_threshold=0.4, class_filter=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            tracker=SimpleNamespace(max_age=3, min_hits=2, iou_threshold=0.2),
            dataset=SimpleNamespace(classes=["barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck"]),
        )

    def test_evaluate_detection_accepts_lidar_native_nuscenes_boxes(self):
        engine = InferenceEngine(settings=self.build_settings())
        dataset = DummyNuScenesDataset()
        engine.dataset = dataset
        detections = DetectionSequence(frames=[FrameDetection(
            frame="sample-token-1",
            highest_score_index=torch.tensor(0),
            dets=[Detection(score=torch.tensor(0.95), label=torch.tensor(4), box=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5]))],
            targets=dataset.targets,
        )])
        results = engine.evaluate_detection(detections=detections, classes=[4])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["frame"], "sample-token-1")
        self.assertTrue(torch.isclose(results[0]["mAP"], torch.tensor(1.0)))


if __name__ == "__main__":
    unittest.main()
