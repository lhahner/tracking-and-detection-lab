import numpy as np
import torch
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from inference_engine import InferenceEngine
from entities.detection import DetectionSequence, FrameDetection, Detection

class TestInferenceEngine(unittest.TestCase):
    def build_settings(self):
        return SimpleNamespace(
            paths=SimpleNamespace(
                detection_path="/tmp/detections",
                dataset_path="/tmp/dataset",
                config_file="/tmp/config.py",
            ),
            runtime=SimpleNamespace(
                datatype="png",
                dataset="kitti3d",
                display=False,
            ),
            benchmark=SimpleNamespace(iou_threshold=0.4),
            tracker=SimpleNamespace(max_age=3, min_hits=2, iou_threshold=0.2),
            dataset=SimpleNamespace(classes=["Car", "Pedestrian"]),
        )

    @patch("inference_engine.Sort")
    @patch("inference_engine.Visualizer")
    @patch("inference_engine.Evaluation")
    def test_evaluate_detection_with_simple_detection_sequence(
        self,
        mock_evaluation,
        mock_visualizer,
        mock_sort,
    ):
        inference_engine = InferenceEngine(settings=self.build_settings())
        inference_engine.dataset = MagicMock()
        ground_truth_tensor = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 1, 1]], dtype=torch.float32)
        inference_engine.dataset.convert_ground_truth.return_value = ground_truth_tensor
        inference_engine.dataset._load_calib.return_value = ({
            "Tr_velo_to_cam": np.array([[1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
            "R0_rect": np.eye(3, dtype=np.float32),
        }, "/tmp/calib.txt")

        mock_metric = mock_evaluation.return_value
        mock_metric.compute_mAP_3D.return_value = torch.tensor(0.75)

        target = {
            "type": "Car",
            "truncated": 0.1,
            "occluded": 2,
            "alpha": 0.1,
            "bbox": [1, 2, 3],
            "dimensions": [1, 2, 3],
            "location": [1, 2, 3],
            "rotation_y": 0.2,
        }
        detections = DetectionSequence(
            frames=[
                FrameDetection(
                    frame=0,
                    highest_score_index=0,
                    dets=[
                        Detection(
                            score=0.9,
                            label=1,
                            box=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
                        )
                    ],
                    targets=[target],
                )
                ])
        classes = torch.tensor([1])

        results = inference_engine.evaluate_detection(
            detections=detections,
            classes=classes,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["frame"], 0)
        self.assertTrue(torch.equal(results[0]["mAP"], torch.tensor(0.75)))
