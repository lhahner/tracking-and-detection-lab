import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from detector.detr3d.detr3d_mmdetection3d import DETR3DMMDetections3D


NUSCENES_CLASSES = {
    "Background": 0,
    "barrier": 1,
    "bicycle": 2,
    "bus": 3,
    "car": 4,
    "construction_vehicle": 5,
    "motorcycle": 6,
    "pedestrian": 7,
    "traffic_cone": 8,
    "trailer": 9,
    "truck": 10,
}


def make_prediction(scores, bboxes, labels=None):
    if labels is None:
        labels = list(range(len(scores)))
    pred_instances = types.SimpleNamespace(
        scores_3d=torch.tensor(scores, dtype=torch.float32),
        labels_3d=torch.tensor(labels, dtype=torch.int64),
        bboxes_3d=types.SimpleNamespace(tensor=torch.tensor(bboxes, dtype=torch.float32)),
    )
    return [[types.SimpleNamespace(pred_instances_3d=pred_instances)]]


class DummyNuScenesDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.items = [{
            "points": np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            "target": [{
                "type": "car",
                "label": 4,
                "box": np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5], dtype=np.float32),
            }],
            "sample_id": "sample-token-1",
        }]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def custom_collate(self, batch):
        return ([item["points"] for item in batch], [item["target"] for item in batch], [item["sample_id"] for item in batch])


class TestDETR3DMMDetections3D(unittest.TestCase):
    def setUp(self):
        self.apis_module = types.ModuleType("mmdet3d.apis")
        self.apis_module.init_model = Mock(name="init_model", return_value="model")
        self.apis_module.inference_detector = Mock(name="inference_detector")
        fake_modules = {"mmdet3d": types.ModuleType("mmdet3d"), "mmdet3d.apis": self.apis_module}
        self.modules_patcher = patch.dict(sys.modules, fake_modules)
        self.cuda_patcher = patch("torch.cuda.is_available", return_value=True)
        self.modules_patcher.start()
        self.cuda_patcher.start()

    def tearDown(self):
        self.cuda_patcher.stop()
        self.modules_patcher.stop()

    def test_init_sets_attributes_and_initializes_model(self):
        dataset = Mock()
        detector = DETR3DMMDetections3D(dataset=dataset, config_file="config.py", classes=NUSCENES_CLASSES,
                               settings=Mock(), checkpoint_file="checkpoint.pth", batch_size=2,
                               num_inference_samples=3, device="cuda:0")
        self.assertIs(detector.dataset, dataset)
        self.assertEqual(detector.config_file, "config.py")
        self.assertEqual(detector.checkpoint_file, "checkpoint.pth")
        self.assertEqual(detector.model, "model")
        self.assertEqual(detector.batch_size, 2)
        self.assertEqual(detector.num_inference_samples, 3)
        self.apis_module.init_model.assert_called_once_with("config.py", "checkpoint.pth", device="cuda:0")

    def test_detect_returns_detection_sequence_for_nuscenes_labels(self):
        detector = DETR3DMMDetections3D(DummyNuScenesDataset(), "config.py", NUSCENES_CLASSES, Mock(), "checkpoint.pth", batch_size=1)
        self.apis_module.inference_detector.return_value = make_prediction(
            scores=[0.91], bboxes=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5]], labels=[3])
        detections = detector.detect()
        self.assertEqual(len(detections.frames), 1)
        frame = detections.frames[0]
        self.assertEqual(frame.frame, "sample-token-1")
        self.assertEqual(len(frame.dets), 1)
        self.assertTrue(torch.equal(frame.dets[0].label, torch.tensor(4)))
        self.assertTrue(torch.isclose(frame.dets[0].score, torch.tensor(0.91)))

    def test_detect_adds_empty_frame_for_no_predictions(self):
        detector = DETR3DMMDetections3D(DummyNuScenesDataset(), "config.py", NUSCENES_CLASSES, Mock(), "checkpoint.pth", batch_size=1)
        self.apis_module.inference_detector.return_value = make_prediction(scores=[], bboxes=[], labels=[])
        detections = detector.detect()
        self.assertEqual(len(detections.frames), 1)
        self.assertEqual(detections.frames[0].dets, [])
        self.assertIsNone(detections.frames[0].highest_score_index)

    def test_detect_respects_num_inference_samples_limit(self):
        dataset = DummyNuScenesDataset()
        dataset.items.append({**dataset.items[0], "sample_id": "sample-token-2"})
        detector = DETR3DMMDetections3D(dataset, "config.py", NUSCENES_CLASSES, Mock(), "checkpoint.pth", batch_size=1, num_inference_samples=1)
        self.apis_module.inference_detector.return_value = make_prediction(
            scores=[0.91], bboxes=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5]], labels=[3])
        detections = detector.detect()
        self.assertEqual(len(detections.frames), 1)
        self.assertEqual(detections.frames[0].frame, "sample-token-1")


if __name__ == "__main__":
    unittest.main()
