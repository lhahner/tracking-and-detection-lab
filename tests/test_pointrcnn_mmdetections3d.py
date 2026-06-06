import importlib
import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch


def load_pointrcnn_module():
    apis_module = types.ModuleType("mmdet3d.apis")
    apis_module.init_model = Mock(name="init_model", return_value="model")
    apis_module.inference_detector = Mock(name="inference_detector")

    bbox_module = types.ModuleType("mmdet3d.structures.bbox_3d")

    class FakeBox3DMode:
        LIDAR = "lidar"
        CAM = "cam"

        @staticmethod
        def convert(tensor, source, destination):
            return tensor

    bbox_module.Box3DMode = FakeBox3DMode

    logging_module = types.ModuleType("util.logging_config")

    class FakeLoggingConfig:
        def get_logger(self, name):
            return Mock(name=f"logger:{name}")

    logging_module.LoggingConfig = FakeLoggingConfig

    fake_modules = {
        "mmdet3d": types.ModuleType("mmdet3d"),
        "mmdet3d.apis": apis_module,
        "mmdet3d.structures": types.ModuleType("mmdet3d.structures"),
        "mmdet3d.structures.bbox_3d": bbox_module,
        "util.logging_config": logging_module,
    }

    module_name = "detector.pointrcnn.pointrcnn_mmdetection3d"
    sys.modules.pop(module_name, None)
    with patch.dict(sys.modules, fake_modules), patch("torch.cuda.is_available", return_value=True):
        return importlib.import_module(module_name)


def make_prediction(scores, bboxes, labels=None):
    if labels is None:
        labels = list(range(len(scores)))
    pred_instances = types.SimpleNamespace(
        scores_3d=torch.tensor(scores, dtype=torch.float32),
        labels_3d=torch.tensor(labels, dtype=torch.int64),
        bboxes_3d=types.SimpleNamespace(tensor=torch.tensor(bboxes, dtype=torch.float32)),
    )
    return [[types.SimpleNamespace(pred_instances_3d=pred_instances)]]


class TestPointRCNNmmDetections3D(unittest.TestCase):
    def setUp(self):
        self.module = load_pointrcnn_module()

    def make_detector_without_init(self, **overrides):
        detector = self.module.PointRCNNmmDetections3D.__new__(self.module.PointRCNNmmDetections3D)
        detector.dataset = [{"points": "point-cloud-1", "sample_id": 42, "target": "target-1"}]
        detector.batch_size = 1
        detector.model = "model"
        detector.num_inference_samples = 1
        detector.classes = {0: "Car"}
        detector.settings = Mock()
        detector.serializer = Mock()
        for key, value in overrides.items():
            setattr(detector, key, value)
        return detector

    def test_init_sets_attributes_and_initializes_model(self):
        dataset = Mock()

        detector = self.module.PointRCNNmmDetections3D(
            dataset=dataset,
            config_file="config.py",
            classes={0: "Car"},
            checkpoint_file="checkpoint.pth",
            batch_size=2,
            settings=Mock(),
            num_inference_samples=3,
        )

        self.assertIs(detector.dataset, dataset)
        self.assertEqual(detector.config_file, "config.py")
        self.assertEqual(detector.checkpoint_file, "checkpoint.pth")
        self.assertEqual(detector.model, "model")
        self.assertEqual(detector.classes, {0: "Car"})
        self.assertEqual(detector.batch_size, 2)
        self.assertEqual(detector.num_inference_samples, 3)
        self.module.init_model.assert_called_once_with("config.py", "checkpoint.pth")

    def test_init_propagates_model_initialization_error(self):
        self.module.init_model.side_effect = RuntimeError("failed to load model")

        with self.assertRaises(RuntimeError):
            self.module.PointRCNNmmDetections3D(
                dataset=[],
                config_file="config.py",
                classes={0: "Car"},
                settings=Mock(),
                checkpoint_file="checkpoint.pth",
            )

    @patch("torch.utils.data.DataLoader", side_effect=lambda dataset, batch_size, collate_fn: [collate_fn(dataset)])
    def test_detect_returns_detection_sequence(self, _mock_dataloader):
        detector = self.make_detector_without_init()
        self.module.inference_detector.side_effect = [
            make_prediction(scores=[0.9], bboxes=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1]], labels=[2]),
            make_prediction(scores=[0.7], bboxes=[[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0.2]], labels=[2]),
        ]

        detection_sequence = detector.detect()

        self.assertEqual(len(detection_sequence.frames), 1)
        frame = detection_sequence.frames[0]
        self.assertEqual(frame.frame, [42])
        self.assertEqual(frame.highest_score_index.item(), 0)
        self.assertEqual(len(frame.dets), 1)
        self.assertTrue(torch.equal(frame.dets[0].score, torch.tensor(0.7)))
        self.assertEqual(frame.dets[0].label.item(), 2)
        self.assertTrue(torch.equal(frame.dets[0].box, torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0.2])))

    @patch("torch.utils.data.DataLoader", side_effect=lambda dataset, batch_size, collate_fn: [collate_fn(dataset)])
    def test_detect_propagates_inference_error(self, _mock_dataloader):
        detector = self.make_detector_without_init()
        self.module.inference_detector.side_effect = RuntimeError("inference failed")

        with self.assertRaises(RuntimeError):
            detector.detect()

    def test_sample_pads_short_predictions_to_expected_object_count(self):
        detector = self.make_detector_without_init(num_inference_samples=1)
        self.module.inference_detector.return_value = make_prediction(
            scores=[0.8],
            bboxes=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1]],
        )

        all_bboxes, all_scores = detector._PointRCNNmmDetections3D__sample("point-cloud-1", num_obj=2)

        self.assertEqual(len(all_bboxes), 1)
        self.assertTrue(torch.equal(all_bboxes[0][0], torch.zeros(7)))
        self.assertTrue(torch.equal(all_bboxes[0][1], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1])))
        self.assertTrue(torch.equal(all_scores[0], torch.tensor([0.8, 0.0])))

    def test_sample_rejects_mismatched_bbox_and_score_counts(self):
        detector = self.make_detector_without_init(num_inference_samples=1)
        self.module.inference_detector.return_value = make_prediction(
            scores=[0.8],
            bboxes=[
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1],
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.2],
            ],
        )

        with self.assertRaises(ValueError):
            detector._PointRCNNmmDetections3D__sample("point-cloud-1", num_obj=2)

    def test_mean_nonzero_averages_only_nonzero_values(self):
        detector = self.make_detector_without_init()
        tensor = torch.tensor([
            [[1.0, 0.0], [0.0, 4.0]],
            [[3.0, 2.0], [6.0, 0.0]],
        ])

        result = detector._PointRCNNmmDetections3D__mean_nonzero(tensor)

        self.assertTrue(torch.equal(result, torch.tensor([[[2.0, 2.0], [6.0, 4.0]]])))

    def test_mean_nonzero_rejects_empty_tensor(self):
        detector = self.make_detector_without_init()

        with self.assertRaises(ValueError):
            detector._PointRCNNmmDetections3D__mean_nonzero(torch.tensor([]))

    def test_custom_collate_returns_points_and_sample_ids(self):
        batch = [
                {"points": "point-cloud-1", "sample_id": "000001", "target": "target-1"},
                {"points": "point-cloud-2", "sample_id": "000002", "target": "target-2"},
        ]

        points, targets, sample_ids = self.module.custom_collate(batch)

        self.assertEqual(points, ["point-cloud-1", "point-cloud-2"])
        self.assertEqual(sample_ids, ["000001", "000002"])

    def test_custom_collate_rejects_missing_sample_id(self):
        batch = [{"points": "point-cloud-1"},
                 {"targets": "point-cloud-2"}]

        with self.assertRaises(KeyError):
            self.module.custom_collate(batch)


if __name__ == "__main__":
    unittest.main()
