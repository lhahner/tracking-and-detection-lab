import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from detector.centerpoint.centerpoint_mmdetection3d import CenterPointMMDetections3D
from entities.detection import DetectionSequence


class TestCenterPointMMDetections3D(unittest.TestCase):
    def test_initializes_mmdetection3d_model_with_config_checkpoint_and_device(self):
        init_model = Mock(return_value="model")
        inference_detector = Mock()
        dataset = Mock()

        with patch.object(
            CenterPointMMDetections3D,
            "_CenterPointMMDetections3D__load_mmdetection3d_apis",
            return_value=(init_model, inference_detector),
        ):
            detector = CenterPointMMDetections3D(
                dataset=dataset,
                classes=[1, 2, 3],
                settings=SimpleNamespace(),
                config_file="config.py",
                checkpoint_file="checkpoint.pth",
                device="cpu",
            )

        init_model.assert_called_once_with("config.py", "checkpoint.pth", device="cpu")
        self.assertEqual(detector.model, "model")
        self.assertTrue(torch.equal(detector.class_map, torch.tensor([1, 2, 3])))

    def test_builds_class_map_from_mmdetection3d_class_order(self):
        model = SimpleNamespace(dataset_meta={"classes": ("car", "truck", "pedestrian")})

        with patch.object(
            CenterPointMMDetections3D,
            "_CenterPointMMDetections3D__load_mmdetection3d_apis",
            return_value=(Mock(return_value=model), Mock()),
        ):
            detector = CenterPointMMDetections3D(
                dataset=Mock(),
                classes={
                    "Background": 0,
                    "pedestrian": 7,
                    "car": 4,
                    "truck": 10,
                },
                settings=SimpleNamespace(),
                config_file="config.py",
                checkpoint_file="checkpoint.pth",
                device="cpu",
            )

        self.assertTrue(torch.equal(detector.class_map, torch.tensor([4, 10, 7])))
        self.assertTrue(torch.equal(
            detector._CenterPointMMDetections3D__map_labels(torch.tensor([0, 2])),
            torch.tensor([4, 7]),
        ))

    def test_detect_runs_inference_for_each_dataset_sample_and_returns_detection_sequence(self):
        init_model = Mock(return_value="model")
        inference_detector = Mock()
        inference_detector.return_value = SimpleNamespace(
            pred_instances_3d=SimpleNamespace(
                scores_3d=torch.tensor([0.25, 0.9]),
                bboxes_3d=SimpleNamespace(
                    tensor=torch.tensor(
                        [
                            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.01, 0.02],
                            [7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 0.2, 0.03, 0.04],
                        ]
                    )
                ),
                labels_3d=torch.tensor([0, 2]),
            )
        )

        class OneSampleDataset:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {
                    "points": np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32),
                    "target": ["target"],
                    "sample_id": "sample-token",
                }

            def custom_collate(self, batch):
                return (
                    [item["points"] for item in batch],
                    [item["target"] for item in batch],
                    [item["sample_id"] for item in batch],
                )

        with patch.object(
            CenterPointMMDetections3D,
            "_CenterPointMMDetections3D__load_mmdetection3d_apis",
            return_value=(init_model, inference_detector),
        ):
            detector = CenterPointMMDetections3D(
                dataset=OneSampleDataset(),
                classes=torch.tensor([1, 2, 3]),
                settings=SimpleNamespace(),
                config_file="config.py",
                checkpoint_file="checkpoint.pth",
                batch_size=1,
                device="cpu",
            )
            detections = detector.detect()

        self.assertIsInstance(detections, DetectionSequence)
        self.assertEqual(len(detections.frames), 1)
        self.assertEqual(detections.frames[0].frame, "sample-token")
        self.assertEqual(detections.frames[0].targets, ["target"])
        self.assertEqual(detections.frames[0].highest_score_index, 1)
        self.assertEqual([detection.label for detection in detections.frames[0].dets], [1, 3])
        self.assertAlmostEqual(detections.frames[0].dets[0].score, 0.25)
        self.assertAlmostEqual(detections.frames[0].dets[1].score, 0.9)
        self.assertEqual(detections.frames[0].dets[0].box.shape, torch.Size([7]))
        self.assertTrue(torch.equal(
            detections.frames[0].dets[0].box,
            torch.tensor([1.0, 2.0, 6.0, 4.0, 5.0, 6.0, 0.1]),
        ))
        inference_detector.assert_called_once()
        np.testing.assert_array_equal(
            inference_detector.call_args.args[1],
            np.array([[1.0, 2.0, 3.0, 0.5, 0.0]], dtype=np.float32),
        )

    def test_prepare_point_input_adds_time_lag_channel_required_by_centerpoint(self):
        with patch.object(
            CenterPointMMDetections3D,
            "_CenterPointMMDetections3D__load_mmdetection3d_apis",
            return_value=(Mock(return_value="model"), Mock()),
        ):
            detector = CenterPointMMDetections3D(
                dataset=Mock(),
                classes=[1],
                settings=SimpleNamespace(),
                config_file="config.py",
                checkpoint_file="checkpoint.pth",
                device="cpu",
            )

        prepared = detector._CenterPointMMDetections3D__prepare_point_input(
            torch.tensor([[1.0, 2.0, 3.0, 0.5]])
        )

        np.testing.assert_array_equal(
            prepared,
            np.array([[1.0, 2.0, 3.0, 0.5, 0.0]], dtype=np.float32),
        )

    def test_predict_instances_unwraps_nested_mmdetection3d_results(self):
        with patch.object(
            CenterPointMMDetections3D,
            "_CenterPointMMDetections3D__load_mmdetection3d_apis",
            return_value=(Mock(return_value="model"), Mock(return_value=[[SimpleNamespace(pred_instances_3d="instances")]])),
        ):
            detector = CenterPointMMDetections3D(
                dataset=Mock(),
                classes=[1],
                settings=SimpleNamespace(),
                config_file="config.py",
                checkpoint_file="checkpoint.pth",
                device="cpu",
            )

        instances = detector._CenterPointMMDetections3D__predict_instances(torch.tensor([[1.0, 2.0, 3.0, 0.5]]))
        self.assertEqual(instances, "instances")

    def test_predict_instances_rejects_results_without_pred_instances_3d(self):
        with patch.object(
            CenterPointMMDetections3D,
            "_CenterPointMMDetections3D__load_mmdetection3d_apis",
            return_value=(Mock(return_value="model"), Mock(return_value=SimpleNamespace())),
        ):
            detector = CenterPointMMDetections3D(
                dataset=Mock(),
                classes=[1],
                settings=SimpleNamespace(),
                config_file="config.py",
                checkpoint_file="checkpoint.pth",
                device="cpu",
            )

        with self.assertRaisesRegex(ValueError, "pred_instances_3d"):
            detector._CenterPointMMDetections3D__predict_instances(np.zeros((1, 4), dtype=np.float32))

    def test_convert_instances_returns_empty_frame_when_model_predicts_no_boxes(self):
        with patch.object(
            CenterPointMMDetections3D,
            "_CenterPointMMDetections3D__load_mmdetection3d_apis",
            return_value=(Mock(return_value="model"), Mock()),
        ):
            detector = CenterPointMMDetections3D(
                dataset=Mock(),
                classes=[1, 2],
                settings=SimpleNamespace(),
                config_file="config.py",
                checkpoint_file="checkpoint.pth",
                device="cpu",
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
        with patch.object(
            CenterPointMMDetections3D,
            "_CenterPointMMDetections3D__load_mmdetection3d_apis",
            return_value=(Mock(return_value="model"), Mock()),
        ):
            detector = CenterPointMMDetections3D(
                dataset=Mock(),
                classes=[4, 7],
                settings=SimpleNamespace(),
                config_file="config.py",
                checkpoint_file="checkpoint.pth",
                device="cpu",
            )

        self.assertTrue(
            torch.equal(
                detector._CenterPointMMDetections3D__map_labels(torch.tensor([0, 1])),
                torch.tensor([4, 7]),
            )
        )
        with self.assertRaisesRegex(ValueError, "outside the configured label map"):
            detector._CenterPointMMDetections3D__map_labels(torch.tensor([2]))


if __name__ == "__main__":
    unittest.main()
