import csv
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from entities.detection import Detection, DetectionSequence, FrameDetection
from util.evaluation import Evaluation


class TestEvaluationIoUAndAnalysis:
    def build_detection_sequence(self):
        frame = FrameDetection(
            frame="sample-1",
            highest_score_index=0,
            dets=[
                Detection(
                    score=torch.tensor(0.91),
                    label=4,
                    box=torch.tensor([1.0, 2.0, 3.0, 4.0, 1.8, 1.6, 0.1]),
                ),
                Detection(
                    score=torch.tensor(0.42),
                    label=7,
                    box=torch.tensor([6.0, 2.5, 1.0, 0.8, 0.7, 1.7, 0.0]),
                ),
            ],
            targets=[
                {
                    "type": "Cyclist",
                    "label": 2,
                    "box": np.asarray([1.0, 2.0, 3.0, 4.0, 1.8, 1.6, 0.1], dtype=np.float32),
                },
                {
                    "type": "Car",
                    "label": 3,
                    "box": np.asarray([10.0, 1.0, 1.0, 4.5, 1.9, 1.5, 0.0], dtype=np.float32),
                },
            ],
        )
        return DetectionSequence(frames=[frame])

    def install_fake_pytorch3d(self, iou_matrix):
        fake_root = types.ModuleType("pytorch3d")
        fake_ops = types.ModuleType("pytorch3d.ops")

        def fake_box3d_overlap(prediction_corner_boxes, ground_truth_corner_boxes):
            volume = torch.zeros((prediction_corner_boxes.shape[0], ground_truth_corner_boxes.shape[0]))
            return volume, iou_matrix.clone()

        fake_ops.box3d_overlap = fake_box3d_overlap
        return patch.dict(sys.modules, {"pytorch3d": fake_root, "pytorch3d.ops": fake_ops})

    def test_compute_iou_3d_returns_per_frame_iou_matrix(self):
        detection_sequence = self.build_detection_sequence()
        evaluation = Evaluation()
        expected_iou = torch.tensor([[0.8, 0.1], [0.05, 0.2]], dtype=torch.float32)

        with self.install_fake_pytorch3d(expected_iou), patch(
            "util.coordinate_converter.CoordinateConverter.boxes_3d_to_corners",
            side_effect=lambda boxes, mode: boxes.unsqueeze(1).repeat(1, 8, 1),
        ):
            results = evaluation.compute_IoU_3D(detection_sequence)

        assert len(results) == 1
        assert results[0]["sample_id"] == "sample-1"
        torch.testing.assert_close(results[0]["iou_matrix"], expected_iou)

    def test_export_nuscenes_kitti3d_iou_analysis_writes_expected_csv(self, tmp_path):
        detection_sequence = self.build_detection_sequence()
        evaluation = Evaluation()
        serializer = Mock()
        expected_iou = torch.tensor([[0.8, 0.1], [0.05, 0.2]], dtype=torch.float32)
        output_path = tmp_path / "analysis.csv"

        with self.install_fake_pytorch3d(expected_iou), patch(
            "util.coordinate_converter.CoordinateConverter.boxes_3d_to_corners",
            side_effect=lambda boxes, mode: boxes.unsqueeze(1).repeat(1, 8, 1),
        ):
            rows = evaluation.export_nuscenes_kitti3d_iou_analysis(
                detection_sequence,
                output_path,
                serializer=serializer,
                serialize_predictions=True,
            )

        serializer.format_nuScenes_detections.assert_called_once_with(detection_sequence)
        assert len(rows) == 2
        assert rows[0]["sample_id"] == "sample-1"
        assert rows[0]["IoU"] == 0.8
        assert rows[0]["ground_truth_class"] == "Cyclist"
        assert rows[0]["predicted_class"] == "car"
        assert rows[1]["ground_truth_class"] == "Car"
        assert rows[1]["predicted_class"] == "pedestrian"

        with output_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file, delimiter=";")
            written_rows = list(reader)

        assert written_rows[0]["sample_id"] == "sample-1"
        assert written_rows[0]["IoU"] == "0.8"
        assert written_rows[0]["ground_truth_class"] == "Cyclist"
        assert written_rows[0]["predicted_class"] == "car"
        assert written_rows[1]["ground_truth_class"] == "Car"
