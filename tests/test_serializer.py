import json
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch
import os
import torch
import glob
from entities.detection import Detection, FrameDetection, DetectionSequence
from data_io.serializer import Serializer


@dataclass(frozen=True)
class SerializableSample:
    frame: int
    scores: torch.Tensor


@dataclass(frozen=True)
class KittiDetection:
    score: float
    label: str
    box: torch.Tensor


@dataclass(frozen=True)
class KittiFrame:
    frame: int
    highest_score_index: int
    dets: list[KittiDetection]

class TestSerializer(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(
                    paths = SimpleNamespace(
                        detection_path = "tests/data/dets/"
            )
        )

    def buildDetectionSequenceAndSerializer(self):
        serializer = Serializer(self.settings, data_format="kitti", file_name="detections")
        detection = Detection(
            score=torch.tensor(0.88),
            label="Car",
            box=torch.tensor([1.234, 2.346, 3.456, 4.0, 5.0, 6.0, 7.0, 0.123]),
        )
        detection_sequence = DetectionSequence(
                frames=[FrameDetection(frame=0, highest_score_index=0, dets=[detection])])
        return serializer, detection_sequence


    def test_serialize_json_encodes_dataclass_and_tensor(self):
        serializer = Serializer(settings=self.settings, file_name="detections")
        data = SerializableSample(frame=3, scores=torch.tensor([0.1, 0.9], dtype=torch.float64))

        result = serializer.serialize(data)
        
        self.assertTrue(len(glob.glob(self.settings.paths.detection_path + "*.json")) > 0)
        self.assertEqual(json.loads(result), {"frame": 3, "scores": [0.1, 0.9]})

    def test_serialize_json_rejects_unknown_object(self):
        serializer = Serializer(settings=self.settings)

        with self.assertRaises(ValueError):
            serializer.serialize(object())

    @patch("data_io.serializer.write_output")
    def test_serialize_kitti_formats_detection_sequence(self, mock_write_output):
        serializer, detection_sequence = self.buildDetectionSequenceAndSerializer()
        serializer.serialize(detection_sequence)

        mock_write_output.assert_called_once_with(
            "tests/data/dets/detections.txt",
            "Car 0 -1 0 0 0 0 0 6.0 5.0 7.0 1.23 2.35 3.46 0.12 0.88",
        )

    def test_serialize_unknown_format_returns_none(self):
        serializer = Serializer(settings=self.settings, data_format="xml")
        self.assertIsNone(serializer.serialize({"frame": 1}))


    @patch("data_io.serializer.write_output")
    def test_format_kitti3d_detections_writes_expected_line(self, mock_write_output):
        serializer, detection_sequence = self.buildDetectionSequenceAndSerializer()
        serializer.format_kitti3d_detections(detection_sequence)

        mock_write_output.assert_called_once_with(
            "tests/data/dets/detections.txt",
            "Car 0 -1 0 0 0 0 0 6.0 5.0 7.0 1.23 2.35 3.46 0.12 0.88",
        )

    def test_format_kitti3d_detections_rejects_invalid_box_dimensions(self):
        serializer = Serializer(self.settings, data_format="kitti", file_name="detections")
        
        detection = Detection(
            score=torch.tensor(0.88),
            label="Car",
            box=torch.tensor([1.234, 2.346]),
        )
        detection_sequence = DetectionSequence(
                frames=[FrameDetection(frame=0, highest_score_index=0, dets=[detection])])

        with self.assertRaises(IndexError):
            serializer.format_kitti3d_detections(detection_sequence)

    def test_format_nuscenes_detections_returns_expected_csv(self):
        serializer = Serializer(self.settings, data_format="nuscenes", file_name="detections")
        detection = Detection(
            score=torch.tensor(0.91),
            label=4,
            box=torch.tensor([1.234, 2.345, 3.456, 4.567, 1.891, 1.678, 0.123]),
        )
        detection_sequence = DetectionSequence(
            frames=[FrameDetection(frame="sample-token-123", highest_score_index=0, dets=[detection])]
        )

        result = serializer.format_nuScenes_detections(detection_sequence)

        self.assertEqual(
            result,
            "sample_token,detection_name,detection_score,x,y,z,length,width,height,yaw,velocity_x,velocity_y,attribute_name\r\n"
            "sample-token-123,car,0.91,1.23,2.35,3.46,4.57,1.89,1.68,0.12,0.0,0.0,\r\n",
        )

    def test_build_kitti_gt_string_formats_values(self):
        serializer = Serializer(settings=self.settings)

        result = serializer._Serializer__build_kitti_gt_string(
            obj_type="Car",
            truncated=0,
            occluded=-1,
            alpha=0,
            bbox_2d=torch.tensor([1.111, 2.222, 3.333, 4.444]),
            dimensions=torch.tensor([1.5, 2.5, 3.5]),
            location=torch.tensor([4.444, 5.556, 6.666]),
            rotation_y=torch.tensor(0.987),
            score=torch.tensor(0.654),
        )

        self.assertEqual(result, 
                         "Car 0 -1 0 1.11 2.22 3.33 4.44 1.5 2.5 3.5 4.44 5.56 6.67 0.99 0.65")

    def test_build_kitti_gt_string_rejects_location_without_numpy(self):
        serializer = Serializer(settings=self.settings)

        with self.assertRaises(AttributeError):
            serializer._Serializer__build_kitti_gt_string(
                obj_type="Car",
                truncated=0,
                occluded=-1,
                alpha=0,
                bbox_2d=torch.tensor([0.0, 0.0, 0.0, 0.0]),
                dimensions=torch.tensor([1.0, 2.0, 3.0]),
                location=[1.0, 2.0, 3.0],
                rotation_y=torch.tensor(0.1),
                score=torch.tensor(0.2),
            )


if __name__ == "__main__":
    unittest.main()
