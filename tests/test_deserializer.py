import json
import unittest

import torch

from entities.detection import Detection
from src.io.deserializer import Deserializer


class TestDeserializer(unittest.TestCase):
    def test_deserialize_detection_from_json(self):
        deserializer = Deserializer()
        serialized = json.dumps({
            "score": 0.91,
            "label": 2,
            "box": [1.0, 2.0, 3.0, 4.0],
        })

        detection = deserializer.deserialize(Detection, serialized)

        self.assertIsInstance(detection, Detection)
        self.assertTrue(torch.equal(detection.score, torch.tensor(0.91)))
        self.assertEqual(detection.label, 2)
        self.assertTrue(torch.equal(detection.box, torch.tensor([1.0, 2.0, 3.0, 4.0])))

    def test_deserialize_rejects_invalid_json(self):
        deserializer = Deserializer()

        with self.assertRaises(json.JSONDecodeError):
            deserializer.deserialize(Detection, "{not valid json")

    def test_deserialize_rejects_unsupported_format(self):
        deserializer = Deserializer(data_format="kitti")
        serialized = json.dumps({
            "score": 0.91,
            "label": 2,
            "box": [1.0, 2.0, 3.0, 4.0],
        })

        with self.assertRaises(NotImplementedError):
            deserializer.deserialize(Detection, serialized)

    def test_deserialize_rejects_unsupported_target_type(self):
        deserializer = Deserializer()

        with self.assertRaises(ValueError):
            deserializer.deserialize(dict, "{}")

    def test_decode_detection_value_builds_detection(self):
        deserializer = Deserializer()
        value = {
            "score": 0.42,
            "label": 1,
            "box": [5.0, 6.0, 7.0],
        }

        detection = deserializer._Deserializer__decode_detection_value(value)

        self.assertIsInstance(detection, Detection)
        self.assertTrue(torch.equal(detection.score, torch.tensor(0.42)))
        self.assertEqual(detection.label, 1)
        self.assertTrue(torch.equal(detection.box, torch.tensor([5.0, 6.0, 7.0])))

    def test_decode_detection_value_rejects_missing_label(self):
        deserializer = Deserializer()
        value = {
            "score": 0.42,
            "box": [5.0, 6.0, 7.0],
        }

        with self.assertRaises(KeyError):
            deserializer._Deserializer__decode_detection_value(value)


if __name__ == "__main__":
    unittest.main()
