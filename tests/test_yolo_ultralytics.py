import os
import sys
import types
import unittest
import numpy as np

from unittest.mock import MagicMock, patch
from detector.yolo.yolo_ultralytics import YoloUltralytics
from entities.detection import DetectionSequence

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class TestYoloUltralyticsDetector(unittest.TestCase):
    def test_detect_returns_correct_dto_object(self):
        yolo = YoloUltralytics(input_path=TESTS_DIR + "/data/kitti3d_dummy/training/image_2/",
                               model="yolo11n")
        detection_sequence = yolo.detect()
        self.assertTrue(isinstance(detection_sequence, DetectionSequence))
         