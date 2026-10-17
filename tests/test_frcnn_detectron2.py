import os
import sys
import types
import unittest
import numpy as np

from unittest.mock import MagicMock, patch
from detector.frcnn.frcnn_detectron2 import FasterRCNNDetectron2
from entities.detection import DetectionSequence

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class TestFasterRCNNDetectron2(unittest.TestCase):
    def test_detect_returns_correct_dto_object(self):
        frcnn = FasterRCNNDetectron2(input_path=TESTS_DIR + "/data/kitti3d_dummy/training/image_2/")
        detection_sequence = frcnn.detect()
        self.assertTrue(isinstance(detection_sequence, DetectionSequence))