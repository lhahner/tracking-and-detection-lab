import os
import sys
import torch
import unittest
from pathlib import Path

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
MMDET3D_SRC_ROOT = os.path.join(PROJECT_ROOT, "external", "mmdetection3d-cpu-only")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if MMDET3D_SRC_ROOT not in sys.path:
    sys.path.insert(0, MMDET3D_SRC_ROOT)

from datasets.kitti3D import Kitti3D, CLASSES
from util.settings_loader import SettingsLoader

KITTI3D_DUMMY_PATH = Path(os.path.join(TESTS_DIR, "data", "kitti3d_dummy"))


class TestPointRCNNMMDetection3DInference(unittest.TestCase):
    def load_mmdet3d_path_from_settings(self):
        settings_loader = SettingsLoader()
        return settings_loader.load().paths.mmdetection3d_path

    def test_non_gpu_test(self):
        kitti3d: Kitti3D = Kitti3D(KITTI3D_DUMMY_PATH)
        if not torch.cuda.is_available():
            with self.assertRaises(EnvironmentError):
                from detector.pointrcnn.pointrcnn_mmdetection3d import PointRCNNmmDetections3D
                pointrcnn: PointRCNNmmDetections3D = PointRCNNmmDetections3D(
                        dataset=kitti3d,
                        config_file=f"{self.load_mmdet3d_path_from_settings()}/point-rcnn_8xb2_kitti-3d-3class.py",
                        classes=CLASSES)
        else:
            self.skipTest("Test test requires CPU")

    def test_kitti3d_dummy_detection(self):
        if torch.cuda.is_available():
            from detector.pointrcnn.pointrcnn_mmdetection3d import PointRCNNmmDetections3D

            kitti3d: Kitti3D = Kitti3D(KITTI3D_DUMMY_PATH)
            pointrcnn: PointRCNNmmDetections3D = PointRCNNmmDetections3D(
                dataset=kitti3d,
                config_file=f"{self.load_mmdet3d_path_from_settings()}/point-rcnn_8xb2_kitti-3d-3class.py",
                classes=CLASSES)
            out: list = pointrcnn.detect()
            self.assetTrue(len(out) > 0)
        else:
            self.skipTest("This requires GPU")  # Smoke Test
