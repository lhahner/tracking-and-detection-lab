import unittest
from datasets.kitti3D import Kitti3D
from detector.pointpillars.pointpillars import Pointpillars
from unittest.mock import Mock, patch
from entities.detection import DetectionSequence, FrameDetection, Detection

class TetsPointPillars(unittest.TestCase):
    def test_pointpillars_basic_inference_works_and_result_are_not_empty(self):
        dataset_dummy = Kitti3D(data_root="tests/data/kitti3d_dummy/") 
        pointpillars = Pointpillars(
                dataset=dataset_dummy,
                config_file=Mock(),
                classes={
                    'Pedestrian': 0, 
                    'Cyclist': 1, 
                    'Car': 2
                },
                settings=Mock(),
                batch_size=1,
                num_inference_samples=1,
                device='cpu',
                checkpoint_file="third_party/pointpillars/_ext_src/pretrained/epoch_160.pth"
                )
        detection_sequence = pointpillars.detect()
        self.assertTrue(isinstance(detection_sequence, DetectionSequence))
        self.assertTrue(len(detection_sequence.frames) > 0)
