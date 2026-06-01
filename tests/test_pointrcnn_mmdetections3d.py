import os
import sys
import unittest
import torch
from unittest.mock import Mock, patch, create_autospec
import numpy as np

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class TestPointRCNNmmDetections3D(unittest.TestCase):
    @patch("datasets.kitti3D.Kitti3D")
    def test_format_kitti3d_detections(self, mock_kitti3D):
        if not torch.cuda.is_available():
            self.skipTest("These tests need GPU support")
        from detector.pointrcnn.pointrcnn_mmdetection3d import PointRCNNmmDetections3D
        fake_model = Mock()
        with patch("detector.pointrcnn.pointrcnn_mmdetection3d.init_model", return_value=fake_model):
            pointrcnnmmdetections3D = PointRCNNmmDetections3D(
                    dataset=mock_kitti3D,
                    config_file=f"{SRC_ROOT}/detector/pointrcnn/model/point-rcnn_2x8_kitti-3d-3class_legacyckpt.py",
                    checkpoint_file=f"{SRC_ROOT}/detector/pointrcnn/model/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth",
                    classes={0: "Car", 1: "Pedestrian"})
        # object_type,truncation,occlusion,alpha,left,top,right,bottom,height,width,length,x,y,z,rotation_y 
        # Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
        bboxes = torch.tensor([
            [0.1, 0.2, 0.3, 0.5, 0.5, 0.5, 0.1],
            ]).squeeze(0)

        xyz_centroids = bboxes[:][:3]
        lwh_box = bboxes[:][3:6]
        yaw = bboxes[:][6]
        det_scroes = np.array([0.78])
        formatted_detections = pointrcnnmmdetections3D.format_kitti3d_detections(
                xyz_centroids=xyz_centroids,
                lwh_box=lwh_box,
                yaw=yaw,
                det_score=det_scroes,
                obj_index=0
                )
        kitti_string = f"Car 0 -1 0 0 0 0 0 0.5 0.5 0.5 0.1 0.2 0.3 0.1 0.78"
        self.assertTrue(len(formatted_detections) > 0)
        self.assertEquals(kitti_string, formatted_detections)
