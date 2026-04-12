import os
import sys
import unittest
from unittest.mock import MagicMock, patch

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from datasets.kitti3D import Kitti3D

BASE_DIR = "/media/lennart/LaCie/gau/advanced-research-training-applied-system-development/datasets/kitti_3D_object_detection"

class Kitti3DTest(unittest.TestCase):
    def test_dataset_loaded_and_get_item(self):
        kitti3D = Kitti3D(BASE_DIR, split="train")
        first_frame = kitti3D[1]
        
        self.assertEqual(
            first_frame["sample_id"], "000001",
            "Should be the first id of the frame"
        )
        self.assertEqual(
            first_frame["target"][0]["type"], "Truck", 
            "Should be truck as 000001.txt is Truck"
        )
        self.assertTrue(
            first_frame["calib"]["P0"][0][0] != 0,
            "Should be the first value of the key P0"
        )
        self.assertTrue(
            first_frame["image"] is not None, "Image shouldn't be empty"
        )
        self.assertTrue(
            first_frame["points"] is not None, "Points shouldn't be empty"
        )
    
    @patch("datasets.kitti3D.Kitti3D._load_label")
    def test__load_label(self, mock_load_label):
      mock_load_label.return_value = ("1", "/fake/path")
      kitti3D = Kitti3D(BASE_DIR, split="train")
      tmp = kitti3D._load_label(1)
      assert tmp == ("1", "/fake/path")
    
    @patch("datasets.kitti3D.Kitti3D._load_label")
    @patch("datasets.kitti3D.Kitti3D.filter_supported_objects")
    def test__build_object_index(self, mock_filter_supported_objects, mock_load_label):
        objects = [
            {
                "type": "Truck",
                "truncated": 0.00,
                "occluded": 0,
                "alpha": -1.57,
                "bbox": [599.41, 156.40, 629.75, 189.25],
                "dimensions": [2.85, 2.63, 12.34],
                "location": [0.47, 1.49, 69.44],
                "rotation_y": -1.56,
            },
            {
                "type": "Car",
                "truncated": 0.00,
                "occluded": 0,
                "alpha": -1.57,
                "bbox": [599.41, 156.40, 629.75, 189.25],
                "dimensions": [2.85, 2.63, 12.34],
                "location": [0.47, 1.49, 69.44],
                "rotation_y": -1.56,
            },
            {
                "type": "DontCare",
                "truncated": 0.00,
                "occluded": 0,
                "alpha": -1.57,
                "bbox": [599.41, 156.40, 629.75, 189.25],
                "dimensions": [2.85, 2.63, 12.34],
                "location": [0.47, 1.49, 69.44],
                "rotation_y": -1.56,
            }
        ]
        mock_load_label.return_value = (objects, "/fake/path/000000.txt")
        mock_filter_supported_objects.return_value = [
            (0, objects[0]),
            (1, objects[1]),
        ]
        
        kitti3D = Kitti3D(BASE_DIR, split="train")
        kitti3D.sample_ids = ["000000", "000001", "000002"]
        object_index = kitti3D._build_object_index()
        
        self.assertEqual(
            object_index,
            [
                ("000000", 0),
                ("000000", 1),
                ("000001", 0),
                ("000001", 1),
                ("000002", 0),
                ("000002", 1),
            ],
        )
        self.assertEqual(mock_load_label.call_count, 3)
        self.assertEqual(mock_filter_supported_objects.call_count, 3)
        
        
        
if __name__ == "__main__":
    unittest.main()
