import unittest

from src.datasets.kitti3D import Kitti3D

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
            first_frame["calib"]["P0"][0] != 0,
            "Should be the first value of the key P0"
        )
        self.assertTrue(
            first_frame["image"] is not None, "Image shouldn't be empty"
        )
        self.assertTrue(
            first_frame["points"] is not None, "Points shouldn't be empty"
        )

if __name__ == "__main__":
    unittest.main()