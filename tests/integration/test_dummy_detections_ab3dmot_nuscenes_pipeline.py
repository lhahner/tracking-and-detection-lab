import json
import os.path
import unittest

from definitions import ROOT_DIR
from evaluation.evaluation import Evaluation
from tracker.AB3DMOT import AB3DMOT


class TestAB3DMOTNuScenesDetectionsPipeline(unittest.TestCase):
    def test_read_and_process_dummy_detections_for_nuScenes(self):
        nuscenes_data_root = f"{ROOT_DIR}/tests/data/nuScenes_dummy"
        if not os.path.exists(nuscenes_data_root):
            raise unittest.SkipTest("NuScenes Dataset not found skipping test")

        detection_path = f"{ROOT_DIR}/tests/data/dummy_simpletrack_detections.json"
        output_path = f"{ROOT_DIR}/src/tracker/AB3DMOT/tracks/dummy_ab3dmot_tracks.json"
        tracker = AB3DMOT(output_path=output_path, split="val", detector_name="centerpoint")
        tracking_results = tracker.track(detection_path)

        self.assertTrue(len(tracking_results) > 0)
        self.assertTrue(len(tracking_results[0]["tracks"]) > 0)
        self.assertEqual(tracking_results[0]["tracks"][0]["label"], "car")
        self.assertEqual(len(tracking_results[0]["tracks"][0]["bbox_3d"]), 8)

        formatted_detection_file = (
            f"{ROOT_DIR}/src/tracker/AB3DMOT/formatted_detections/centerpoint_Car_val/default.txt"
        )
        self.assertTrue(os.path.isfile(formatted_detection_file))
        self.assertTrue(os.path.isfile(output_path))

        with open(output_path, "r", encoding="utf-8") as result_file:
            payload = json.load(result_file)
        self.assertIn("meta", payload)
        self.assertIn("results", payload)
        self.assertIn(tracking_results[0]["sample_token"], payload["results"])


if __name__ == "__main__":
    unittest.main()
