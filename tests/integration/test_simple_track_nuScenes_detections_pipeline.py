import unittest

from types import SimpleNamespace
from data_io.deserializer import Deserializer
from definitions import ROOT_DIR
from tracker.SimpleTrack import SimpleTrack
from evaluation.evaluation import Evaluation


class TestSimpleTrackNuScenesDetectionsPipeline(unittest.TestCase):
    def test_read_and_process_dummy_detections_for_nuScenes(self):
        nuScenes_data_root = f"{ROOT_DIR}/tests/data/nuScenes_dummy/"
        settings = SimpleNamespace(
            paths=SimpleNamespace(
                ground_truth_file_path=f"{ROOT_DIR}/tests/data/nuScenes_dummy/v1.0-mini/",
                predicted_tracking_file_path=f"{ROOT_DIR}/src/tracker/SimpleTrack/tracks/nuscenes_tracks.json"
            )
        )
        simple_track = SimpleTrack(output_path=settings.paths.predicted_tracking_file_path)
        detections = Deserializer().deserialize(f"{ROOT_DIR}/tests/data/dets/dummy_detections.json")
        self.assertTrue(len(detections) > 0)
        self.assertIn("bbox_3d", detections[0]["detections"][0])

        tracks = simple_track.track(detections)
        self.assertTrue(len(tracks) > 0)
        self.assertTrue(len(tracks[0]["tracks"]) > 0)
        self.assertEqual(tracks[0]["tracks"][0]["label"], "car")
        self.assertEqual(len(tracks[0]["tracks"][0]["bbox_3d"]), 8)

        evaluation = Evaluation()
        results = evaluation.evaluate_simpletrack_nuscenes_result_file(
            result_path=settings.paths.predicted_tracking_file_path,
            dataroot=nuScenes_data_root,
            output_dir=f"{ROOT_DIR}/tests/data/"
        )
        self.assertTrue(len(results) > 0)

        mota = evaluation.get_metric_value(results, "mota")
        motp = evaluation.get_metric_value(results, "motp")
        precision = evaluation.get_metric_value(results, "precision")
        recall = evaluation.get_metric_value(results, "recall")

        self.assertTrue(mota <= 1.0)
        self.assertTrue(motp >= 0.0)
        self.assertTrue(precision <= 1.0)
        self.assertTrue(recall <= 1.0)


if __name__ == "__main__":
    unittest.main()
