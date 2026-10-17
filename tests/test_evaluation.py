import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation import Evaluation


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.evaluation = Evaluation(iou_threshold=0.5)
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.test_dir = Path(temporary_directory.name)

    def write_mot_file(self, filename, rows):
        path = self.test_dir / filename
        path.write_text(rows, encoding="utf-8")
        return path

    def test_read_mot_file_groups_rows_by_frame(self):
        mot_file = self.write_mot_file(
            "tracks.txt",
            "1,10,1,2,3,4,1,1\n"
            "1,11,5,6,7,8,1,1\n"
            "2,10,9,8,7,6,1,1\n",
        )

        self.assertEqual(self.evaluation.read_mot_file(mot_file), {
            1: [(10, [1, 2, 3, 4]), (11, [5, 6, 7, 8])],
            2: [(10, [9, 8, 7, 6])],
        })

    def test_read_mot_file_filters_confidence_and_class(self):
        mot_file = self.write_mot_file(
            "ground_truth.txt",
            "1,10,1,2,3,4,1,1\n"
            "1,11,5,6,7,8,0,1\n"
            "2,12,9,8,7,6,1,2\n"
            "2,13,2,3,4,5,1,-1\n",
        )

        self.assertEqual(self.evaluation.read_mot_file(
            mot_file,
            filter_ground_truth_by_confidence=True,
            allowed_class_ids={1},
        ), {
            1: [(10, [1, 2, 3, 4])],
            2: [(13, [2, 3, 4, 5])],
        })

    def test_pedestrian_sequence_filter(self):
        for sequence_name, expected in (
            ("MOT17-01", True),
            ("kitti-0001", True),
            ("ETH-Bahnhof", True),
            ("nuScenes-mini", False),
        ):
            with self.subTest(sequence_name=sequence_name):
                self.assertIs(
                    self.evaluation.should_filter_ground_truth_to_pedestrians(sequence_name),
                    expected,
                )

    def test_convert_trackers_to_mot_items_accepts_single_row_and_empty_input(self):
        self.assertEqual(self.evaluation.convert_trackers_to_mot_items([]), [])
        self.assertEqual(
            self.evaluation.convert_trackers_to_mot_items(
                np.array([1, 2, 5, 8, 42], dtype=float)
            ),
            [(42, [1, 2, 4, 6])],
        )

    def test_compute_cumulative_tracking_metrics_for_perfect_match(self):
        accumulator = self.evaluation.create_mot_accumulator()
        ground_truth = {1: [(7, [0, 0, 10, 10])]}
        trackers = np.array([[0, 0, 10, 10, 9]], dtype=float)

        result = self.evaluation.compute_cumulative_tracking_metrics(
            accumulator, 1, ground_truth, trackers
        )

        self.assertAlmostEqual(result["idf1"], 1.0)
        self.assertAlmostEqual(result["mota"], 1.0)
        self.assertAlmostEqual(result["motp"], 0.0)

    def test_evaluate_sequence_counts_unmatched_prediction_frame(self):
        ground_truth_file = self.write_mot_file(
            "ground_truth.txt", "1,7,0,0,10,10,1,1\n"
        )
        prediction_file = self.write_mot_file(
            "predictions.txt",
            "1,9,0,0,10,10,1,1\n"
            "2,9,0,0,10,10,1,1\n",
        )

        summary = self.evaluation.evaluate_sequence(
            ground_truth_file, prediction_file, sequence_name="test_sequence"
        )

        self.assertAlmostEqual(summary.loc["test_sequence", "mota"], 0.0)
        self.assertAlmostEqual(summary.loc["test_sequence", "precision"], 0.5)
        self.assertAlmostEqual(summary.loc["test_sequence", "recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
