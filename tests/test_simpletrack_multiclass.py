import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


try:
    simpletrack_module = importlib.import_module("tracker.SimpleTrack.SimpleTrack")
except ModuleNotFoundError as exc:
    simpletrack_module = None
    SIMPLETRACK_IMPORT_ERROR = exc
else:
    SIMPLETRACK_IMPORT_ERROR = None


class FakeMOTModel:
    instances = []

    def __init__(self, config):
        self.config = config
        self.calls = []
        FakeMOTModel.instances.append(self)

    def frame_mot(self, frame_data):
        self.calls.append(list(frame_data.det_types))
        if not frame_data.dets:
            return []

        labels = set(frame_data.det_types)
        assert len(labels) == 1
        det_type = frame_data.det_types[0]
        return [(frame_data.dets[0], 0, "alive", det_type)]


@unittest.skipIf(
    simpletrack_module is None,
    f"SimpleTrack runtime dependencies are not installed: {SIMPLETRACK_IMPORT_ERROR}",
)
class TestSimpleTrackMultiClass(unittest.TestCase):
    def test_uses_one_tracker_per_label_and_unique_public_ids(self):
        FakeMOTModel.instances = []

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            simpletrack_module, "MOTModel", FakeMOTModel
        ):
            tmp_path = Path(tmpdir)
            config_path = tmp_path / "config.yaml"
            config_path.write_text("running: {}\n", encoding="utf-8")
            output_path = tmp_path / "tracks.json"

            detections = [
                {
                    "frame": 1,
                    "sample_token": "sample-1",
                    "lidar_to_global": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "detections": [
                        {"label": "car", "bbox_3d": [1, 2, 3, 0, 4, 5, 6], "score": 0.9},
                        {"label": "pedestrian", "bbox_3d": [2, 3, 4, 0, 1, 1, 2], "score": 0.8},
                    ],
                }
            ]

            tracks = simpletrack_module.SimpleTrack(
                config_path=config_path, output_path=output_path
            ).track(detections)

        self.assertEqual(len(FakeMOTModel.instances), 2)
        self.assertEqual(
            sorted(instance.calls for instance in FakeMOTModel.instances),
            [[['car']], [['pedestrian']]],
        )

        frame_tracks = tracks[0]["tracks"]
        self.assertEqual({track["label"] for track in frame_tracks}, {"car", "pedestrian"})
        self.assertEqual(len({track["track_id"] for track in frame_tracks}), 2)


if __name__ == "__main__":
    unittest.main()
