import importlib.util
import json
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from definitions import ROOT_DIR
from tracker.SimpleTrack import SimpleTrack
from evaluation.evaluation import Evaluation
from data_io import Deserializer
from data_io import Serializer
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import torch


class TestPointPillarsSimpleTrack(unittest.TestCase):
    def test_predict_and_evaluate_from_inference_engine_with_nuscenes_mini(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("GPU not there")
        if importlib.util.find_spec("mmdet3d") is None:
            raise unittest.SkipTest("MMDetection3D is not installed")
        if importlib.util.find_spec("nuscenes") is None:
            raise unittest.SkipTest("nuScenes devkit is not installed")

        from inference_engine import InferenceEngine

        root_dir = Path(ROOT_DIR)
        checkpoint_file = root_dir / "src/detector/pointpillars/model/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
        dataset_path = root_dir / "tests/data/nuScenes_dummy"
        nuScenes_data_root = str(dataset_path)
        if not checkpoint_file.exists():
            raise unittest.SkipTest(f"PointPillars checkpoint is missing: {checkpoint_file}")
        if not dataset_path.exists():
            raise unittest.SkipTest(f"nuScenes dummy dataset is missing: {dataset_path}")

        settings = SimpleNamespace(
            paths=SimpleNamespace(
                detection_path=str(root_dir / "output"),
                dataset_path=str(dataset_path),
                config_file=root_dir / "third_party/mmdetection3d/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py",
                output_path_detections=root_dir / "src/detector/pointpillars/dets/pointpillars_simpletrack.json",
                output_path_tracks=root_dir / "src/tracker/tracks/pointpillars_simpletrack.json"
            ),
            runtime=SimpleNamespace(
                datatype="bin",
                dataset="nuscenes-mini",
                display=False,
            ),
            benchmark=SimpleNamespace(
                iou_threshold=0.4,
                class_filter=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ),
            tracker=SimpleNamespace(max_age=3, min_hits=2, iou_threshold=0.2),
            dataset=SimpleNamespace(
                classes=[
                    "barrier",
                    "bicycle",
                    "bus",
                    "car",
                    "construction_vehicle",
                    "motorcycle",
                    "pedestrian",
                    "traffic_cone",
                    "trailer",
                    "truck",
                ]
            ),
        )

        inference_engine = InferenceEngine(settings=settings)
        dataset = inference_engine.load(split="mini_val", max_samples=1)
        self.assertEqual(len(dataset), 1)

        predictions = inference_engine.predict(
            detector_name="pointpillars_mmdetection3d",
            dataset_path=str(dataset_path),
            detection_path=str(Path(settings.paths.detection_path)),
            model_path=str(checkpoint_file),
        )
        self.assertEqual(len(predictions.frames), 1)
        self.assertEqual(predictions.frames[0].frame, dataset.sample_records[0]["sample_token"])
        self.assertIsInstance(predictions.frames[0].dets, list)
        
        results = inference_engine.evaluate_detection(
            detections=predictions,
            classes=settings.benchmark.class_filter,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["frame"], predictions.frames[0].frame)
        self.assertIn("mAP", results[0])
        self.assertGreater(float(results[0]["mAP"]), 0.0)
        self.assertLessEqual(float(results[0]["mAP"]), 1.0)

        serializer = Serializer(settings=settings,
                                data_format="simple_track",
                                file_name=settings.paths.output_path_detections)
        serializer.serialize(predictions)
        self.assertTrue(os.path.exists(settings.paths.output_path_detections))
        tracker = SimpleTrack(
                output_path=settings.paths.output_path_tracks
        )
        deserializer = Deserializer(data_format="simple_track")
        deserialized_detections = deserializer.deserialize(document_path=settings.paths.output_path_detections)
        tracking_results = tracker.track(detections=deserialized_detections)
        self.assertTrue(len(tracking_results) > 0)
        self.assertTrue(os.path.exists(settings.paths.output_path_tracks))
        
        self._pad_nuscenes_results_for_split(
            result_path=settings.paths.output_path_tracks,
            dataroot=nuScenes_data_root,
            version="v1.0-mini",
            split="mini_val",
        )

        evaluation = Evaluation()
        first_sample_results = evaluation.evaluate_simpletrack_nuscenes_sample_tokens(
            result_path=settings.paths.output_path_tracks,
            dataroot=nuScenes_data_root,
            sample_tokens=[predictions.frames[0].frame],
        )
        first_sample_mota = first_sample_results["mota"]
        self.assertGreater(first_sample_results["num_matches"], 0)
        self.assertGreater(first_sample_results["recall"], 0.0)
        self.assertLessEqual(first_sample_mota, 1.0)

        results_tracker = evaluation.evaluate_simpletrack_nuscenes_result_file(
            result_path=settings.paths.output_path_tracks,
            dataroot=nuScenes_data_root,
            output_dir=f"{ROOT_DIR}/tests/data/"
        )
        self.assertTrue(len(results_tracker) > 0)
        mota = results_tracker["mota"]
        motp = results_tracker["motp"]
        recall = results_tracker["recall"]
        self.assertTrue(mota <= 1.0)
        self.assertTrue(motp >= 0.0)
        self.assertTrue(recall <= 1.0)        
       

    def _pad_nuscenes_results_for_split(self, result_path, dataroot, version, split):
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        split_scene_names = set(create_splits_scenes()[split])
        split_sample_tokens = [
            sample["token"]
            for sample in nusc.sample
            if nusc.get("scene", sample["scene_token"])["name"] in split_scene_names
        ]

        result_path = Path(result_path)
        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        result_payload.setdefault("results", {})
        for sample_token in split_sample_tokens:
            result_payload["results"].setdefault(sample_token, [])
        result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    def test_prediction_and_serialization_on_nuScenes_mini(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA GPU required for this test")
        if importlib.util.find_spec("mmdet3d") is None:
            raise unittest.SkipTest("MMDetection3D is not installed")
        if importlib.util.find_spec("nuscenes") is None:
            raise unittest.SkipTest("nuScenes devkit is not installed")

        from inference_engine import InferenceEngine

        root_dir = Path(ROOT_DIR)
        checkpoint_file = root_dir / "src/detector/pointpillars/model/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
        dataset_path = root_dir / "tests/data/nuScenes_dummy"
        if not checkpoint_file.exists():
            raise unittest.SkipTest(f"PointPillars checkpoint is missing: {checkpoint_file}")
        if not dataset_path.exists():
            raise unittest.SkipTest(f"nuScenes dummy dataset is missing: {dataset_path}")

        settings = SimpleNamespace(
            paths=SimpleNamespace(
                detection_path=str(root_dir / "src/detector/pointpillars/detections/"),
                dataset_path=str(dataset_path),
                config_file=root_dir / "third_party/mmdetection3d/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
            ),
            runtime=SimpleNamespace(
                datatype="bin",
                dataset="nuscenes-mini",
                display=False,
            ),
            benchmark=SimpleNamespace(
                iou_threshold=0.4,
                class_filter=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ),
            tracker=SimpleNamespace(max_age=3, min_hits=2, iou_threshold=0.2),
            dataset=SimpleNamespace(
                classes=[
                    "barrier",
                    "bicycle",
                    "bus",
                    "car",
                    "construction_vehicle",
                    "motorcycle",
                    "pedestrian",
                    "traffic_cone",
                    "trailer",
                    "truck",
                ]
            ),
        )

        inference_engine = InferenceEngine(settings=settings)
        dataset = inference_engine.load(split="mini_val", max_samples=100000)
        predictions = inference_engine.predict(
            detector_name="pointpillars_mmdetection3d",
            dataset_path=str(dataset_path),
            detection_path=str(Path(settings.paths.detection_path)),
            model_path=str(checkpoint_file),
        )
        self.assertEqual(predictions.frames[0].frame, dataset.sample_records[0]["sample_token"])
        self.assertIsInstance(predictions.frames[0].dets, list)
        Serializer(settings=settings,
                   data_format="nuscenes",
                   file_name="pointpillars-test-dets").serialize(data=predictions)
        self.assertTrue(os.path.isfile(f"{root_dir}/src/detector/pointpillars/detections/pointpillars-test-dets.csv")) 

if __name__ == "__main__":
    unittest.main()
