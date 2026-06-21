import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from definitions import ROOT_DIR
from inference_engine import InferenceEngine


class TestSecondOpenpcdetNuScenesPipeline(unittest.TestCase):
    def test_predict_and_evaluate_from_inference_engine_with_nuscenes_mini(self):
        if importlib.util.find_spec("pcdet") is None:
            raise unittest.SkipTest("OpenPcDet is not installed")
        if importlib.util.find_spec("nuscenes") is None:
            raise unittest.SkipTest("nuScenes devkit is not installed")

        root_dir = Path(ROOT_DIR)
        checkpoint_file = root_dir / "src/detector/centerpoint/model/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth"
        config_file = root_dir / "src/detector/centerpoint/model/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
        dataset_path = root_dir / "tests/data/nuScenes_dummy"
        if not checkpoint_file.exists():
            raise unittest.SkipTest(f"CenterPoint checkpoint is missing: {checkpoint_file}")
        if not config_file.exists():
            raise unittest.SkipTest(f"CenterPoint config is missing: {config_file}")
        if not dataset_path.exists():
            raise unittest.SkipTest(f"nuScenes dummy dataset is missing: {dataset_path}")

        settings = SimpleNamespace(
            paths=SimpleNamespace(
                detection_path=str(root_dir / "output"),
                dataset_path=str(dataset_path),
                config_file=str(config_file),
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
        dataset = inference_engine.load(split="mini_val", 
                                        max_samples=1,
                                        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(len(dataset), 1)

        predictions = inference_engine.predict(
            detector_name="centerpoint_mmdetection3d",
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


if __name__ == "__main__":
    unittest.main()
