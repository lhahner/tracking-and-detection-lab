import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from definitions import ROOT_DIR
from inference_engine import InferenceEngine
import torch


class TestVoxelNextOpenpcdetNuScenesPipeline(unittest.TestCase):
    def test_predict_and_evaluate_from_inference_engine_with_nuscenes_mini(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Needs GPU")
        if importlib.util.find_spec("pcdet") is None:
            raise unittest.SkipTest("OpenPcDet is not installed")
        if importlib.util.find_spec("nuscenes") is None:
            raise unittest.SkipTest("nuScenes devkit is not installed")

        root_dir = Path(ROOT_DIR)
        checkpoint_file = root_dir / "src/detector/voxelnext/model/voxelnext_nuscenes_kernel1.pth"
        config_file = root_dir / "third_party/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
        dataset_path = root_dir / "tests/data/nuScenes_dummy"
        if not checkpoint_file.exists():
            raise unittest.SkipTest(f"CenterPoint checkpoint is missing: {checkpoint_file}")
        if not config_file.exists():
            raise unittest.SkipTest(f"CenterPoint config is missing: {config_file}")
        if not dataset_path.exists():
            raise unittest.SkipTest(f"nuScenes dummy dataset is missing: {dataset_path}")
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        checkpoint_keys = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint)).keys()
        if not any(key.startswith(("vfe.", "backbone_3d.", "backbone_2d.", "dense_head.")) for key in checkpoint_keys):
            raise unittest.SkipTest(f"OpenPCDet-compatible SECOND checkpoint is missing: {checkpoint_file}")

        settings = SimpleNamespace(
            paths=SimpleNamespace(
                detection_path=str(root_dir / "output"),
                dataset_path=str(dataset_path),
                config_file=str(config_file),
            ),
            runtime=SimpleNamespace(
                datatype="bin",
                dataset="nuscenes-mini_openpcdet",
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
        self.assertTrue(len(dataset) > 1)

        predictions = inference_engine.predict(
            detector_name="second_openpcdet",
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
