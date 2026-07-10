import importlib.util
import unittest
import torch
import os
from pathlib import Path
from types import SimpleNamespace
from definitions import ROOT_DIR
from data_io import Serializer

class TestRegnetMMDetection3DNuScenesPipeline(unittest.TestCase):
    def test_prediction_and_serialization_on_nuScenes_mini(self):
        if not torch.cuda.is_available():
            unittest.SkipTest("CUDA GPU required for this test")
        if importlib.util.find_spec("mmdet3d") is None:
            raise unittest.SkipTest("MMDetection3D is not installed")
        if importlib.util.find_spec("nuscenes") is None:
            raise unittest.SkipTest("nuScenes devkit is not installed")

        from inference_engine import InferenceEngine

        root_dir = Path(ROOT_DIR)
        checkpoint_file = root_dir / "src/detector/regnet/model/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d_20200629_050311-dcd4e090.pth"
        dataset_path = root_dir / "tests/data/nuScenes_dummy"
        if not checkpoint_file.exists():
            raise unittest.SkipTest(f"Regnet checkpoint is missing: {checkpoint_file}")
        if not dataset_path.exists():
            raise unittest.SkipTest(f"nuScenes dummy dataset is missing: {dataset_path}")

        settings = SimpleNamespace(
            paths=SimpleNamespace(
                detection_path=str(root_dir / "src/detector/regnet/detections/"),
                dataset_path=str(dataset_path),
                config_file=root_dir / "third_party/mmdetection3d/configs/regnet/pointpillars_hv_regnet-1.6gf_fpn_sbn-all_8xb4-2x_nus-3d.py"
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
            detector_name="regnet_mmdetection3d",
            dataset_path=str(dataset_path),
            detection_path=str(Path(settings.paths.detection_path)),
            model_path=str(checkpoint_file),
        )
        self.assertEqual(predictions.frames[0].frame, dataset.sample_records[0]["sample_token"])
        self.assertIsInstance(predictions.frames[0].dets, list)
        file_name = "regnet-detections-nuScenes-mini-val"
        Serializer(settings=settings,
                   data_format="nuscenes",
                   file_name=file_name).serialize(data=predictions)
        self.assertTrue(os.path.isfile(f"{root_dir}/src/detector/regnet/detections/{file_name}.csv")) 

if __name__ == "__main__":
    unittest.main()
