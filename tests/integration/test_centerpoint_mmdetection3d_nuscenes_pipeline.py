import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from helpers.helpers import validate_mmdetection3d_integration_environment, load_model
from settings.dummy_settings import generate_nuscenes_mini_settings_with_custom_detector
from definitions import ROOT_DIR
from inference_engine import InferenceEngine
import torch


url = ("https://download.openmmlab.com/mmdetection3d/v1.0.0_models/"
       "centerpoint/"
       "centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus/"
)
mmdet3d_config_folder = f"{ROOT_DIR}/" \
                         "third_party/mmdetection3d/mmdet3d/configs/" \
                         "centerpoint"

class TestCenterPointMMDetection3DNuScenesPipeline(unittest.TestCase):
    def test_predict_and_evaluate_from_inference_engine_with_nuscenes_mini(self):
        config_file = "centerpoint_voxel01_second_secfpn_8xb4_cyclic_20e_nus_3d.py"
        checkpoint_file = "centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth"
        checkpoint_path = load_model(url=f"{url}/{checkpoint_file}",
                                   checkpoint_file=checkpoint_file
        )
        validate_mmdetection3d_integration_environment()
        settings = generate_nuscenes_mini_settings_with_custom_detector(detector_name="centerpoint_mmdetection3d",
                                                                        config_file_path=f"{mmdet3d_config_folder}/{config_file}",
                                                                        checkpoint_path=checkpoint_path
        )
        inference_engine = InferenceEngine(settings=settings)
        dataset = inference_engine.load(split="mini_val", 
                                        max_samples=1,
                                        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(len(dataset), 1)

        predictions = inference_engine.predict(
            detector_name="centerpoint_mmdetection3d",
            dataset_path=str(settings.paths.dataset_path),
            detection_path=str(Path(settings.paths.detection_path)),
            model_path=str(checkpoint_path),
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
