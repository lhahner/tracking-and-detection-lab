import os
import importlib.util
import unittest
import torch

from pathlib import Path
from types import SimpleNamespace
from helpers.helpers import validate_mmdetection3d_integration_environment, load_model
validate_mmdetection3d_integration_environment()

from settings.dummy_settings import generate_nuscenes_mini_settings_with_custom_detector
from definitions import ROOT_DIR
from inference_engine import InferenceEngine
from data_io.serializer import Serializer

url = ("https://download.openmmlab.com/mmdetection3d/v1.0.0_models/"
       "pointpillars/"
       "hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d"
)
mmdet3d_config_folder = f"{ROOT_DIR}/" \
                         "third_party/mmdetection3d/configs/" \
                         "pointpillars/"

class TestPointPillarsMMDetection3DNuScenesPipeline(unittest.TestCase):
    def test_predict_and_evaluate_from_inference_engine_with_nuscenes_mini(self):
        config_file = "pointpillars_hv_fpn_sbn-all_8xb2-amp-2x_nus-3d.py"
        checkpoint_file = "hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
        checkpoint_path = load_model(url=f"{url}/{checkpoint_file}",
                                   checkpoint_file=checkpoint_file
        )
        settings = generate_nuscenes_mini_settings_with_custom_detector(detector_name="pointpillars_mmdetection3d",
                                                                        config_file_path=f"{mmdet3d_config_folder}/{config_file}",
                                                                        checkpoint_path=checkpoint_path
        )
        inference_engine = InferenceEngine(settings=settings)
        dataset = inference_engine.load(split="mini_val", max_samples=100)

        predictions = inference_engine.predict(
            detector_name="pointpillars_mmdetection3d",
            dataset_path=str(settings.paths.dataset_path),
            detection_path=str(Path(settings.paths.detection_path)),
            model_path=str(checkpoint_path),
        )
        self.assertEqual(predictions.frames[0].frame, dataset.sample_records[0]["sample_token"])
        self.assertIsInstance(predictions.frames[0].dets, list)

        results = inference_engine.evaluate_detection(
            detections=predictions,
            classes=settings.benchmark.class_filter,
        )
        self.assertEqual(results[0]["frame"], predictions.frames[0].frame)
        self.assertIn("mAP", results[0])
        self.assertGreater(float(results[0]["mAP"]), 0.0)
        self.assertLessEqual(float(results[0]["mAP"]), 1.0)

    def test_prediction_and_serialization_on_nuScenes_mini(self):
        config_file = "pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py"
        checkpoint_file = "hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
        checkpoint_path = load_model(url=f"{url}/{checkpoint_file}",
                                   checkpoint_file=checkpoint_file
        )
        validate_mmdetection3d_integration_environment()
        settings = generate_nuscenes_mini_settings_with_custom_detector(detector_name="pointpillars",
                                                                        config_file_path=f"{mmdet3d_config_folder}/{config_file}",
                                                                        checkpoint_path=checkpoint_path
        )
        inference_engine = InferenceEngine(settings=settings)
        dataset = inference_engine.load(split="mini_val", max_samples=100000)
        predictions = inference_engine.predict(
            detector_name="pointpillars_mmdetection3d",
            dataset_path=str(settings.paths.dataset_path),
            detection_path=str(Path(settings.paths.detection_path)),
            model_path=str(checkpoint_path),
        )
        self.assertEqual(predictions.frames[0].frame, dataset.sample_records[0]["sample_token"])
        self.assertIsInstance(predictions.frames[0].dets, list)
        Serializer(settings=settings,
                   data_format="nuscenes",
                   file_name="pointpillars-test-dets").serialize(data=predictions)
        self.assertTrue(os.path.isfile(f"{ROOT_DIR}/src/detector/pointpillars/detections/pointpillars-test-dets.csv")) 

if __name__ == "__main__":
    unittest.main()
