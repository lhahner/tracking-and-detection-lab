import importlib.util
import unittest
import os
from pathlib import Path
from types import SimpleNamespace
from helpers.helpers import validate_mmdetection3d_integration_environment, load_model
from settings.dummy_settings import generate_nuscenes_mini_settings_with_custom_detector
from definitions import ROOT_DIR
validate_mmdetection3d_integration_environment()

from inference_engine import InferenceEngine
from data_io.serializer import Serializer
import torch

url = ("https://download.openmmlab.com/mmdetection3d/v0.1.0_models/"
       "regnet/"
       "hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d"
)
mmdet3d_config_folder = f"{ROOT_DIR}/" \
                         "third_party/mmdetection3d/configs/" \
                         "regnet"

class TestRegnetMMDetection3DNuScenesPipeline(unittest.TestCase):
    def test_prediction_and_serialization_on_nuScenes_mini(self):
        config_file = "pointpillars_hv_regnet-1.6gf_fpn_sbn-all_8xb4-2x_nus-3d.py"
        checkpoint_file = "hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d_20200629_050311-dcd4e090.pth"
        checkpoint_path = load_model(url=f"{url}/{checkpoint_file}",
                                   checkpoint_file=checkpoint_file
        )
        settings = generate_nuscenes_mini_settings_with_custom_detector(detector_name="regnet",
                                                                        config_file_path=f"{mmdet3d_config_folder}/{config_file}",
                                                                        checkpoint_path=checkpoint_path
        )

        inference_engine = InferenceEngine(settings=settings)
        dataset = inference_engine.load(split="mini_val", max_samples=100000)
        predictions = inference_engine.predict(
            detector_name="regnet_mmdetection3d",
            dataset_path=str(settings.paths.dataset_path),
            detection_path=str(Path(settings.paths.detection_path)),
            model_path=str(checkpoint_path),
        )
        self.assertEqual(predictions.frames[0].frame, dataset.sample_records[0]["sample_token"])
        self.assertIsInstance(predictions.frames[0].dets, list)
        file_name = "regnet-detections-nuScenes-mini-val"
        Serializer(settings=settings,
                   data_format="nuscenes",
                   file_name=file_name).serialize(data=predictions)
        self.assertTrue(os.path.isfile(f"{ROOT_DIR}/src/detector/regnet/detections/{file_name}.csv")) 

if __name__ == "__main__":
    unittest.main()
