import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from definitions import ROOT_DIR

from settings.dummy_settings import generate_kitti3d_settings_with_custom_detector
from helpers.helpers import validate_mmdetection3d_integration_environment, load_model
validate_mmdetection3d_integration_environment()

from inference_engine import InferenceEngine
import torch

url = ("https://download.openmmlab.com/mmdetection3d/v0.1.0_models/"
       "point_rcnn/"
)
mmdet3d_config_folder = f"{ROOT_DIR}/" \
                         "third_party/mmdetection3d/configs/" \
                         "point_rcnn" 
                         

class TestPointRCNNMMDetection3DKitti3DPipeline(unittest.TestCase):
    def test_predict_and_evaluate_from_inference_engine_with_kitti3d(self):
        checkpoint_file = "point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth"
        config_file = "point-rcnn_8xb2_kitti-3d-3class.py"
        checkpoint_path = load_model(url=f"{url}/{checkpoint_file}",
                                     checkpoint_file=checkpoint_file)
        settings = generate_kitti3d_settings_with_custom_detector(detector_name="pointrcnn",
                                                                     config_file_path=f"{mmdet3d_config_folder}/{config_file}",
                                                                     checkpoint_path=checkpoint_path
        )
        inference_engine = InferenceEngine(settings=settings)
        dataset = inference_engine.load(split="val", 
                                        max_samples=1,
                                        labels={
                                            "Car": 3,
                                            "Pedestrian": 1,
                                            "Cyclist": 2,
                                        })
        self.assertEqual(len(dataset), 1)
        predictions = inference_engine.predict(
            detector_name="pointrcnn_mmdetection3d",
            dataset_path=str(settings.paths.dataset_path),
            detection_path=str(Path(settings.paths.detection_path)),
            model_path=str(settings.paths.checkpoint_path),
        )
        self.assertEqual(len(predictions.frames), 1)
        self.assertIsInstance(predictions.frames[0].dets, list)

        results = inference_engine.evaluate_detection(
            detections=predictions,
            classes=[3, 1, 2],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["frame"], predictions.frames[0].frame)


if __name__ == "__main__":
    unittest.main()
