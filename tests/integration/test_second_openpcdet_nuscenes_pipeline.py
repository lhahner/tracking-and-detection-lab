import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from helpers.helpers import validate_openpcdet_integration_environment, load_model
from settings.dummy_settings import generate_nuscenes_mini_settings_with_custom_detector
from definitions import ROOT_DIR
validate_openpcdet_integration_environment()

from inference_engine import InferenceEngine
import torch

class TestSecondOpenpcdetNuScenesPipeline(unittest.TestCase):
    def test_predict_and_evaluate_from_inference_engine_with_nuscenes_mini(self):
        checkpoint_file = f"{ROOT_DIR}/src/detector/second" \
                            "/model/cbgs_second_multihead_nds6229_updated.pth"
        config_file = f"{ROOT_DIR}/third_party/OpenPCDet/tools/cfgs/nuscenes_models/" \
                       "cbgs_second_multihead.yaml"
        dataset_path = f"{ROOT_DIR}/tests/data/nuScenes_dummy"
        
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        checkpoint_keys = checkpoint.get("model_state", 
                                         checkpoint.get("state_dict", checkpoint)).keys()
        if not any(key.startswith(("vfe.", 
                                   "backbone_3d.", 
                                   "backbone_2d.", 
                                   "dense_head.")) for key in checkpoint_keys):
            raise unittest.SkipTest(f"OpenPCDet-compatible SECOND checkpoint is missing:" \
                                     "{checkpoint_file}")
        
        settings = generate_nuscenes_mini_settings_with_custom_detector(detector_name="regnet",
                                                                        config_file_path=config_file,
                                                                        checkpoint_path=checkpoint_file,
                                                                        dataset_name="nuscenes-mini_openpcdet"
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
