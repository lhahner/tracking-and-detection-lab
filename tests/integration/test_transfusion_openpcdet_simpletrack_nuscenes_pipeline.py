import json
import unittest
import os
import torch

from pathlib import Path
from helpers.helpers import validate_openpcdet_integration_environment
validate_openpcdet_integration_environment()

from inference_engine import InferenceEngine
from settings.dummy_settings import generate_nuscenes_mini_settings_with_custom_detector_and_custom_tracker
from definitions import ROOT_DIR
from tracker.SimpleTrack import SimpleTrack
from evaluation.evaluation import Evaluation
from data_io import Deserializer
from data_io import Serializer
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


class TestTransfusionOpenpcdetSimpleTrack(unittest.TestCase):
    def test_predict_and_evaluate_from_inference_engine_with_nuscenes_mini(self):
        checkpoint_file = f"{ROOT_DIR}/src/detector/transfusion/model/cbgs_transfusion_lidar.pth"
        config_file = f"{ROOT_DIR}/third_party/OpenPCDet/tools/cfgs/nuscenes_models/" \
                       "transfusion_lidar.yaml"
        dataset_path = f"{ROOT_DIR}/tests/data/nuScenes_dummy"
        
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        checkpoint_keys = checkpoint.get("model_state", 
                                         checkpoint.get("state_dict", checkpoint)).keys()
        if not any(key.startswith(("vfe.", 
                                   "backbone_3d.", 
                                   "backbone_2d.", 
                                   "dense_head.")) for key in checkpoint_keys):
            raise unittest.SkipTest(f"OpenPCDet-compatible TransFusion checkpoint is missing: "
                                     f"{checkpoint_file}")
        
        settings = generate_nuscenes_mini_settings_with_custom_detector_and_custom_tracker(detector_name="transfusion",
                                                                                           config_file_path=config_file,
                                                                                           checkpoint_path=checkpoint_file,
                                                                                           tracker_name="SimpleTrack",
                                                                                           dataset_name="nuscenes-mini_openpcdet")
        settings.runtime.dataset = "nuscenes-mini_openpcdet"
        inference_engine = InferenceEngine(settings=settings)
        dataset = inference_engine.load(split="mini_val", 
                                        max_samples=1,
                                        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertTrue(len(dataset) > 1)

        predictions = inference_engine.predict(
            detector_name="transfusion_openpcdet",
            dataset_path=str(dataset_path),
            detection_path=str(Path(settings.paths.detection_path)),
            model_path=str(checkpoint_file),
        )

        self.assertEqual(len(predictions.frames), 1)
        self.assertEqual(predictions.frames[0].frame, dataset.sample_records[0]["sample_token"])
        self.assertIsInstance(predictions.frames[0].dets, list)
        
        # Detect and Serialize
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
                                file_name=f"{settings.paths.detection_path}detections.json")
        serializer.serialize(predictions)
        self.assertTrue(os.path.exists(settings.paths.detection_path))
        
        # Deserialize and Track
        tracker = SimpleTrack(
                output_path=settings.paths.tracking_path
        )
        deserializer = Deserializer(data_format="simple_track")
        deserialized_detections = deserializer.deserialize(document_path=f"{settings.paths.detection_path}detections.json")
        tracking_results = tracker.track(detections=deserialized_detections)
        self.assertTrue(len(tracking_results) > 0)
        self.assertTrue(os.path.exists(settings.paths.tracking_path))
        
        self.__pad_nuscenes_results_for_split(
            result_path=settings.paths.tracking_path,
            dataroot=settings.paths.dataset_path,
            version="v1.0-mini",
            split="mini_val",
        )
        
        # Evaluate Tracking
        evaluation = Evaluation()
        first_sample_results = evaluation.evaluate_simpletrack_nuscenes_sample_tokens(
            result_path=settings.paths.tracking_path,
            dataroot=settings.paths.dataset_path,
            sample_tokens=[predictions.frames[0].frame],
        )
        first_sample_mota = first_sample_results["mota"]
        self.assertGreater(first_sample_results["num_matches"], 0)
        self.assertGreater(first_sample_results["recall"], 0.0)
        self.assertLessEqual(first_sample_mota, 1.0)

        results_tracker = evaluation.evaluate_simpletrack_nuscenes_result_file(
            result_path=settings.paths.tracking_path,
            dataroot=settings.paths.dataset_path,
            output_dir=f"{ROOT_DIR}/tests/data/"
        )
        self.assertTrue(len(results_tracker) > 0)
        mota = results_tracker["mota"]
        motp = results_tracker["motp"]
        recall = results_tracker["recall"]
        self.assertTrue(mota <= 1.0)
        self.assertTrue(motp >= 0.0)
        self.assertTrue(recall <= 1.0)

    def __pad_nuscenes_results_for_split(self, result_path, dataroot, version, split):
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


if __name__ == "__main__":
    unittest.main()
