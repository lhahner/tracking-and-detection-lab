import importlib.util
import unittest
import torch
import os
from pathlib import Path
from types import SimpleNamespace
from definitions import ROOT_DIR
from data_io import Serializer
from entities.detection import Detection
from geometry.coordinate_converter import CoordinateConverter

class TestPGDMMDetection3DNuScenesPipeline(unittest.TestCase):
    def test_prediction_and_serialization_on_nuScenes_mini(self):
        if not torch.cuda.is_available():
            unittest.SkipTest("CUDA GPU required for this test")
        if importlib.util.find_spec("mmdet3d") is None:
            raise unittest.SkipTest("MMDetection3D is not installed")
        if importlib.util.find_spec("nuscenes") is None:
            raise unittest.SkipTest("nuScenes devkit is not installed")

        from inference_engine import InferenceEngine

        root_dir = Path(ROOT_DIR)
        checkpoint_file = root_dir / "src/detector/pgd/model/pgd_r101_caffe_fpn_gn-head_2x16_2x_nus-mono3d_finetune_20211114_162135-5ec7c1cd.pth"
        dataset_path = root_dir / "tests/data/nuScenes_dummy"
        if not checkpoint_file.exists():
            raise unittest.SkipTest(f"PGD checkpoint is missing: {checkpoint_file}")
        if not dataset_path.exists():
            raise unittest.SkipTest(f"nuScenes dummy dataset is missing: {dataset_path}")

        settings = SimpleNamespace(
            paths=SimpleNamespace(
                detection_path=str(root_dir / "src/detector/pgd/detections/"),
                dataset_path=str(dataset_path),
                config_file=root_dir / "third_party/mmdetection3d/configs/pgd/pgd_r101-caffe_fpn_head-gn_16xb2-2x_nus-mono3d_finetune.py"
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
            detector_name="pgd_mmdetection3d",
            dataset_path=str(dataset_path),
            detection_path=str(Path(settings.paths.detection_path)),
            model_path=str(checkpoint_file),
        )
        self.assertEqual(predictions.frames[0].frame, dataset.sample_records[0]["sample_token"])
        self.assertIsInstance(predictions.frames[0].dets, list)
        self._convert_predictions_to_lidar(predictions, dataset)
        file_name = "pgd-detections-nuScenes-mini-val"
        Serializer(settings=settings,
                   data_format="nuscenes",
                   file_name=file_name).serialize(data=predictions)
        self.assertTrue(os.path.isfile(f"{root_dir}/src/detector/pgd/detections/{file_name}.csv")) 

    def _convert_predictions_to_lidar(self, predictions, dataset):
        converter = CoordinateConverter()
        camera_channel = getattr(dataset, "camera_channel", "CAM_FRONT")

        for frame in predictions.frames:
            if not frame.dets:
                continue

            sample = dataset.nusc.get("sample", frame.frame)
            lidar_record = dataset.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            camera_record = dataset.nusc.get("sample_data", sample["data"][camera_channel])
            camera_to_lidar = dataset._sensor_to_sensor_transform(camera_record, lidar_record)

            converted_dets = []
            for detection in frame.dets:
                box = converter.convert_boxes_3d(
                    detection.box.reshape(1, -1),
                    src_mode="camera",
                    dst_mode="lidar",
                    rt_mat=camera_to_lidar,
                )[0]
                converted_dets.append(Detection(
                    score=detection.score,
                    label=detection.label,
                    box=box,
                ))
            frame.dets[:] = converted_dets

if __name__ == "__main__":
    unittest.main()
