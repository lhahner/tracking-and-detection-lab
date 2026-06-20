import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from definitions import ROOT_DIR
from inference_engine import InferenceEngine


class TestPointRCNNMMDetection3DKitti3DPipeline(unittest.TestCase):
    def test_predict_and_evaluate_from_inference_engine_with_kitti3d(self):
        if importlib.util.find_spec("mmdet3d") is None:
            raise unittest.SkipTest("MMDetection3D is not installed")

        root_dir = Path(ROOT_DIR)
        checkpoint_file = root_dir / "src/detector/pointrcnn/model/point_rcnn_2x8_kitti-3d-3classes_20211208_151344.pth"
        config_file = root_dir / "third_party/mmdetection3d/configs/point_rcnn/point-rcnn_8xb2_kitti-3d-3class.py"
        dataset_path = root_dir / "tests/data/kitti3d_dummy"
        if not checkpoint_file.exists():
            raise unittest.SkipTest(f"PointRCNN checkpoint is missing: {checkpoint_file}")
        if not config_file.exists():
            raise unittest.SkipTest(f"PointRCNN config is missing: {config_file}")
        if not dataset_path.exists():
            raise unittest.SkipTest(f"Kitti3d dummy dataset is missing: {dataset_path}")

        settings = SimpleNamespace(
            paths=SimpleNamespace(
                detection_path=str(root_dir / "output"),
                dataset_path=str(dataset_path),
                config_file=str(config_file),
            ),
            runtime=SimpleNamespace(
                datatype="bin",
                dataset="kitti3d",
                display=False,
            ),
            benchmark=SimpleNamespace(
                iou_threshold=0.4,
            ),
            tracker=SimpleNamespace(max_age=3, min_hits=2, iou_threshold=0.2),
            dataset=SimpleNamespace(
                classes={
                    "Car": 3,
                    "Pedestrian": 1,
                    "Cyclist": 2,
                }
            ),
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
            dataset_path=str(dataset_path),
            detection_path=str(Path(settings.paths.detection_path)),
            model_path=str(checkpoint_file),
        )
        self.assertEqual(len(predictions.frames), 1)
        self.assertIsInstance(predictions.frames[0].dets, list)

        results = inference_engine.evaluate_detection(
            detections=predictions,
            classes=[3, 1, 2],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["frame"], predictions.frames[0].frame)
        self.assertIn("mAP", results[0])
        self.assertGreater(float(results[0]["mAP"]), 0.0)
        self.assertLessEqual(float(results[0]["mAP"]), 1.0)


if __name__ == "__main__":
    unittest.main()
