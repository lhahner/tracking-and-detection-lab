import unittest
from inference_engine import InferenceEngine
from types import SimpleNamespace
from definitions import ROOT_DIR

class TestPointpillarsEvaluationPipeline(unittest.TestCase):
    def build_settings(self):
        return SimpleNamespace(
            paths=SimpleNamespace(
                detection_path="output/",
                dataset_path="tests/data/kitti3d_dummy",
                config_file="",
            ),
            runtime=SimpleNamespace(
                datatype="png",
                dataset="kitti3d",
                display=False,
            ),
            benchmark=SimpleNamespace(iou_threshold=0.4, class_filter=[1, 2, 3]),
            tracker=SimpleNamespace(max_age=3, min_hits=2, iou_threshold=0.2),
            dataset=SimpleNamespace(classes=["Pedestrian", "Cyclist", "Car"]),
        )

    def test_predict_and_evaluate_from_inference_engine(self):
        detector_name = "pointpillars"
        dataset_path = "tests/data/kitti3d_dummy/"
        detection_path = "output/"
        model_path = "{ROOT_DIR}/third_party/pointpillars/_ext_src/pretrained/epoch_160.pth"
        inference_engine = InferenceEngine(settings=self.build_settings())
        inference_engine.load(split="val", max_samples=10)
        predictions = inference_engine.predict(
                detector_name=detector_name,
                dataset_path=dataset_path,
                detection_path=detection_path,
                model_path=model_path
                )

        self.assertTrue(len(predictions.frames) >= 1)
        results = inference_engine.evaluate_detection(detections=predictions,
                                                      classes=[1, 2, 3])
        self.assertTrue(len(results) >= 1)
