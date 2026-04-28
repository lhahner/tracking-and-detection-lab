import importlib.util
import os
import sys
import unittest
from pathlib import Path

TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


class TestMMDetection3DCompatability(unittest.TestCase):
    def test_cpu_only_compatability(self):
        if importlib.util.find_spec("mmdet3d") is None:
            self.skipTest("mmdet3d is not installed in the active environment.")

        config_file = os.environ.get("MMDET3D_CONFIG")
        checkpoint_file = os.environ.get("MMDET3D_CHECKPOINT")

        if not config_file or not checkpoint_file:
            self.skipTest(
                "Set MMDET3D_CONFIG and MMDET3D_CHECKPOINT to run the compatibility test."
            )

        config_path = Path(config_file).expanduser()
        checkpoint_path = Path(checkpoint_file).expanduser()
        sample_path = Path(PROJECT_ROOT) / "tests" / "point_sample.bin"

        if not config_path.exists():
            self.skipTest(f"MMDET3D config file not found: {config_path}")
        if not checkpoint_path.exists():
            self.skipTest(f"MMDET3D checkpoint file not found: {checkpoint_path}")
        if not sample_path.exists():
            self.skipTest(f"Point-cloud sample file not found: {sample_path}")

        from mmdet3d.apis import inference_detector, init_model

        model = init_model(
            str(config_path),
            str(checkpoint_path),
            device="cpu",
        )
        result, data = inference_detector(model, str(sample_path))

        self.assertIsNotNone(model)
        self.assertIsNotNone(result)
        self.assertIsNotNone(data)
        self.assertTrue(
            hasattr(result, "pred_instances_3d"),
            "Inference returned no pred_instances_3d attribute.",
        )
