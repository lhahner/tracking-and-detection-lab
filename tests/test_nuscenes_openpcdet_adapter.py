import unittest
import torch
import importlib.util
from pathlib import Path

from definitions import ROOT_DIR
from detector.openpcdet_config import load_openpcdet_config


class TestNuScenesOpenPCDetAdapter(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Needs GPU")
        if importlib.util.find_spec("pcdet") is None:
            raise unittest.SkipTest("OpenPcDet is not installed")
        if importlib.util.find_spec("nuscenes") is None:
            raise unittest.SkipTest("nuScenes devkit is not installed")

    def _build_adapter(self, max_samples=1):
        from datasets.nuScenes_openpcdet_adapter import NuScenesOpenPCDetAdapter

        config_file = Path(ROOT_DIR) / "third_party/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_second_multihead.yaml"
        cfg = load_openpcdet_config(config_file)
        return NuScenesOpenPCDetAdapter(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            source_root=Path(ROOT_DIR) / "tests/data/nuScenes_dummy",
            version="v1.0-mini",
            max_samples=max_samples,
        )

    def test_retrieve_first_sample_from_dataset_adapter(self):
        dataset = self._build_adapter()
        data_dict = dataset[0]
        self.assertTrue(len(data_dict["points"]) > 0)
        self.assertEqual(data_dict["frame_id"], dataset.sample_records[0]["sample_token"])

    def test_metadata_retrieved_in_first_sample(self):
        dataset = self._build_adapter()
        data_dict = dataset[0]
        self.assertTrue(len(data_dict["metadata"].lidar_to_global) > 0)

    def test_native_prediction_api_is_available(self):
        dataset = self._build_adapter()
        self.assertTrue(hasattr(dataset, "generate_prediction_dicts"))
        self.assertTrue(hasattr(dataset, "evaluation"))
