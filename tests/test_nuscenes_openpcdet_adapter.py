import unittest
import torch
import importlib.util
from datasets.nuScenes import NuScenesDataset
from definitions import ROOT_DIR


class TestNuScenesOpenPCDetAdapter(unittest.TestCase):
    def test_retrieve_first_sample_from_dataset_adapter(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Needs GPU")
        if importlib.util.find_spec("pcdet") is None:
            raise unittest.SkipTest("OpenPcDet is not installed")
        if importlib.util.find_spec("nuscenes") is None:
            raise unittest.SkipTest("nuScenes devkit is not installed")
        from datasets.nuScenes_openpcdet_adapter import NuScenesOpenPCDetAdapter
        nuScenes = NuScenesDataset(data_root=f"{ROOT_DIR}/tests/data/nuScenes_dummy",
                                   split="mini_val")
        nuScenesOpenPCDetAdapter = NuScenesOpenPCDetAdapter(nuScenes=nuScenes,
                                                            class_names=["car",
                                                                         "pedestrian"])
        data_dict = nuScenesOpenPCDetAdapter[0]
        self.assertTrue(len(data_dict["points"]) > 0)

    def test_metadata_retrieved_in_first_sample(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Needs GPU")
        if importlib.util.find_spec("pcdet") is None:
            raise unittest.SkipTest("OpenPcDet is not installed")
        if importlib.util.find_spec("nuscenes") is None:
            raise unittest.SkipTest("nuScenes devkit is not installed")
        from datasets.nuScenes_openpcdet_adapter import NuScenesOpenPCDetAdapter
        nuScenes = NuScenesDataset(data_root=f"{ROOT_DIR}/tests/data/nuScenes_dummy",
                                   split="mini_val")
        nuScenesOpenPCDetAdapter = NuScenesOpenPCDetAdapter(nuScenes=nuScenes,
                                                            class_names=["car",
                                                                         "pedestrian"])
        data_dict = nuScenesOpenPCDetAdapter[0]
        self.assertTrue(len(data_dict["metadata"].lidar_to_global) > 0)
