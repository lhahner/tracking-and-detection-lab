import unittest
from datasets.nuScenes import NuScenesDataset
from datasets.nuScenes_openpcdet_adapter import NuScenesOpenPCDetAdapter
from definitions import ROOT_DIR


class TestNuScenesOpenPCDetAdapter(unittest.TestCase):
    def test_retrieve_first_sample_from_dataset_adapter(self):
        nuScenes = NuScenesDataset(data_root=f"{ROOT_DIR}/tests/data/nuscenes",
                                   split="mini-val")
        nuScenesOpenPCDetAdapter = NuScenesOpenPCDetAdapter(nuScenes=nuScenes,
                                                            class_names=["car",
                                                                         "pedestrian"])
        data_dict = nuScenesOpenPCDetAdapter[0]
        breakpoint()
        self.assertTrue(data_dict["points"] > 0)
