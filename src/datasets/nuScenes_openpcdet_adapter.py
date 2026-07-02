from pcdet.datasets.dataset import DatasetTemplate
from pathlib import Path
from easydict import EasyDict
from pcdet.config import cfg_from_yaml_file
import numpy as np


class NuScenesOpenPCDetAdapter(DatasetTemplate):
    def __init__(
            self,
            nuScenes,
            class_names,
            root_path=".",
            logger=None,
            max_samples=None,
        ):
            cfg = EasyDict()
            cfg_from_yaml_file(str(Path(__file__).resolve().parents[2] / "third_party/OpenPCDet/tools/cfgs/dataset_configs/nuscenes_dataset.yaml"), cfg)
            dataset_cfg = cfg
            super().__init__(
                dataset_cfg=dataset_cfg,
                class_names=class_names,
                training=False,
                root_path=Path(root_path),
                logger=logger,
            )
            self.nuScenes = nuScenes
            self.max_samples = max_samples
            self.sample_records = self.nuScenes.sample_records
    
    def __len__(self):
        return len(self.sample_records)

    def __getitem__(self, index):
        sample = self.nuScenes[self.nuScenes.sample_records.index(self.sample_records[index])]
        points = sample["points"].astype(np.float32)
        if points.shape[1] == 4:
            points = np.concatenate([points, np.zeros((points.shape[0], 1), dtype=np.float32)], axis=1)
        sample_id = sample["sample_id"]

        input_dict = {
                "points": points,
                "frame_id": sample_id
                }
        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict["target"] = sample["target"]
        return data_dict

    def convert_ground_truth(self, ground_truth_dicts, frame):
        return self.nuScenes.convert_ground_truth(ground_truth_dicts, frame)
