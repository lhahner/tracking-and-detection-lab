from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from datasets.nuScenes import DETECTION_CLASSES
from entities.metadata import NuScenesMetadata
from pcdet.datasets.nuscenes.nuscenes_dataset import (
    NuScenesDataset as OpenPCDetNuScenesDataset,
    create_nuscenes_info,
)
from pcdet.utils import common_utils


class NuScenesOpenPCDetAdapter(OpenPCDetNuScenesDataset):
    def __init__(
        self,
        dataset_cfg,
        class_names,
        source_root,
        version,
        max_samples=None,
        logger=None,
    ):
        self.source_root = Path(source_root).resolve()
        self.version = version
        self.max_samples = max_samples
        self._cache_root = self.__ensure_openpcdet_layout(dataset_cfg)

        dataset_cfg.VERSION = version
        if logger is None:
            logger = common_utils.create_logger(log_file=None)

        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=self._cache_root,
            logger=logger,
        )

        self.sample_records = [{"sample_token": info["token"]} for info in self.infos]
        self.labels = DETECTION_CLASSES

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        info = self.infos[index]
        sample_token = info["token"]
        data_dict["frame_id"] = sample_token
        data_dict["metadata"] = self.__metadata_from_info(info)
        data_dict["target"] = self.__targets_from_info(info)
        return data_dict

    def convert_ground_truth(self, ground_truth_dicts, frame):
        rows = []
        for target in ground_truth_dicts:
            box = target.get("box")
            if box is None:
                continue
            rows.append(
                torch.tensor(
                    [*np.asarray(box, dtype=np.float32), 0.0, target["label"]],
                    dtype=torch.float32,
                )
            )
        if not rows:
            return torch.empty((0, 9), dtype=torch.float32)
        return torch.stack(rows)

    def __ensure_openpcdet_layout(self, dataset_cfg):
        cache_root = self.__cache_root_for_source()
        cache_version_root = cache_root / self.version
        cache_version_root.mkdir(parents=True, exist_ok=True)

        for name in ("samples", "sweeps", "maps", self.version):
            source_path = self.source_root / name
            target_path = cache_version_root / name
            if not source_path.exists():
                continue
            self.__ensure_link(source_path, target_path)

        max_sweeps = int(dataset_cfg.get("MAX_SWEEPS", 10))
        train_info = cache_version_root / f"nuscenes_infos_{max_sweeps}sweeps_train.pkl"
        val_info = cache_version_root / f"nuscenes_infos_{max_sweeps}sweeps_val.pkl"
        if not train_info.exists() or not val_info.exists():
            create_nuscenes_info(
                self.version,
                cache_root,
                cache_root,
                max_sweeps=max_sweeps,
                with_cam=bool(dataset_cfg.get("CAMERA_CONFIG", None)),
            )

        return cache_root

    def __cache_root_for_source(self):
        source_hash = hashlib.sha1(str(self.source_root).encode("utf-8")).hexdigest()[:12]
        return (
            Path(tempfile.gettempdir())
            / "tracking_and_detection_lab_openpcdet_nuscenes"
            / source_hash
        )

    @staticmethod
    def __ensure_link(source_path, target_path):
        if target_path.is_symlink() and Path(os.readlink(target_path)) == source_path:
            return
        if target_path.exists() or target_path.is_symlink():
            if target_path.is_dir() and not target_path.is_symlink():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
        target_path.symlink_to(source_path, target_is_directory=source_path.is_dir())

    @staticmethod
    def __metadata_from_info(info):
        lidar_to_global = np.linalg.inv(info["car_from_global"]) @ np.linalg.inv(
            info["ref_from_car"]
        )
        ego_to_global = np.linalg.inv(info["car_from_global"])
        return NuScenesMetadata(
            sample_token=info["token"],
            time_stamp=float(info["timestamp"]),
            lidar_to_global=lidar_to_global.tolist(),
            ego=ego_to_global.tolist(),
            aux_info={"is_key_frame": True},
        )

    @staticmethod
    def __targets_from_info(info):
        targets = []
        gt_boxes = info.get("gt_boxes", np.empty((0, 9), dtype=np.float32))
        gt_names = info.get("gt_names", [])
        for box, name in zip(gt_boxes, gt_names):
            if name not in DETECTION_CLASSES:
                continue
            box_array = np.asarray(box[:7], dtype=np.float32)
            targets.append(
                {
                    "type": name,
                    "label": DETECTION_CLASSES[name],
                    "box": box_array,
                    "location": box_array[:3],
                    "dimensions": box_array[3:6],
                    "yaw": np.float32(box_array[6]),
                    "rotation_y": np.float32(box_array[6]),
                }
            )
        return targets


NuScenesOpenPCDetNativeDataset = NuScenesOpenPCDetAdapter
