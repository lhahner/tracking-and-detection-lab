import os
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from detector.pointnet.proposals import generate_proposals
from util.kitti_boxes import extract_points_in_box, image_box_to_lidar_proposal, proposal_iou_bev
from util.kitti_calib import parse_kitti_calibration
from util.logging_config import LoggerConfig

CLASSES = {
    "Background": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
    "Car": 3,
}

SUPPORTED_OBJECT_TYPES = tuple(name for name in CLASSES if name != "Background")


class Kitti3D(Dataset):
    """Single KITTI dataset entry point for frame and object-crop access."""

    def __init__(
        self,
        data_root,
        split="training",
        mode="frame",
        num_points=1024,
        include_background=False,
        background_iou_threshold=0.1,
        transform=None,
        logger=None
    ):
        if logger is None:
            raise ValueError("Provide logger for KITTI3D Dataset class")
        
        if split not in ["training", "testing"]:
            raise ValueError("Split name has to be training or testing")
        
        self.data_root = data_root
        self.split = split
        self.mode = mode
        self.num_points = num_points
        self.include_background = include_background
        self.background_iou_threshold = background_iou_threshold
        self.transform = transform

        self.dir_calib = os.path.join(data_root, self.split, "calib")
        self.dir_label = os.path.join(data_root, "training", "label_2")
        self.dir_velodyne = os.path.join(data_root, self.split, "velodyne")
        self.dir_image = os.path.join(data_root, self.split, "image_2")

        self.split_file = os.path.join(data_root, "ImageSets", f"{split}.txt")
        with open(self.split_file, "r", encoding="utf-8") as f:
            self.sample_ids = [line.strip() for line in f if line.strip()]

        if self.mode not in {"frame", "object"}:
            raise ValueError(f"Unsupported mode '{self.mode}'. Expected 'frame' or 'object'.")
         
        self.object_index = self._build_object_index() if self.mode == "object" else []

    def __len__(self):
        return len(self.object_index) if self.mode == "object" else len(self.sample_ids)

    def __getitem__(self, idx):
        if self.mode == "object":
            return self._get_object_item(idx)
        return self._get_frame_item(self.sample_ids[idx])

    def _get_frame_item(self, sample_id):
        image, image_path = self._load_image(sample_id)
        calib, calib_path = self._load_calib(sample_id)
        points, points_path = self._load_velodyne(sample_id)
        target, target_path = self._load_label(sample_id)

        if self.transform is not None and image is not None:
            image = self.transform(image)

        return {
            "image": image,
            "image_path": image_path,
            "points": points,
            "points_path": points_path,
            "calib": calib,
            "calib_path": calib_path,
            "target": target,
            "target_path": target_path,
            "sample_id": sample_id,
        }

    def _get_object_item(self, idx):
        item = self.object_index[idx]

        if item[0] == "background":
            _, sample_id, center, dimensions, yaw = item
            frame = self._get_frame_item(sample_id)
            proposal = {
                "center": center,
                "dimensions": dimensions,
                "yaw": yaw,
            }
            cropped_points = extract_points_in_box(frame["points"], proposal)
            sampled_points = self._sample_or_pad_points(cropped_points)

            return {
                "points": sampled_points,
                "raw_points": cropped_points,
                "label": CLASSES["Background"],
                "label_name": "Background",
                "proposal": proposal,
                "target": None,
                "sample_id": sample_id,
            }

        _, sample_id, object_idx = item
        frame = self._get_frame_item(sample_id)
        target_object = frame["target"][object_idx]
        proposal = image_box_to_lidar_proposal(target_object, frame["calib"])
        cropped_points = extract_points_in_box(frame["points"], proposal)
        sampled_points = self._sample_or_pad_points(cropped_points)
        
        label_name = target_object["type"]
        if label_name not in CLASSES:
            raise ValueError(f"Unsupported KITTI object type for training")
        
        return {
            "points": sampled_points,
            "raw_points": cropped_points,
            "label": CLASSES[label_name],
            "label_name": target_object["type"],
            "proposal": proposal,
            "target": target_object,
            "sample_id": sample_id,
        }

    def _build_object_index(self):
        """
        Building an object index object that contains all 
        the objects that were identified in the Dataset by
        the labels.
        """
        object_index = []
        if self.split == "testing":
            return object_index

        for sample_id in self.sample_ids:
            self.logger.info(f"Building object index for sample {sample_id}")
            
            # Loads labels into object array.
            objects, _ = self._load_label(sample_id)
            self.logger.info(f"Loaded objects {objects.shape}")
            
            # Filters out dont care values
            filtered_objects = self.filter_supported_objects(objects)
            self.logger.info(f"Loaded filtered objects {filtered_objects.shape}")
           
            # Builds object_index map with the data for training 
            for object_idx, _ in filtered_objects:
                object_index.append(("object", sample_id, object_idx))
                
            if self.include_background:
                object_index.extend(self._build_background_index(sample_id, filtered_objects))
        
        return object_index

    def _build_background_index(self, sample_id, filtered_objects):
        """
        Building up an background index object which hold
        all the points that are clustered and below our threshold
        so to consider as background points.
        """
        self.logger.inf(f"Building background index with sample {sample_id}")
        
        frame = self._get_frame_item(sample_id)
        if frame["points"] is None or frame["points"].size == 0:
            return []
        
        # Building ground truth bounding boxes from the labels (object = labels)
        gt_proposals = [
            image_box_to_lidar_proposal(obj, frame["calib"])
            for _, obj in filtered_objects
        ]
        background_index = []
        # generate_proposals will generate the clustered points that should represent an object
        for proposal in generate_proposals(frame["points"]):
            max_iou = max(
                (proposal_iou_bev(proposal, gt_proposal) for gt_proposal in gt_proposals),
                default=0.0,
            )
            # We only consider points to be objects whenever the IOU is below our threshold
            if max_iou < self.background_iou_threshold:
                center_proposal = proposal["center"].astype(np.float32)
                self.logger.debug(f"background;sample_id={sample},center={center_proposal}")
                
                background_index.append(
                    (
                        "background",
                        sample_id,
                        center_proposal,
                        proposal["dimensions"].astype(np.float32),
                        np.float32(proposal["yaw"]),
                    )
                )

        return background_index

    def filter_supported_objects(self, objects):
        """
        Filters the label classes to a list of supported objects.
        KITTI can have labels like "DontCare" or "Background" 
        which are not in interest for this project.
        """
        filtered = []
        for object_idx, obj in enumerate(objects):
            object_type = obj["type"]
            if object_type in SUPPORTED_OBJECT_TYPES:
                filtered.append((object_idx, obj))

        return filtered

    def _sample_or_pad_points(self, points):
        if points.size == 0:
            return np.zeros((self.num_points, 4), dtype=np.float32)

        if points.shape[0] >= self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            return points[choice].astype(np.float32)

        padding = np.zeros((self.num_points - points.shape[0], points.shape[1]), dtype=np.float32)
        return np.concatenate([points.astype(np.float32), padding], axis=0)

    def _load_image(self, sample_id):
        path = os.path.join(self.dir_image, f"{sample_id}.png")
        if not os.path.exists(path):
            return None, path
        return Image.open(path).convert("RGB"), path

    def _load_velodyne(self, sample_id):
        path = os.path.join(self.dir_velodyne, f"{sample_id}.bin")
        if not os.path.exists(path):
            return None, path
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4), path

    def _load_calib(self, sample_id):
        path = os.path.join(self.dir_calib, f"{sample_id}.txt")
        return parse_kitti_calibration(path), path

    def _load_label(self, sample_id):
        """
        Opens the label file and parses the data to a object that contains
        all the labels for a given frame.
        """
        if self.split == "testing":
            return [], None

        path = os.path.join(self.dir_label, f"{sample_id}.txt")
        if not os.path.exists(path):
            return [], path

        objects: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                fields = line.strip().split()
                if not fields:
                    continue
                objects.append(
                    {
                        "type": fields[0],
                        "truncated": float(fields[1]),
                        "occluded": int(fields[2]),
                        "alpha": float(fields[3]),
                        "bbox": np.array(fields[4:8], dtype=np.float32),
                        "dimensions": np.array(fields[8:11], dtype=np.float32),
                        "location": np.array(fields[11:14], dtype=np.float32),
                        "rotation_y": float(fields[14]),
                    }
                )
        return objects, path
