"""nuScenes dataset adapter for the tracking and detection lab.

The adapter uses the official nuScenes devkit for metadata and coordinate
transforms. It deliberately avoids SECOND-specific preprocessing, registries,
pickle schemas, and evaluation code.

Internal 3D boxes use LiDAR coordinates and the layout:
    [x, y, z, length, width, height, yaw]

Point clouds use:
    [x, y, z, intensity] when ``include_time_lag`` is false, otherwise
    [x, y, z, intensity, time_lag_seconds].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DETECTION_CLASSES = {
    "Background": 0,
    "barrier": 1,
    "bicycle": 2,
    "bus": 3,
    "car": 4,
    "construction_vehicle": 5,
    "motorcycle": 6,
    "pedestrian": 7,
    "traffic_cone": 8,
    "trailer": 9,
    "truck": 10,
}

TRACKING_CLASSES = {
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "pedestrian",
    "trailer",
    "truck",
}

CATEGORY_MAPPING = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

CAMERA_CHANNELS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


class NuScenesDataset(Dataset):
    """Expose scene-ordered nuScenes keyframes through the project dataset API."""

    def __init__(
        self,
        data_root: str | Path,
        version: str = "v1.0-mini",
        split: str = "mini_train",
        max_sweeps: int = 1,
        include_time_lag: bool = False,
        load_images: bool = True,
        camera_channel: str = "CAM_FRONT",
        class_names: Iterable[str] | None = None,
        transform=None,
        verbose: bool = False,
    ):
        """Load nuScenes metadata and build a scene-preserving sample index.

        Args:
            data_root: Directory containing the nuScenes dataset.
            version: nuScenes release, for example ``v1.0-mini``.
            split: Official split compatible with ``version``.
            max_sweeps: Number of LiDAR sweeps including the keyframe.
            include_time_lag: Append sweep age as a fifth point feature.
            load_images: Load the selected camera image in ``__getitem__``.
            camera_channel: Camera used for the top-level ``image`` field.
            class_names: Optional mapped detection classes to retain.
            transform: Optional image transform.
            verbose: Forward verbose loading to the official devkit.
        """
        if max_sweeps < 1:
            raise ValueError("max_sweeps must be at least 1")
        if camera_channel not in CAMERA_CHANNELS:
            raise ValueError(
                f"Unsupported camera channel '{camera_channel}'. "
                f"Expected one of {CAMERA_CHANNELS}."
            )

        NuScenes, create_splits_scenes = self._import_devkit()

        self.data_root = Path(data_root)
        self.version = version
        self.split = split
        self.max_sweeps = max_sweeps
        self.include_time_lag = include_time_lag
        self.load_images = load_images
        self.camera_channel = camera_channel
        selected_classes = (
            DETECTION_CLASSES.keys() if class_names is None else class_names
        )
        self.class_names = set(selected_classes)
        self.class_names.discard("Background")
        self.transform = transform
        self.mode = "frame"

        self._validate_split_version()
        self.nusc = NuScenes(
            version=self.version,
            dataroot=str(self.data_root),
            verbose=verbose,
        )
        split_scenes = create_splits_scenes(verbose=False)
        if self.split not in split_scenes:
            raise ValueError(
                f"Unknown nuScenes split '{self.split}'. "
                f"Available splits: {sorted(split_scenes)}"
            )

        self.sample_records = self._build_sample_index(split_scenes[self.split])

    @staticmethod
    def _import_devkit():
        """Import optional nuScenes dependencies only when the dataset is used."""
        try:
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError as exc:
            raise ImportError(
                "NuScenesDataset requires the official nuScenes devkit. "
                "Install it with `pip install nuscenes-devkit`."
            ) from exc
        return NuScenes, create_splits_scenes

    def _validate_split_version(self) -> None:
        """Reject official splits that do not belong to the selected release."""
        compatible_splits = {
            "v1.0-mini": {"mini_train", "mini_val"},
            "v1.0-trainval": {"train", "val", "train_detect", "train_track"},
            "v1.0-test": {"test"},
        }
        if self.version not in compatible_splits:
            raise ValueError(
                f"Unsupported nuScenes version '{self.version}'. "
                f"Expected one of {sorted(compatible_splits)}."
            )
        if self.split not in compatible_splits[self.version]:
            raise ValueError(
                f"Split '{self.split}' is not compatible with '{self.version}'. "
                f"Expected one of {sorted(compatible_splits[self.version])}."
            )

    def __len__(self) -> int:
        """Return the number of keyframe samples in the selected split."""
        return len(self.sample_records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load one keyframe, its optional past sweeps, image, and annotations."""
        index_record = self.sample_records[idx]
        sample = self.nusc.get("sample", index_record["sample_token"])
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_record = self.nusc.get("sample_data", lidar_token)

        points, sweep_paths = self._load_lidar_sweeps(lidar_record)
        image, image_path = self._load_camera_image(sample)
        calibration = self._load_calibration(sample)
        targets = self._load_annotations(lidar_token)

        if self.transform is not None and image is not None:
            image = self.transform(image)

        lidar_path = self._resolve_data_path(lidar_record["filename"])
        return {
            "image": image,
            "image_path": image_path,
            "images": calibration["camera_paths"],
            "points": points,
            "points_path": str(lidar_path),
            "sweep_paths": sweep_paths,
            "calib": calibration,
            "calib_path": None,
            "target": targets,
            "target_path": None,
            "sample_id": sample["token"],
            "sample_token": sample["token"],
            "scene_id": sample["scene_token"],
            "scene_name": index_record["scene_name"],
            "frame_index": index_record["frame_index"],
            "timestamp": sample["timestamp"],
            "is_first_frame": index_record["is_first_frame"],
            "is_last_frame": index_record["is_last_frame"],
        }

    def _build_sample_index(self, split_scene_names: Iterable[str]) -> list[dict[str, Any]]:
        """Create an index ordered by scene and by each sample's linked list."""
        selected_names = set(split_scene_names)
        scenes = [
            scene for scene in self.nusc.scene if scene["name"] in selected_names
        ]
        scenes.sort(key=lambda scene: scene["name"])

        sample_records = []
        for scene in scenes:
            scene_samples = []
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = self.nusc.get("sample", sample_token)
                scene_samples.append(sample)
                sample_token = sample["next"]

            for frame_index, sample in enumerate(scene_samples):
                sample_records.append(
                    {
                        "sample_token": sample["token"],
                        "scene_token": scene["token"],
                        "scene_name": scene["name"],
                        "frame_index": frame_index,
                        "is_first_frame": frame_index == 0,
                        "is_last_frame": frame_index == len(scene_samples) - 1,
                    }
                )

        return sample_records

    def _load_lidar_sweeps(
        self, keyframe_record: dict[str, Any]
    ) -> tuple[np.ndarray, list[str]]:
        """Load and motion-compensate past LiDAR sweeps into the keyframe frame."""
        keyframe_time = keyframe_record["timestamp"] / 1e6
        point_sets = []
        sweep_paths = []
        sweep_record = keyframe_record

        for sweep_index in range(self.max_sweeps):
            points, path = self._load_lidar_file(sweep_record)
            if sweep_index > 0:
                transform = self._sensor_to_sensor_transform(
                    source_sensor_record=sweep_record,
                    target_sensor_record=keyframe_record,
                )
                points[:, :3] = self._transform_points(points[:, :3], transform)

            time_lag = keyframe_time - sweep_record["timestamp"] / 1e6
            if self.include_time_lag:
                time_column = np.full(
                    (points.shape[0], 1), time_lag, dtype=np.float32
                )
                points = np.concatenate([points, time_column], axis=1)

            point_sets.append(points)
            sweep_paths.append(str(path))

            previous_token = sweep_record["prev"]
            if not previous_token:
                break
            sweep_record = self.nusc.get("sample_data", previous_token)

        if not point_sets:
            feature_count = 5 if self.include_time_lag else 4
            return np.empty((0, feature_count), dtype=np.float32), sweep_paths
        return np.concatenate(point_sets, axis=0), sweep_paths

    def _load_lidar_file(
        self, sample_data_record: dict[str, Any]
    ) -> tuple[np.ndarray, Path]:
        """Read a nuScenes LiDAR binary file as normalized XYZI points."""
        path = self._resolve_data_path(sample_data_record["filename"])
        raw_points = np.fromfile(path, dtype=np.float32)
        if raw_points.size % 5 != 0:
            raise ValueError(
                f"Invalid nuScenes LiDAR file '{path}': "
                f"{raw_points.size} values cannot be reshaped to [N, 5]."
            )

        points = raw_points.reshape(-1, 5)[:, :4].copy()
        points[:, 3] /= 255.0
        return points.astype(np.float32, copy=False), path

    def _load_camera_image(
        self, sample: dict[str, Any]
    ) -> tuple[Image.Image | None, str]:
        """Load the configured camera image while always returning its path."""
        camera_token = sample["data"][self.camera_channel]
        camera_record = self.nusc.get("sample_data", camera_token)
        image_path = self._resolve_data_path(camera_record["filename"])
        if not self.load_images:
            return None, str(image_path)
        return Image.open(image_path).convert("RGB"), str(image_path)

    def _load_calibration(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Collect sensor calibration, ego pose, intrinsics, and image paths."""
        lidar_record = self.nusc.get(
            "sample_data", sample["data"]["LIDAR_TOP"]
        )
        lidar_calibration = self.nusc.get(
            "calibrated_sensor", lidar_record["calibrated_sensor_token"]
        )
        lidar_pose = self.nusc.get("ego_pose", lidar_record["ego_pose_token"])

        cameras = {}
        camera_paths = {}
        for channel in CAMERA_CHANNELS:
            camera_record = self.nusc.get("sample_data", sample["data"][channel])
            camera_calibration = self.nusc.get(
                "calibrated_sensor",
                camera_record["calibrated_sensor_token"],
            )
            camera_pose = self.nusc.get(
                "ego_pose", camera_record["ego_pose_token"]
            )
            camera_path = self._resolve_data_path(camera_record["filename"])
            camera_paths[channel] = str(camera_path)
            cameras[channel] = {
                "sample_data_token": camera_record["token"],
                "translation": np.asarray(
                    camera_calibration["translation"], dtype=np.float32
                ),
                "rotation": np.asarray(
                    camera_calibration["rotation"], dtype=np.float32
                ),
                "camera_intrinsic": np.asarray(
                    camera_calibration["camera_intrinsic"], dtype=np.float32
                ),
                "ego_translation": np.asarray(
                    camera_pose["translation"], dtype=np.float32
                ),
                "ego_rotation": np.asarray(
                    camera_pose["rotation"], dtype=np.float32
                ),
            }

        return {
            "lidar": {
                "sample_data_token": lidar_record["token"],
                "translation": np.asarray(
                    lidar_calibration["translation"], dtype=np.float32
                ),
                "rotation": np.asarray(
                    lidar_calibration["rotation"], dtype=np.float32
                ),
                "ego_translation": np.asarray(
                    lidar_pose["translation"], dtype=np.float32
                ),
                "ego_rotation": np.asarray(
                    lidar_pose["rotation"], dtype=np.float32
                ),
            },
            "cameras": cameras,
            "camera_paths": camera_paths,
        }

    def _load_annotations(self, lidar_token: str) -> list[dict[str, Any]]:
        """Convert keyframe annotations to the internal LiDAR box convention."""
        _, boxes, _ = self.nusc.get_sample_data(lidar_token)
        targets = []
        for box in boxes:
            annotation = self.nusc.get("sample_annotation", box.token)
            category_name = annotation["category_name"]
            mapped_name = CATEGORY_MAPPING.get(category_name)
            if mapped_name is None or mapped_name not in self.class_names:
                continue

            length = float(box.wlh[1])
            width = float(box.wlh[0])
            height = float(box.wlh[2])
            yaw = float(box.orientation.yaw_pitch_roll[0])
            velocity_global = self.nusc.box_velocity(box.token)
            box_array = np.asarray(
                [
                    box.center[0],
                    box.center[1],
                    box.center[2],
                    length,
                    width,
                    height,
                    yaw,
                ],
                dtype=np.float32,
            )

            targets.append(
                {
                    "type": mapped_name,
                    "label": DETECTION_CLASSES[mapped_name],
                    "box": box_array,
                    "location": box_array[:3],
                    "dimensions": box_array[3:6],
                    "yaw": np.float32(yaw),
                    "rotation_y": np.float32(yaw),
                    "sample_annotation_token": annotation["token"],
                    "instance_token": annotation["instance_token"],
                    "attribute_tokens": list(annotation["attribute_tokens"]),
                    "visibility_token": annotation["visibility_token"],
                    "num_lidar_pts": annotation["num_lidar_pts"],
                    "num_radar_pts": annotation["num_radar_pts"],
                    "velocity_global": np.asarray(
                        velocity_global[:2], dtype=np.float32
                    ),
                    "is_tracking_class": mapped_name in TRACKING_CLASSES,
                }
            )
        return targets

    def convert_ground_truth(
        self, ground_truth_dicts: Iterable[dict[str, Any]]
    ) -> torch.Tensor:
        """Convert annotations to the project's ``box, score, label`` tensor."""
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

    def _sensor_to_sensor_transform(
        self,
        source_sensor_record: dict[str, Any],
        target_sensor_record: dict[str, Any],
    ) -> np.ndarray:
        """Build a homogeneous transform from one sensor frame to another."""
        source_to_global = self._sensor_to_global_transform(source_sensor_record)
        target_to_global = self._sensor_to_global_transform(target_sensor_record)
        return np.linalg.inv(target_to_global) @ source_to_global

    def _sensor_to_global_transform(
        self, sensor_record: dict[str, Any]
    ) -> np.ndarray:
        """Build a homogeneous transform from a sensor frame to global space."""
        calibrated_sensor = self.nusc.get(
            "calibrated_sensor", sensor_record["calibrated_sensor_token"]
        )
        ego_pose = self.nusc.get("ego_pose", sensor_record["ego_pose_token"])
        sensor_to_ego = self._transform_matrix(
            calibrated_sensor["translation"],
            calibrated_sensor["rotation"],
        )
        ego_to_global = self._transform_matrix(
            ego_pose["translation"],
            ego_pose["rotation"],
        )
        return ego_to_global @ sensor_to_ego

    @staticmethod
    def _transform_matrix(
        translation: Iterable[float], quaternion: Iterable[float]
    ) -> np.ndarray:
        """Create a 4x4 transform from translation and ``w, x, y, z`` rotation."""
        rotation = NuScenesDataset._quaternion_rotation_matrix(quaternion)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.asarray(translation, dtype=np.float64)
        return transform

    @staticmethod
    def _quaternion_rotation_matrix(
        quaternion: Iterable[float],
    ) -> np.ndarray:
        """Convert a ``w, x, y, z`` quaternion to a 3x3 rotation matrix."""
        w, x, y, z = np.asarray(quaternion, dtype=np.float64)
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm == 0:
            raise ValueError("Quaternion must have a non-zero norm")
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return np.asarray(
            [
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ],
                [
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ],
                [
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Apply a homogeneous transform to an ``[N, 3]`` point array."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")
        homogeneous = np.concatenate(
            [points.astype(np.float64), np.ones((points.shape[0], 1))],
            axis=1,
        )
        transformed = homogeneous @ transform.T
        return transformed[:, :3].astype(np.float32)

    def _resolve_data_path(self, filename: str) -> Path:
        """Resolve devkit-relative filenames against the dataset root."""
        path = Path(filename)
        if path.is_absolute():
            return path
        return self.data_root / path


# Keep the shorter spelling available for configuration and imports.
NuScenes = NuScenesDataset
