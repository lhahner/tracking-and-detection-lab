from __future__ import annotations

import math

import numpy as np

from util.kitti_calib import camera_to_lidar


def rotation_matrix_z(yaw: float) -> np.ndarray:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def image_box_to_lidar_proposal(obj: dict, calib: dict[str, np.ndarray], padding: float = 0.25) -> dict:
    """Approximate one KITTI labeled object as a LiDAR-frame oriented 3D box."""
    height, width, length = obj["dimensions"].astype(np.float32)
    center_camera = obj["location"].astype(np.float32).reshape(1, 3)
    center_lidar = camera_to_lidar(center_camera, calib)[0]

    # KITTI labels use bottom-centered camera boxes. Shift to approximate box center.
    center_lidar[2] += height / 2.0

    return {
        "center": center_lidar,
        "dimensions": np.array([length + padding, width + padding, height + padding], dtype=np.float32),
        "yaw": np.float32(-(obj["rotation_y"] + math.pi / 2.0)),
        "bbox_2d": obj["bbox"].astype(np.float32),
        "label": obj["type"],
    }


def extract_points_in_box(points: np.ndarray, proposal: dict) -> np.ndarray:
    """Return LiDAR points whose xyz coordinates lie inside an oriented 3D box."""
    if points is None or points.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    xyz = points[:, :3].astype(np.float32)
    center = proposal["center"].astype(np.float32)
    dims = proposal["dimensions"].astype(np.float32)
    yaw = float(proposal["yaw"])

    rotation = rotation_matrix_z(-yaw)
    local_xyz = (xyz - center) @ rotation.T
    half_dims = dims / 2.0
    mask = np.all(np.abs(local_xyz) <= half_dims, axis=1)
    return points[mask]


def proposal_iou_bev(first: dict, second: dict) -> float:
    """Cheap BEV IoU using axis-aligned boxes in local LiDAR coordinates."""
    first_center = first["center"][:2]
    second_center = second["center"][:2]
    first_dims = first["dimensions"][:2]
    second_dims = second["dimensions"][:2]

    first_min = first_center - first_dims / 2.0
    first_max = first_center + first_dims / 2.0
    second_min = second_center - second_dims / 2.0
    second_max = second_center + second_dims / 2.0

    inter_min = np.maximum(first_min, second_min)
    inter_max = np.minimum(first_max, second_max)
    inter_dims = np.maximum(0.0, inter_max - inter_min)
    inter_area = inter_dims[0] * inter_dims[1]

    first_area = first_dims[0] * first_dims[1]
    second_area = second_dims[0] * second_dims[1]
    union = first_area + second_area - inter_area
    if union <= 0:
        return 0.0
    return float(inter_area / union)
