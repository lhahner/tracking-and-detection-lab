from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_kitti_calibration(path: str | Path) -> dict[str, np.ndarray]:
    """Parse one KITTI calibration file into matrix-shaped numpy arrays."""
    parsed: dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            raw = np.array([float(x) for x in value.split()], dtype=np.float32)
            if key.startswith("P"):
                parsed[key] = raw.reshape(3, 4)
            elif key == "R0_rect":
                parsed[key] = raw.reshape(3, 3)
            elif key in {"Tr_velo_to_cam", "Tr_imu_to_velo"}:
                parsed[key] = raw.reshape(3, 4)
            else:
                parsed[key] = raw
    return parsed


def to_homogeneous(points_xyz: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    return np.concatenate([points_xyz, ones], axis=1)


def extend_to_4x4(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (4, 4):
        return matrix
    extended = np.eye(4, dtype=matrix.dtype)
    extended[: matrix.shape[0], : matrix.shape[1]] = matrix
    return extended


def camera_to_lidar(points_camera: np.ndarray, calib: dict[str, np.ndarray]) -> np.ndarray:
    """Convert Nx3 camera coordinates into LiDAR coordinates."""
    r0_rect = extend_to_4x4(calib["R0_rect"])
    tr_velo_to_cam = extend_to_4x4(calib["Tr_velo_to_cam"])
    camera_to_velo = np.linalg.inv(r0_rect @ tr_velo_to_cam)
    points_h = to_homogeneous(points_camera.astype(np.float32))
    transformed = (camera_to_velo @ points_h.T).T
    return transformed[:, :3]
