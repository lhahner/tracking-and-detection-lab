from __future__ import annotations

import numpy as np
import torch

from util.logging_config import LoggingConfig

logging_config = LoggingConfig()
logger = logging_config.get_logger(__name__)


def sample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Returns a random set of points from the clustered Dataset.
    """
    breakpoint()
    if logger is None:
        raise ValueError("""
            This functions needs to have a logger object provided to run
            """)
    if points.shape[0] == 0:
        logger.warn("Some points are empty, continue with zeros")

        if points.ndim == 2:
            return np.zeros(num_points, points.shape[1], dtype=np.float32)
        else:
            return np.zeros(num_points, 4, dtype=np.float32)

    replace = points.shape[0] < num_points
    choice = np.random.choice(points.shape[0], num_points, replace=replace)
    return points[choice].astype(np.float32)


def normalize_points(points: np.ndarray) -> np.ndarray:
    """
    Normalizing by moving the local proposal point cloud so its mean is at the origin,
    computes the largest Euclidean distance from the centered origin,
    the proposal fits inside a unit sphere, the farthest point has distance from the
    origin has distance 1.0.
    """
    if points.size == 0:
        return points.astype(np.float32)
    normalized = points.astype(np.float32).copy()
    centroid = normalized[:, :3].mean(axis=0, keepdims=True)
    normalized[:, :3] -= centroid
    scale = np.linalg.norm(normalized[:, :3], axis=1).max()
    if scale > 0:
        normalized[:, :3] /= scale
    return normalized


def prepare_crop(
    points: np.ndarray, num_points: int, use_intensity: bool = True
) -> torch.Tensor:
    """
    Sample points and normalize them to provide a normalized,
    transposed sample set of the data points from the point
    cloud.
    """
    features = sample_points(points, num_points)
    if not use_intensity and features.shape[1] > 3:
        features = features[:, :3]
    features = normalize_points(features)
    return torch.from_numpy(features.T).float()
