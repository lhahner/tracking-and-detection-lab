from __future__ import annotations

from collections import deque

import numpy as np
import numpy as np
import open3d as o3d

from util.logging_config import LoggingConfig

def filter_points(points: np.ndarray, point_cloud_range: tuple[float, float, float, float, float, float]):
    """
    Filters out points according to the provided point_cloud_range.
    In e.g. KITTI-3D for example has no labels in the background this a filter should 
    cut out any points behind the vehicle.
    """
    logging_config = LoggingConfig()
    logger = logging_config.get_logger(__name__)

    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    xyz = points[:, :3]
    mask = (
        (xyz[:, 0] >= x_min)
        & (xyz[:, 0] <= x_max)
        & (xyz[:, 1] >= y_min)
        & (xyz[:, 1] <= y_max)
        & (xyz[:, 2] >= z_min)
        & (xyz[:, 2] <= z_max)
    )
    logger.info(f"filtered points from {points.shape} to {points[mask].shape}")
    return points[mask]

def remove_ground(points: np.ndarray, ground_threshold: float = -1.4) -> np.ndarray:
    return points[points[:, 2] > ground_threshold]

def dbscan_clustering(
      points: np.ndarray,
      eps: float = 1.0,
      min_points: int = 30,
      max_points: int = 5000,
):
    """
    Implements the Euclidean clustering implementation
    by Open3D, the clustering is based on the distance 
    of the individual data-points between each other.
    Farer data points are not likely to be in one cluster.
    """
    if points.size == 0:
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64, copy=False))

    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=float(eps),
            min_points=int(min_points),
            print_progress=False,
        )  
    )

    clusters = []
    for label in range(labels.max() + 1):
        cluster = points[labels == label]
        if cluster.shape[0] <= max_points:
            clusters.append(cluster)

    return clusters


def cluster_to_proposal(cluster: np.ndarray) -> dict:
    """
    Transforms the cluster to the actual proposal
    where the proposal of the objects location
    also includes the center, dimensions, yaw and 
    the actual data points of the cluster.
    """
    logging_config = LoggingConfig()
    logger = logging_config.get_logger(__name__)

    xyz = cluster[:, :3]
    min_xyz = xyz.min(axis=0)
    max_xyz = xyz.max(axis=0)
    center = (min_xyz + max_xyz) / 2.0
    dimensions = np.maximum(max_xyz - min_xyz, 1e-3)
    
    logger.debug(f"Cluster;min={min_xyz},max={max_xyz},center={center},dimensions={dimensions}")
    return {
        "center": center.astype(np.float32),
        "dimensions": dimensions.astype(np.float32),
        "yaw": np.float32(0.0),
        "points": cluster,
    }


def generate_proposals(
    points: np.ndarray,
    point_cloud_range=(0.0, -40.0, -3.0, 70.4, 40.0, 3.0),
    ground_threshold=-1.4,
    cluster_eps=1.0,
    min_points=30,
):
    filtered = filter_points(points, point_cloud_range)
    filtered = remove_ground(filtered, ground_threshold=ground_threshold)
    clusters = dbscan_clustering(filtered, eps=cluster_eps, min_points=min_points)
    return [cluster_to_proposal(cluster) for cluster in clusters]
