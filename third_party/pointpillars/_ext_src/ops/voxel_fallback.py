"""
WARNING
This code is AI Generated and currently
not verified by human-hand nor written 
any tests.
"""
import numpy as np
import torch


def hard_voxelize(points,
                  voxels,
                  coors,
                  num_points_per_voxel,
                  voxel_size,
                  coors_range,
                  max_points,
                  max_voxels,
                  ndim=3,
                  deterministic=True):
    del deterministic

    points_cpu = points.detach().cpu()
    points_array = points_cpu.numpy()
    voxel_size = np.asarray(voxel_size, dtype=np.float32)
    coors_range = np.asarray(coors_range, dtype=np.float32)
    grid_size = np.rint(
        (coors_range[ndim:] - coors_range[:ndim]) / voxel_size
    ).astype(np.int64)

    point_coors = np.floor(
        (points_array[:, :ndim] - coors_range[:ndim]) / voxel_size
    ).astype(np.int64)
    valid = np.all(
        (point_coors >= 0) & (point_coors < grid_size),
        axis=1,
    )
    valid_points = points_array[valid]
    point_coors = point_coors[valid, ::-1]

    if point_coors.shape[0] == 0:
        return 0

    unique_coors, first_indices, inverse = np.unique(
        point_coors,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    voxel_order = np.argsort(first_indices, kind="stable")
    if max_voxels != -1:
        voxel_order = voxel_order[:max_voxels]

    voxel_count = len(voxel_order)
    voxels_cpu = torch.zeros(
        (voxel_count, max_points, points.shape[1]),
        dtype=points_cpu.dtype,
    )
    coors_cpu = torch.from_numpy(
        unique_coors[voxel_order].astype(np.int32, copy=False)
    )
    counts_cpu = torch.zeros(voxel_count, dtype=torch.int32)

    for output_index, unique_index in enumerate(voxel_order):
        point_indices = np.flatnonzero(inverse == unique_index)[:max_points]
        point_count = len(point_indices)
        if point_count:
            voxels_cpu[output_index, :point_count] = points_cpu[
                torch.from_numpy(point_indices)
            ]
            counts_cpu[output_index] = point_count

    voxels[:voxel_count].copy_(voxels_cpu.to(voxels.device))
    coors[:voxel_count].copy_(coors_cpu.to(coors.device))
    num_points_per_voxel[:voxel_count].copy_(
        counts_cpu.to(num_points_per_voxel.device)
    )
    return voxel_count
