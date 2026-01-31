#!/usr/bin/env python3
import argparse
import numpy as np
import open3d as o3d


def load_kitti_bin(bin_path: str) -> np.ndarray:
    """
    Loads a LiDAR .bin file in the common KITTI format:
    float32 points with [x, y, z, intensity].
    Returns Nx4 array.
    """
    points = np.fromfile(bin_path, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(
            f"File size does not match XYZI float32 format. "
            f"Total float32 values={points.size}, not divisible by 4."
        )
    return points.reshape(-1, 4)


def visualize(points_xyz: np.ndarray, intensity: np.ndarray = None, axis_size: float = 2.0):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_xyz))

    # Colorize by intensity if available
    geometries = []
    if intensity is not None:
        i = intensity.astype(np.float32)
        i = (i - i.min()) / (i.max() - i.min() + 1e-12)
        colors = np.stack([i, i, i], axis=1)  # grayscale
        pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries.append(pcd)

    # Add coordinate frame (X=red, Y=green, Z=blue)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size, origin=[0.0, 0.0, 0.0]
    )
    geometries.append(coord_frame)

    o3d.visualization.draw_geometries(geometries)


def main():
    parser = argparse.ArgumentParser(description="Visualize LiDAR .bin point cloud (XYZI float32)")
    parser.add_argument("bin_file", type=str, help="Path to LiDAR .bin file")

    parser.add_argument("--max_range", type=float, default=0.0,
                        help="Filter points beyond this range in meters (0 = disabled)")
    parser.add_argument("--axis_size", type=float, default=2.0,
                        help="Size of coordinate axes in visualization")

    args = parser.parse_args()

    pts = load_kitti_bin(args.bin_file)
    xyz = pts[:, :3]
    intensity = pts[:, 3]

    # Optional range filter
    if args.max_range > 0:
        r = np.linalg.norm(xyz, axis=1)
        mask = r <= args.max_range
        xyz = xyz[mask]
        intensity = intensity[mask]

    print(f"Loaded {xyz.shape[0]} points from {args.bin_file}")
    visualize(xyz, intensity, axis_size=args.axis_size)


if __name__ == "__main__":
    main()

