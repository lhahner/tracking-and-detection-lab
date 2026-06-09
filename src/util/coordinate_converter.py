"""
Direct mmdetection3d references used:

  - LiDAR corners: mmdetection3d/mmdet3d/structures/bbox_3d/lidar_box3d.py:41
  - Camera/LiDAR conversion semantics: mmdetection3d/mmdet3d/structures/bbox_3d/box_3d_mode.py:127
  - Transform application and limit_period: mmdetection3d/mmdet3d/structures/bbox_3d/box_3d_mode.py:220
"""
import numpy as np
import torch


class CoordinateConverter:
    def convert2DDetectionToBox(self, seq_dets, frame):
        """Convert MOT detections for a frame from `xywh` to `xyxy`.

        Args:
            seq_dets: Detection array containing frame IDs and bounding boxes.
            frame: Frame number to extract.

        Returns:
            numpy.ndarray: Bounding boxes for the frame in `xyxy` format with
            scores preserved in the last column.

        Raises:
            ValueError: If the detections array is empty.
        """
        if len(seq_dets) < 1:
            raise ValueError("The sequence detections are empty")
        dets = seq_dets[seq_dets[:,0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2]
        return dets

    def convert_boxes_3d(
            self,
            boxes_3d,
            src_mode,
            dst_mode,
            rt_mat):
        if src_mode == dst_mode:
            return boxes_3d.clone()
        if src_mode == "camera" and dst_mode == "lidar":
            return self.__camera_to_lidar_boxes(boxes_3d, rt_mat)
        if src_mode == "lidar" and dst_mode == "camera":
            return self.__lidar_to_camera_boxes(boxes_3d, rt_mat)
        raise ValueError("Cannot convert")

    def boxes_3d_to_corners(self, boxes_3d, box_mode):
        """
        Args:
            boxes_3d: bounding box in 3d, should be of type torch.tensor
            box_mode: convert either to camera or to lidar
        Returns
            The converter tensor in the required box mode.
        """
        if box_mode == "camera":
            return self.__camera_boxes_to_corners(boxes_3d)
        if box_mode == "lidar":
            return self.__lidar_boxes_to_corners(boxes_3d)
        raise ValueError(f"Unsupported box_mode {box_mode}")

    def __rotation_3d_in_axis(self,
                              points,
                              angles,
                              axis: int = 0):
        """
        MODIFIED from MMDETECTION3D https://mmdetection3d.readthedocs.io/en/latest/
        Rotate points by angles according to axis.

        Args:
            points (np.ndarray or Tensor): Points with shape (N, M, 3).
            angles (np.ndarray or Tensor or float): Vector of angles with shape (N, ).
            axis (int): The axis to be rotated. Defaults to 0.

        Returns:
            Rotated points with shape (N, M, 3) and rotation matrix with
            shape (N, 3, 3).
        """
        rot_sin = torch.sin(angles)
        rot_cos = torch.cos(angles)
        ones = torch.ones_like(rot_cos)
        zeros = torch.zeros_like(rot_cos)

        if axis == 1:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos])
            ])
        elif axis == 2:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, rot_sin, zeros]),
                torch.stack([-rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones])
            ])
        else:
            raise ValueError(
                f'axis should in range [1, 2], got {axis}')

        if points.shape[0] == 0:
            points_new = points
        else:
            points_new = torch.einsum('aij,jka->aik', points, rot_mat_T)
        return points_new

    def __camera_boxes_to_corners(self, boxes_3d):
        """
        MODIFIED from MMDETECTION3D https://mmdetection3d.readthedocs.io/en/latest/
        Convert boxes to corners in clockwise order, in the form of (x0y0z0,
        x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

        .. code-block:: none

                         front z
                              /
                             /
               (x0, y0, z1) + -----------  + (x1, y0, z1)
                           /|            / |
                          / |           /  |
            (x0, y0, z0) + ----------- +   + (x1, y1, z1)
                         |  /      .   |  /
                         | / origin    | /
            (x0, y1, z0) + ----------- + -------> right x
                         |             (x1, y1, z0)
                         |
                         v
                    down y

        Returns:
            Tensor: A tensor with 8 corners of each box in shape (N, 8, 3).
        """
        if boxes_3d.numel() == 0:
            return torch.empty([0, 8, 3], device=boxes_3d.device)

        dims = boxes_3d[:, 3:6]
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin (0.5, 1, 0.5)
        corners_norm = corners_norm - dims.new_tensor([0.5, 1, 0.5])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        corners = self.__rotation_3d_in_axis(corners, boxes_3d[:, 6], axis=1)
        corners += boxes_3d[:, :3].view(-1, 1, 3)
        return corners

    def __lidar_boxes_to_corners(self, boxes_3d):
        """
        MODIFIED from MMDETECTION3D https://mmdetection3d.readthedocs.io/en/latest/
        Source reference: mmdetection3d/mmdet3d/structures/bbox_3d/lidar_box3d.py
        """
        if boxes_3d.numel() == 0:
            return torch.empty([0, 8, 3], device=boxes_3d.device)

        dims = boxes_3d[:, 3:6]
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        corners = self.__rotation_3d_in_axis(corners, boxes_3d[:, 6], axis=2)
        corners += boxes_3d[:, :3].view(-1, 1, 3)
        return corners

    def __camera_to_lidar_boxes(self, boxes_3d, rt_mat):
        """
        MODIFIED from MMDETECTION3D https://mmdetection3d.readthedocs.io/en/latest/
        Source reference: mmdetection3d/mmdet3d/structures/bbox_3d/box_3d_mode.py
        """
        x_size = boxes_3d[..., 3:4]
        y_size = boxes_3d[..., 4:5]
        z_size = boxes_3d[..., 5:6]
        yaw = boxes_3d[..., 6:7]
        xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)

        if not isinstance(rt_mat, torch.Tensor):
            rt_mat = boxes_3d.new_tensor(rt_mat)
        if rt_mat.size(1) == 4:
            extended_xyz = torch.cat(
                [boxes_3d[..., :3], boxes_3d.new_ones(boxes_3d.size(0), 1)], dim=-1)
            xyz = extended_xyz @ rt_mat.t()
        else:
            xyz = boxes_3d[..., :3] @ rt_mat.t()

        yaw = -yaw - np.pi / 2
        yaw = self.__limit_period(yaw, period=np.pi * 2)
        remains = boxes_3d[..., 7:]
        return torch.cat([xyz[..., :3], xyz_size, yaw, remains], dim=-1)

    def __lidar_to_camera_boxes(self, boxes_3d, rt_mat):
        """
        MODIFIED from MMDETECTION3D https://mmdetection3d.readthedocs.io/en/latest/
        Source reference: mmdetection3d/mmdet3d/structures/bbox_3d/box_3d_mode.py
        """
        x_size = boxes_3d[..., 3:4]
        y_size = boxes_3d[..., 4:5]
        z_size = boxes_3d[..., 5:6]
        yaw = boxes_3d[..., 6:7]
        xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)

        if not isinstance(rt_mat, torch.Tensor):
            rt_mat = boxes_3d.new_tensor(rt_mat)
        if rt_mat.size(1) == 4:
            extended_xyz = torch.cat(
                [boxes_3d[..., :3], boxes_3d.new_ones(boxes_3d.size(0), 1)], dim=-1)
            xyz = extended_xyz @ rt_mat.t()
        else:
            xyz = boxes_3d[..., :3] @ rt_mat.t()

        yaw = -yaw - np.pi / 2
        yaw = self.__limit_period(yaw, period=np.pi * 2)
        remains = boxes_3d[..., 7:]
        return torch.cat([xyz[..., :3], xyz_size, yaw, remains], dim=-1)

    def __limit_period(self, val, period=np.pi, offset=0.5):
        """
        MODIFIED from MMDETECTION3D https://mmdetection3d.readthedocs.io/en/latest/
        Source reference: mmdetection3d/mmdet3d/structures/bbox_3d/box_3d_mode.py
        """
        return val - torch.floor(val / period + offset) * period
