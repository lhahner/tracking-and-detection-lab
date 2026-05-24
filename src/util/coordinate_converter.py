import numpy as np
import torch

class CoordinateConverter:
    """Convert detections between the project's supported box formats."""

    def __init__(self):
        """Initialize a coordinate converter for detection arrays."""
        pass
    
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

    def boxes_3d_to_corners(self, boxes_3d):
        if boxes_3d.ndim != 2 or boxes_3d.shape[1] != 7:
            raise ValueError(f"Expected boxes_3d shape [N, 7], got {tuple(boxes_3d.shape)}")

        device = boxes_3d.device
        dtype = boxes_3d.dtype

        centers = boxes_3d[:, 0:3]
        dims = boxes_3d[:, 3:6]
        yaws = boxes_3d[:, 6]

        dx = dims[:, 0]
        dy = dims[:, 1]
        dz = dims[:, 2]

        x_corners = torch.stack(
            [
                dx / 2, dx / 2, -dx / 2, -dx / 2,
                dx / 2, dx / 2, -dx / 2, -dx / 2,
            ],
            dim=1,
        )
        y_corners = torch.stack(
            [
                dy / 2, -dy / 2, -dy / 2, dy / 2,
                dy / 2, -dy / 2, -dy / 2, dy / 2,
            ],
            dim=1,
        )
        z_corners = torch.stack(
            [
                dz / 2, dz / 2, dz / 2, dz / 2,
                -dz / 2, -dz / 2, -dz / 2, -dz / 2,
            ],
            dim=1,
        )

        local_corners = torch.stack(
            [x_corners, y_corners, z_corners],
            dim=-1,
        )
        cos_yaw = torch.cos(yaws)
        sin_yaw = torch.sin(yaws)

        rotation_matrices = torch.zeros((boxes_3d.shape[0], 3, 3), device=device, dtype=dtype)

        rotation_matrices[:, 0, 0] = cos_yaw
        rotation_matrices[:, 0, 1] = -sin_yaw
        rotation_matrices[:, 1, 0] = sin_yaw
        rotation_matrices[:, 1, 1] = cos_yaw
        rotation_matrices[:, 2, 2] = 1.0

        rotated_corners = torch.bmm(
            local_corners,
            rotation_matrices.transpose(1, 2),
        )
        corners = rotated_corners + centers[:, None, :]
        return corners
