import torch
import numpy as np
from detector.detector import Detector
from torch.utils.data import DataLoader
from util.file_handler import write_output
from pytorch3d.structures import Pointclouds

if torch.cuda.is_available():
    from mmdet3d.apis import init_model, inference_detector
    from mmdet3d.structures.bbox_3d import Box3DMode
else:
    exit()
from util.logging_config import LoggingConfig

logging_config = LoggingConfig()
logger = logging_config.get_logger(__name__)


class PointRCNNmmDetections3D(Detector):
    def __init__(self, dataset, config_file, checkpoint_file, batch_size=16):
        self.dataset = dataset
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.model = init_model(self.config_file, self.checkpoint_file)

    def detect(self):
        test_dataloader = DataLoader(dataset=self.dataset, batch_size=16,
                                     collate_fn=custom_collate)
        formatted_detections = []
        for point, sample in test_dataloader:
            detections = inference_detector(self.model, point)[0].detections
            bboxes = detections.pred_instances_3d.bboxes3d
            scores = detections.pred_instances_3d.scores_3d
            formatted_detections.append(self.format_detections(sample,
                                                               bboxes.cpu()[:, :3],
                                                               bboxes.cpu()[:, 3:5],
                                                               bboxes.cpu()[:, 6],
                                                               scores.cpu()
                                                               ))
        write_output(formatted_detections)
        return detections

    def format_detections(self, frame_index: int, xyz_centroids: np.array, lwh_box: np.array,
                          yaw: torch.tensor, det_score: float):
        """
        The tracking system requires the following dimensions
        - x_1: Left boundary coordinate
        - y_1: Top boundary coordinate
        - z_1: Top Left boundary coordinate
        - x_2: Right boundary coordinate
        - y_2: Bottom boundary coordinate
        - z_3: Bottom Right boundary coordinate
        - score: Detection confidence score
        """
        xyz_centroids = Box3DMode.convert(xyz_centroids, Box3DMode.LIDAR, Box3DMode.CAM)
        # Rotation matrix based on yaw at corner z in local coordiantes
        rotation_matrix: torch.tensor = torch.tensor([
                [torch.cos(yaw), torch.sin(yaw), 0],
                [torch.sin(yaw), torch.cos(yaw), 0],
                [0,                     0,                     1]
            ])
        # compute per bounding each corner of every coodinate,
        # 8 corners meaning 8 cooridnates
        local_box_corner: torch.tensor = torch.tensor([
            [lwh_box[:, 0]/2, lwh_box[:, 1]/2, lwh_box[:, 2]/2],  # corner x left front
            [lwh_box[:, 0]/2, -lwh_box[:, 1]/2, lwh_box[:, 2]/2],  # corner x right front
            [-lwh_box[:, 0]/2, -lwh_box[:, 1]/2, lwh_box[:, 2]/2],  # corner x left back
            [-lwh_box[:, 0]/2, lwh_box[:, 1]/2, lwh_box[:, 2]/2],  # corner x right back
            [lwh_box[:, 0]/2, lwh_box[:, 1]/2, -lwh_box[:, 2]/2],  # corner y left front
            [lwh_box[:, 0]/2, -lwh_box[:, 1]/2, -lwh_box[:, 2]/2],  # corner y right front
            [-lwh_box[:, 0]/2, -lwh_box[:, 1]/2, -lwh_box[:, 2]/2],  # corner y left back
            [-lwh_box[:, 0]/2, lwh_box[:, 1]/2, -lwh_box[:, 2]/2]  # corner y right back
        ])
        # The coordiantes of the bounding box in world coordinates
        world_bbox: torch.tensor = xyz_centroids + local_box_corner @ rotation_matrix
        world_bbox = world_bbox.unsqueeze(0)
        breakpoint()
        bboxes: torch.tensor = Pointclouds(points=world_bbox).get_bounding_boxes()
        return f"{frame_index},{bboxes.flatten()},{det_score},1,-1,-1,-1"


@staticmethod
def custom_collate(batch):
    filtered_data = []
    filtered_samples = []
    for item in batch:
        filtered_data.append(item["points"])
        filtered_samples.append(item["sample_id"])
    return filtered_data, filtered_samples
