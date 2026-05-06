import torch
import sys
import os
import math

from detector.detector import Detector
from torch.utils.data import DataLoader
from util.file_handler import write_output

if torch.cuda.is_available():
    from mmdet3d.apis import init_model, inteference_detector
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
    SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
    MMDET3D_SRC_ROOT = os.path.join(PROJECT_ROOT, "external", "mmdetection3d-cpu-only")
    if SRC_ROOT not in sys.path:
        sys.path.insert(0, SRC_ROOT)
    if MMDET3D_SRC_ROOT not in sys.path:
        sys.path.insert(0, MMDET3D_SRC_ROOT)
    from mmdet3d.apis import init_model, inteference_detector

from util.logging_config import LoggingConfig

logging_config = LoggingConfig()
logger = logging_config.get_logger(__name__)


class PointRCNNmmDetections3D(Detector):
    def __init__(self, dataset_dir, dataset, model_path, batch_size=16):
        self.dataset_dir = dataset_dir
        self.dataset = dataset
        self.model_path = model_path
        self.config_file = ''
        self.checkpoint_file = ''
        self.model = init_model(self.config_file, self.checkpoint_file)

    @staticmethod
    def custom_collate(batch):
        filtered_data = []
        filtered_samples = []
        for item in batch:
            filtered_data.append(item["points"])
            filtered_samples.append(item["sample_id"])
        return filtered_data, filtered_samples

    def detect(self):
        test_dataloader = DataLoader(dataset=self.dataset, batch_size=16,\
                                     collate_fn=custom_collate)
        formatted_detections = []
        for point, sample in test_dataloader:
            detections = inference_detector(self.model, point)
            formatted_detections = self.format_detections(sample, detections)
        write_output(detections)
        return detections

    def format_detections(self, frame_index, prev_results, results):
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
        xyz_centroids = \
                results.gt_instances_3d().instance_data[instance_data.bboxes][0]
        lwh_box = results.gt_instances_3d().instance_data[instance_data.bboxes][1]
        rotation_box = results.gt_instances_3d().instance_data[instance_data.bboxes][2]
        det_scores = results.gt_instances_3d().intsance_data[instance_data.det_scores]
        yaw = rotation_box[2]
        
        rotation_matrix = torch.stack(torch.stack([torch.cos(yaw), -torch.sin(yaw), 0]),
                                      torch.stack([torch.sin(yaw), torch.cos(yaw)], 0),
                                      torch.stack([0, 0, 1]))
        box_corner = torch.stack(lwh_box[0]/2, lwh_box[1]/2, lwh_box[2]/2)
        world_bbox = xyz_centroids + rotation_matrix @ box_corner
        return world_bbox

