import torch
import sys
import os

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

    def format_detections(self, frame_index, results):
        # https://mmengine.readthedocs.io/en/v0.10.0/api/generated/mmengine.structures.InstanceData.html  
        xyz_centroids = results.gt_instances_3d().instance_data[instance_data.bboxes]
