if not torch.cuda.is_available():
    raise EnvironmentError("CENTERPOINT MMDetection3D requested CUDA, but no CUDA device is available")
try:
    from mmdet3d.apis import inference_detector, init_model
except ImportError as exc:
    raise ImportError("CenterPointMMDetections3D requires MMDetection3D. Install the OpenMMLab stack first.") from exc

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from detector.detector import Detector
from entities.detection import Detection, DetectionSequence, FrameDetection
from definitions import ROOT_DIR
from utils.logging_config import LoggingConfig

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = ROOT_DIR / "mmdetection3d/configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
DEFAULT_CHECKPOINT_FILE = PROJECT_DIR / "model" / "centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth"

logging_config = LoggingConfig()
logger = logging_config.get_logger(__name__)


class CenterPointMMDetections3D(Detector):
    """MMDetection3D CENTERPOINT wrapper for project 3D detection pipelines."""

    def __init__(self, 
                 dataset, 
                 classes,
                 settings,
                 config_file=DEFAULT_CONFIG_FILE, 
                 checkpoint_file=DEFAULT_CHECKPOINT_FILE, 
                 batch_size=16):
        self.dataset = dataset
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.classes = classes
        if not torch.is_tensor(self.classes):
            raise ValueError("classes should be in format torch.tensor([1, 2, 3, ...])")
        self.batch_size = batch_size
        self.settings = settings
        self.model = self.init_model(
            self.config_file,
            self.checkpoint_file)

    def detect(self):
        test_dataloader = DataLoader(dataset=self.dataset, 
                                     batch_size=self.batch_size,
                                     collate_fn=self.dataset.custom_collate)
        detection_sequence = DetectionSequence()
        for points, targets, samples in test_dataloader:
            for point, target, sample_id in zip(points, targets, samples):
                instance_data = self.__predict_instances(point)
                detections, highest_score_index = self.__convert_instances(instance_data)
                detection_sequence.frames.append(FrameDetection(frame=sample_id, 
                                                                highest_score_index=highest_score_index,
                                                                dets=detections, 
                                                                targets=target))
        return detection_sequence

    def __load_mmdetection3d_apis(self):
        if not torch.cuda.is_available():
            raise EnvironmentError("CENTERPOINT MMDetection3D requested CUDA, but no CUDA device is available")
        try:
            from mmdet3d.apis import inference_detector, init_model
        except ImportError as exc:
            raise ImportError("CenterPointMMDetections3D requires MMDetection3D. Install the OpenMMLab stack first.") from exc
        return init_model, inference_detector

    def __predict_instances(self, point):
        result = self.inference_detector(self.model, point)
        while isinstance(result, (list, tuple)) and len(result) > 0:
            result = result[0]
        if hasattr(result, "pred_instances_3d"):
            return result.pred_instances_3d
        raise ValueError("MMDetection3D inference result does not contain pred_instances_3d")

    def __convert_instances(self, instance_data):
        scores = instance_data.scores_3d.detach().cpu().float()
        bboxes = instance_data.bboxes_3d.tensor.detach().cpu().float()
        labels_reference = instance_data.labels_3d.detach().cpu().long()
        if bboxes.shape[0] == 0:
            return [], None
        labels = self.__map_labels(labels_reference)
        highest_score_index = scores.argmax()
        return [Detection(score=score, label=label, box=box) for score, box, label in zip(scores, bboxes, labels)], highest_score_index

    def __map_labels(self, labels_reference):
        if self.classes.numel() == 0:
            return labels_reference
        valid = labels_reference <= self.classes.numel()
        if not torch.all(valid):
            raise ValueError("MMDetection3D CENTERPOINT returned a class index outside the configured label map")
        return self.classes[labels_reference]
