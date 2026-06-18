import numpy as np
import torch
try:
    from mmdet3d.apis import inference_detector, init_model
except ImportError as exc:
    raise ImportError("CenterPointMMDetections3D requires MMDetection3D. Install the OpenMMLab stack first.") from exc

from torch.utils.data import DataLoader
from pathlib import Path
from detector.detector import Detector
from entities.detection import Detection, DetectionSequence, FrameDetection
from definitions import ROOT_DIR
from config.logging_config import LoggingConfig

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = (
    Path(ROOT_DIR)
    / "mmdetection3d"
    / "configs"
    / "centerpoint"
    / "centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
)
DEFAULT_CHECKPOINT_FILE = (
        PROJECT_DIR
        / "model"
        / "centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_030004-9061688e.pth"
)
logging_config = LoggingConfig()
logger = logging_config.get_logger(__name__)


class CenterPointMMDetections3D(Detector):
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
        self.batch_size = batch_size
        self.settings = settings
        self.model = init_model(
            str(self.config_file),
            str(self.checkpoint_file)
        )
        self.class_map = self.__build_class_map(self.classes)

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

    def __build_class_map(self, classes):
        if not isinstance(classes, dict):
            return torch.as_tensor(classes, dtype=torch.long)

        model_classes = None
        if hasattr(self.model, "dataset_meta"):
            model_classes = self.model.dataset_meta.get("classes")
        if model_classes is None and hasattr(self.model, "cfg"):
            model_classes = self.model.cfg.get("class_names", None)
        if model_classes is None:
            raise ValueError("Unable to determine MMDetection3D CenterPoint class order")

        try:
            return torch.tensor([classes[name] for name in model_classes], dtype=torch.long)
        except KeyError as exc:
            raise ValueError(f"MMDetection3D class {exc.args[0]!r} is not present in the project label map") from exc

    def __predict_instances(self, point):
        result = inference_detector(self.model, self.__prepare_point_input(point))
        while isinstance(result, (list, tuple)) and len(result) > 0:
            result = result[0]
        if hasattr(result, "pred_instances_3d"):
            return result.pred_instances_3d
        raise ValueError("MMDetection3D inference result does not contain pred_instances_3d")

    def __prepare_point_input(self, point):
        if isinstance(point, torch.Tensor):
            point = point.detach().cpu().numpy()
        elif not isinstance(point, np.ndarray):
            point = np.asarray(point)
        point = point.astype(np.float32, copy=False)
        if point.ndim != 2:
            raise ValueError(f"Expected point cloud with shape [N, C], got {point.shape}")
        if point.shape[1] == 4:
            time_lag = np.zeros((point.shape[0], 1), dtype=point.dtype)
            return np.concatenate([point, time_lag], axis=1)
        if point.shape[1] < 4:
            raise ValueError(f"Expected at least four point features [x, y, z, intensity], got {point.shape[1]}")
        return point

    def __convert_instances(self, instance_data):
        scores = instance_data.scores_3d.detach().cpu().float()
        bboxes_3d = instance_data.bboxes_3d
        bbox_tensor = bboxes_3d.tensor if hasattr(bboxes_3d, "tensor") else bboxes_3d
        bboxes = bbox_tensor.detach().cpu().float()
        if bboxes.ndim != 2 or bboxes.shape[1] < 7:
            raise ValueError(f"Expected MMDetection3D boxes with shape [N, >=7], got {tuple(bboxes.shape)}")
        bboxes = bboxes[:, :7].clone()
        bboxes[:, 2] += bboxes[:, 5] / 2 
        labels_reference = instance_data.labels_3d.detach().cpu().long()
        if bboxes.shape[0] == 0:
            return [], None
        labels = self.__map_labels(labels_reference)
        highest_score_index = int(scores.argmax().item())
        return [
            Detection(score=float(score.item()), label=int(label.item()), box=box)
            for score, box, label in zip(scores, bboxes, labels)
        ], highest_score_index

    def __map_labels(self, labels_reference):
        if self.class_map.numel() == 0:
            return labels_reference
        valid = (labels_reference >= 0) & (labels_reference < self.class_map.numel())
        if not torch.all(valid):
            raise ValueError("MMDetection3D CenterPoint returned a class index outside"
                             "the configured label map")
        return self.class_map[labels_reference]
