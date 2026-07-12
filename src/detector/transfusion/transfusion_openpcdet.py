import numpy as np
if not hasattr(np, "int"):
    np.int = int
import torch
try:
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils
    from detector.openpcdet_config import load_openpcdet_config
except ImportError as exc:
    raise ImportError("TransfusionOpenPCDet requires MMDetection3D. Install the OpenMMLab stack first.") from exc

from torch.utils.data import DataLoader
from pathlib import Path
from detector.detector import Detector
from detector.detector_registry import MODELS
from entities.detection import Detection, DetectionSequence, FrameDetection
from definitions import ROOT_DIR
from config.logging_config import LoggingConfig
from easydict import EasyDict

PROJECT_DIR = Path(__file__).resolve().parent

logging_config = LoggingConfig()
logger = logging_config.get_logger(__name__)


@MODELS.register("transfusion_openpcdet")
class TransfusionOpenPCDet(Detector):
    def __init__(self,
                 dataset,
                 classes,
                 settings,
                 config_file,
                 checkpoint_file,
                 batch_size=16):
        self.dataset = dataset
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.classes = classes
        self.batch_size = batch_size
        self.settings = settings
        self.cfg = load_openpcdet_config(self.config_file)
        self.model = build_network(
            self.cfg.MODEL,
            num_class=len(self.dataset.class_names),
            dataset=self.dataset
        )
        checkpoint = torch.load(self.checkpoint_file, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            self.model.load_params_from_file(filename=self.checkpoint_file,
                                             logger=logger,
                                             to_cpu=True)
        else:
            load_result = self.model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
            if len(load_result.missing_keys) == len(self.model.state_dict()):
                raise ValueError(f"Checkpoint {self.checkpoint_file} is not compatible with OpenPCDet SECOND")
        self.model.cuda()
        self.model.eval()
        self.class_map = self.__build_class_map(self.classes)

    def detect(self):
        detection_sequence = DetectionSequence()
        with torch.no_grad():
            sample_count = getattr(self.dataset, "max_samples", None) or len(self.dataset)
            for idx in range(sample_count):
                sample = self.dataset[idx]
                target = sample.pop("target", [])
                metadata = sample.pop("metadata", None)
                data_dict = self.dataset.collate_batch([sample])
                sample_id = data_dict["frame_id"][0]
                load_data_to_gpu(data_dict)
                instance_data, _ = self.model.forward(data_dict)
                detections, highest_score_index = self.__convert_instances(instance_data[0])
                detection_sequence.frames.append(FrameDetection(frame=sample_id,
                                                            highest_score_index=highest_score_index,
                                                            dets=detections,
                                                            targets=target,
                                                            metadata=metadata))
        return detection_sequence

    def __build_class_map(self, classes):
        if not isinstance(classes, dict):
            if classes and isinstance(classes[0], str):
                return torch.tensor([classes.index(name) + 1 for name in self.dataset.class_names], dtype=torch.long)
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
        boxes = instance_data["pred_boxes"].detach().cpu()
        scores = instance_data["pred_scores"].detach().cpu()
        labels = self.__map_labels(instance_data["pred_labels"].detach().cpu() - 1)
        highest_score_index = int(torch.argmax(scores).item()) if scores.numel() else -1
        detections = [
            Detection(score=float(score.item()), label=int(label.item()), box=box[:7])
            for box, score, label in zip(boxes, scores, labels)
        ]
        return detections, highest_score_index

    def __map_labels(self, labels_reference):
        if self.class_map.numel() == 0:
            return labels_reference
        valid = (labels_reference >= 0) & (labels_reference < self.class_map.numel())
        if not torch.all(valid):
            raise ValueError("MMDetection3D CenterPoint returned a class index outside"
                             "the configured label map")
        return self.class_map[labels_reference]
