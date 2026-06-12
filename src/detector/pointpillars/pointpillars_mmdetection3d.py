from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from detector.detector import Detector
from entities.detection import Detection, DetectionSequence, FrameDetection

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[2]
DEFAULT_CONFIG_FILE = (
    REPO_ROOT
    / "mmdetection3d"
    / "configs"
    / "pointpillars"
    / "pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
)
DEFAULT_CHECKPOINT_FILE = (
    PROJECT_DIR
    / "model"
    / "hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
)
NUSCENES_LABELS = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64)


class PointPillarsMMDetections3D(Detector):
    """MMDetection3D PointPillars wrapper for nuScenes LiDAR detection.

    The class is intentionally symmetric to ``PointRCNNmmDetections3D``: it
    initializes an MMDetection3D model, iterates the project's dataset API, and
    emits the shared ``DetectionSequence`` / ``FrameDetection`` / ``Detection``
    objects used by the evaluation pipeline.
    """

    def __init__(
        self,
        dataset,
        config_file=None,
        classes=None,
        settings=None,
        checkpoint_file=None,
        batch_size=16,
        num_inference_samples=None,
        device=None,
    ):
        self.dataset = dataset
        self.config_file = str(config_file or DEFAULT_CONFIG_FILE)
        self.checkpoint_file = str(checkpoint_file or DEFAULT_CHECKPOINT_FILE)
        self.num_inference_samples = num_inference_samples
        self.classes = classes
        self.batch_size = batch_size
        self.settings = settings
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_model, self.inference_detector = self.__load_mmdetection3d_apis()
        self.model = self.init_model(
            self.config_file,
            self.checkpoint_file if self.checkpoint_file else None,
            device=self.device,
        )
        self.label_map = self.__build_label_map(classes)

    def detect(self):
        """Run MMDetection3D inference and return project detection objects."""
        test_dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.dataset.custom_collate,
        )
        detection_sequence = DetectionSequence()
        processed_samples = 0
        for points, targets, samples in test_dataloader:
            for point, target, sample_id in zip(points, targets, samples):
                if self.num_inference_samples is not None and processed_samples >= self.num_inference_samples:
                    return detection_sequence

                instance_data = self.__predict_instances(point)
                detections, highest_score_index = self.__convert_instances(instance_data)
                detection_sequence.frames.append(
                    FrameDetection(
                        frame=sample_id,
                        highest_score_index=highest_score_index,
                        dets=detections,
                        targets=target,
                    )
                )
                processed_samples += 1
        return detection_sequence

    def __load_mmdetection3d_apis(self):
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise EnvironmentError("PointPillars MMDetection3D requested CUDA, but no CUDA device is available")
        try:
            from mmdet3d.apis import inference_detector, init_model
        except ImportError as exc:
            raise ImportError(
                "PointPillarsMMDetections3D requires MMDetection3D. "
                "Install the OpenMMLab stack in the active environment first."
            ) from exc
        return init_model, inference_detector

    def __predict_instances(self, point):
        result = self.inference_detector(self.model, self.__prepare_point_input(point))
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
        labels_reference = instance_data.labels_3d.detach().cpu().long()
        if bboxes.numel() == 0:
            return [], None
        if bboxes.ndim != 2 or bboxes.shape[1] < 7:
            raise ValueError(f"Expected MMDetection3D boxes with shape [N, >=7], got {tuple(bboxes.shape)}")
        bboxes = bboxes[:, :7].clone()
        bboxes[:, 2] += bboxes[:, 5] / 2

        labels = self.__map_labels(labels_reference)
        highest_score_index = int(scores.argmax().item())
        detections = [
            Detection(score=float(score.item()), label=int(label.item()), box=box)
            for score, box, label in zip(scores, bboxes, labels)
        ]
        return detections, highest_score_index

    def __build_label_map(self, classes):
        if classes is None:
            return NUSCENES_LABELS
        if not isinstance(classes, dict):
            return torch.as_tensor(classes, dtype=torch.int64)

        model_classes = None
        if hasattr(self.model, "dataset_meta"):
            model_classes = self.model.dataset_meta.get("classes")
        if model_classes is None and hasattr(self.model, "cfg"):
            model_classes = self.model.cfg.get("class_names", None)
        if model_classes is None:
            raise ValueError("Unable to determine MMDetection3D PointPillars class order")

        try:
            return torch.tensor([classes[name] for name in model_classes], dtype=torch.int64)
        except KeyError as exc:
            raise ValueError(f"MMDetection3D class {exc.args[0]!r} is not present in the project label map") from exc

    def __map_labels(self, labels_reference):
        if self.label_map.numel() == 0:
            return labels_reference
        valid = (labels_reference >= 0) & (labels_reference < self.label_map.numel())
        if not torch.all(valid):
            raise ValueError(
                "MMDetection3D PointPillars returned a class index outside "
                "the configured label map"
            )
        return self.label_map[labels_reference]
