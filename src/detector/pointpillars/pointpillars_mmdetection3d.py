from pathlib import Path

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
        self.label_map = self.__build_label_map(classes)
        self.init_model, self.inference_detector = self.__load_mmdetection3d_apis()
        self.model = self.init_model(
            self.config_file,
            self.checkpoint_file if self.checkpoint_file else None,
            device=self.device,
        )

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
        detections = [
            Detection(score=score, label=label, box=box)
            for score, box, label in zip(scores, bboxes, labels)
        ]
        return detections, highest_score_index

    def __build_label_map(self, classes):
        if classes is None:
            return NUSCENES_LABELS
        if isinstance(classes, dict):
            ordered = [class_id for name, class_id in classes.items() if name != "Background"]
            return torch.tensor(ordered, dtype=torch.int64)
        return torch.as_tensor(classes, dtype=torch.int64)

    def __map_labels(self, labels_reference):
        if self.label_map.numel() == 0:
            return labels_reference
        valid = labels_reference < self.label_map.numel()
        if not torch.all(valid):
            raise ValueError(
                "MMDetection3D PointPillars returned a class index outside "
                "the configured label map"
            )
        return self.label_map[labels_reference]
