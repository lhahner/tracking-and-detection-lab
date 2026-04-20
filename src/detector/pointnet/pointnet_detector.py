from __future__ import annotations

from pathlib import Path

import torch

from datasets.kitti3D import Kitti3D
from detector.detector import Detector
from detector.pointnet.infer import build_model, load_checkpoint, predict_crops
from detector.pointnet.postprocess import (
    attach_predictions,
    non_max_suppression_bev,
    save_kitti_like_detections,
)
from detector.pointnet.preprocess import prepare_crop
from detector.pointnet.proposals import generate_proposals
from util.logging_config import LoggingConfig

logging_config = LoggingConfig()
logger = logging_config.get_logger(__name__)


class PointNetDetector(Detector):
    """PointNet-based LiDAR detector using proposal clustering plus crop classification."""

    def __init__(
        self,
        input_path,
        output_path,
        checkpoint_path,
        num_classes=4,
        num_points=1024,
        use_intensity=True,
        score_threshold=0.5,
        device=None,
    ):
        self.input_path = input_path
        self.output_path = Path(output_path)
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.num_points = num_points
        self.use_intensity = use_intensity
        self.score_threshold = score_threshold
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        input_channels = 4 if use_intensity else 3
        self.model = build_model(num_classes=num_classes, input_channels=input_channels)
        self.model = load_checkpoint(self.model, checkpoint_path, self.device)
        self.class_names = {
            0: "Background",
            1: "Pedestrian",
            2: "Cyclist",
            3: "Car",
        }
        if num_classes > len(self.class_names):
            raise ValueError("Number of classes exceeds class name list.")

    def read_data(self, input_directory):
        return Kitti3D(
            data_root=input_directory,
            split="testing",
            mode="frame",
            num_points=self.num_points,
            logger=logger,
        )

    def detect_points(self, points):
        if points is None or points.size == 0:
            return []

        proposals = generate_proposals(points)
        if not proposals:
            return []

        crop_tensors = [
            prepare_crop(
                proposal["points"], self.num_points, use_intensity=self.use_intensity
            )
            for proposal in proposals
        ]

        batch = torch.stack(crop_tensors, dim=0)
        labels, scores, _ = predict_crops(self.model, batch, self.device)

        detections = attach_predictions(
            proposals,
            labels,
            scores,
            self.class_names,
            self.score_threshold,
        )
        return non_max_suppression_bev(detections)

    def detect(self):
        dataset = self.read_data(self.input_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        for sample in dataset:
            detections = self.detect_points(sample["points"])
            save_kitti_like_detections(
                self.output_path / f"{sample['sample_id']}.txt", detections
            )
