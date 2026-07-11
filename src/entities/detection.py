from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
from entities.metadata import Metadata

@dataclass(frozen=True)
class Detection:
    score: float
    label: int
    box: torch.tensor


@dataclass(frozen=True)
class FrameDetection:
    frame: int
    highest_score_index: int
    dets: list[Detection]
    targets: list = field(default_factory=list)
    metadata: Metadata | None = None


@dataclass(frozen=True)
class DetectionSequence:
    frames: list[FrameDetection] = field(default_factory=list)  # makes frames optional


def convert_to_tensor(detections):
    tmp_list_to_be_stacked = []
    if len(detections) == 0:
        return torch.empty((0, 9))
    for detection in detections:
        tmp_tensor = F.pad(detection.box, (0, 2))
        tmp_tensor[-2] = detection.score
        tmp_tensor[-1] = detection.label
        tmp_list_to_be_stacked.append(tmp_tensor)
    return torch.stack(tmp_list_to_be_stacked)


def convert_classes_to_tensor(classes):
    if isinstance(classes, dict):
        return torch.tensor([
            class_id for name, class_id in classes.items()
            if name != "Background"
        ])
    return torch.as_tensor(classes)
