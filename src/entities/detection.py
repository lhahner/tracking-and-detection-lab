from dataclasses import dataclass
import torch


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


@dataclass(frozen=True)
class DetectionSequence:
    frames: list[FrameDetection] = []
