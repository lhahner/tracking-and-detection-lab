from enum import Enum

class Detector(Enum):
    """Enumerate the detector backends supported by the project."""

    POINTPILLARS="pointpillars",
    YOLO="yolo",
    FRCNN="frcnn"
