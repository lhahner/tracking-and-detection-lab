from abc import ABC, abstractmethod

from entities.detection import DetectionSequence


class Detector(ABC):
    """Define the detector interface used by the
       application."""
    @abstractmethod
    def detect(self) -> DetectionSequence:
        """
        Run detection on the configured input source.
        """
    pass
