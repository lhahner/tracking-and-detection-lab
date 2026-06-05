from abc import ABC, abstractmethod


class Detector(ABC):
    """Define the detector interface used by the
       application."""
    @abstractmethod
    def detect(self):
        """
        Run detection on the configured input source.
        """
    pass
