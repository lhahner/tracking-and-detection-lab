from abc import ABC, abstractmethod

class Detector(ABC):
   """Define the detector interface used by the application."""

   @abstractmethod
   def detect(self):
       """Run detection on the configured input source."""
       pass
       
   @abstractmethod
   def read_data(self, input_directory):
       """Load input data required by a detector implementation.

       Args:
           input_directory: Directory that contains the detector input data.
       """
       pass
