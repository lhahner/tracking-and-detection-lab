from abc import ABC, abstractmethod

class Detector(ABC):
   @abstractmethod
   def detect(self):
       pass
       
   @abstractmethod
   def read_data(self, input_directory):
       pass