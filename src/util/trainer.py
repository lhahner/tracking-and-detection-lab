from abc import ABC, abstractmethod

class Trainer(ABC):
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass