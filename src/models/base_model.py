from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def architecture(self):
        pass

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def infer(self, input_data):
        pass
