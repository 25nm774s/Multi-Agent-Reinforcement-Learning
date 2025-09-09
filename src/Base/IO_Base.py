from abc import ABC, abstractmethod

class BaseModelIO(ABC):
    @abstractmethod
    def save(self, file_path, data):
        pass

    @abstractmethod
    def load(self, file_path):
        pass

class BaseCheckpointHandlar(ABC):
    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
