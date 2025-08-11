from abc import ABC, abstractmethod

class BaseModelIO(ABC):
    @abstractmethod
    def save(self, agent_id: int, data):
        pass

    @abstractmethod
    def load(self, agent_id: int):
        pass

class BaseCheckpointHandlar(ABC):
    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
