from abc import ABC, abstractmethod

class Environment(ABC):
    """Abstract Environment class to define the API methods"""
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self):
        pass
