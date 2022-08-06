from abc import ABC, abstractmethod

class Environment(ABC):
    """Abstract Environment class to define the API methods"""
    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError
