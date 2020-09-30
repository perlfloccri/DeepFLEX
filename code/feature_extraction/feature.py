from abc import ABCMeta, abstractmethod
import numpy as np


class Feature(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        self._name = ''

        pass

    @abstractmethod
    def get_name(self, m_type: int) -> str:
        return self._name

    @abstractmethod
    def measure(self, m_type: int) -> float:
        value = 0
        return value

        pass