from abc import ABC, abstractmethod
import numpy as np

from spectrophane.core.dataclasses import StackCandidates

class Inverter(ABC):

    @abstractmethod
    def invert_xyz(xyz: np.ndarray) -> StackCandidates:
        ...