from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class ControllerAggregationClientI(ABC):
    @abstractmethod
    def configure(self, num_workers: int) -> str:
        ...

    @abstractmethod
    def cleanup(self) -> str:
        ...
