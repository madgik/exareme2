from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from exareme2.aggregation_clients import BaseAggregationClient


class ControllerAggregationClientI(ABC, BaseAggregationClient):
    @abstractmethod
    def configure(self, num_workers: int) -> str:
        ...

    @abstractmethod
    def cleanup(self) -> str:
        ...
