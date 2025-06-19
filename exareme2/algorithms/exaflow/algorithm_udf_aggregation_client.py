from abc import ABC
from abc import abstractmethod
from typing import List

from exareme2.aggregation_client.constants import AggregationType


class AlgorithmUdfAggregationClient(ABC):
    """What a worker cares about (data aggregation only)."""

    # low-level primitive
    @abstractmethod
    def aggregate(
        self, aggregation_type: AggregationType, values: List[float]
    ) -> List[float]:
        ...

    # convenience helpers ----------------------------------------------------
    # These *can* remain abstract, but providing defaults avoids repetition.
    def sum(self, values: List[float]) -> float:
        return self.aggregate(AggregationType.SUM, [sum(values)])

    def min(self, values: List[float]) -> float:
        return self.aggregate(AggregationType.MIN, [min(values)])

    def max(self, values: List[float]) -> float:
        return self.aggregate(AggregationType.MAX, [max(values)])
