from abc import ABC
from abc import abstractmethod
from typing import Iterable
from typing import List

from exareme2.aggregation_clients import AggregationType


class ExaflowUDFAggregationClientI(ABC):
    @abstractmethod
    def aggregate(
        self, aggregation_type: AggregationType, values: List[float]
    ) -> List[float]: ...

    def sum(self, values: List[float]) -> List[float]:
        return self.aggregate(AggregationType.SUM, [sum(values)])

    def min(self, values: List[float]) -> List[float]:
        return self.aggregate(AggregationType.MIN, [min(values)])

    def max(self, values: List[float]) -> List[float]:
        return self.aggregate(AggregationType.MAX, [max(values)])

    @abstractmethod
    def fed_weighted_avg(
        self, array: Iterable[float], weight: float
    ) -> Iterable[float]: ...
