from abc import ABC
from abc import abstractmethod
from typing import Sequence
from typing import Union

import numpy as np

from exaflow.aggregation_clients import AggregationType

ArrayInput = Union[
    Sequence[float],
    Sequence[Sequence[float]],
    np.ndarray,
]


class ExaflowUDFAggregationClientI(ABC):
    @abstractmethod
    def aggregate(
        self, aggregation_type: AggregationType, values: ArrayInput
    ) -> np.ndarray: ...

    def sum(self, values: ArrayInput) -> np.ndarray:
        return self.aggregate(AggregationType.SUM, values)

    def min(self, values: ArrayInput) -> np.ndarray:
        return self.aggregate(AggregationType.MIN, values)

    def max(self, values: ArrayInput) -> np.ndarray:
        return self.aggregate(AggregationType.MAX, values)
