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
    """
    Minimal interface passed into UDFs when `with_aggregation_server=True`.

    Implementations must perform secure aggregation across workers according to
    the requested `AggregationType` (SUM/MIN/MAX). UDFs should pass plain numpy
    arrays or array-likes; the implementation is responsible for returning a
    numpy array of the aggregated result.
    """

    @abstractmethod
    def aggregate(
        self, aggregation_type: AggregationType, values: ArrayInput
    ) -> np.ndarray: ...

    @abstractmethod
    def unregister(self) -> tuple[str, int]: ...

    def sum(self, values: ArrayInput) -> np.ndarray:
        return self.aggregate(AggregationType.SUM, values)

    def min(self, values: ArrayInput) -> np.ndarray:
        return self.aggregate(AggregationType.MIN, values)

    def max(self, values: ArrayInput) -> np.ndarray:
        return self.aggregate(AggregationType.MAX, values)
