from abc import ABC
from itertools import chain
from typing import List


class NumpyAggregationServer(ABC):
    """Provides basic aggregation operations on lists of numeric responses."""

    @staticmethod
    def sum(responses: List[List[float]]) -> List[float]:
        """
        Compute the element-wise sum if all responses have the same length;
        otherwise, compute the sum over all elements.
        """
        if not responses:
            raise ValueError("No responses provided for aggregation.")
        lengths = [len(r) for r in responses]
        if all(l == lengths[0] for l in lengths):
            return [sum(x) for x in zip(*responses)]
        else:
            return [sum(chain(*responses))]

    @staticmethod
    def min(responses: List[List[float]]) -> List[float]:
        """
        Compute the element-wise minimum if all responses have the same length;
        otherwise, compute the minimum over all elements.
        """
        if not responses:
            raise ValueError("No responses provided for aggregation.")
        lengths = [len(r) for r in responses]
        if all(l == lengths[0] for l in lengths):
            return [min(x) for x in zip(*responses)]
        else:
            return [min(chain(*responses))]

    @staticmethod
    def max(responses: List[List[float]]) -> List[float]:
        """
        Compute the element-wise maximum if all responses have the same length;
        otherwise, compute the maximum over all elements.
        """
        if not responses:
            raise ValueError("No responses provided for aggregation.")
        lengths = [len(r) for r in responses]
        if all(l == lengths[0] for l in lengths):
            return [max(x) for x in zip(*responses)]
        else:
            return [max(chain(*responses))]
