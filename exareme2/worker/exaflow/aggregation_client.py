from typing import List

import numpy as np

from exareme2.aggregation_client import AggregationType
from exareme2.aggregation_client.base_aggregation_client import BaseAggregationClient
from exareme2.algorithms.exaflow.algorithm_udf_aggregation_client import (
    AlgorithmUdfAggregationClient,
)


class AlgorithmUdfWorkerAggregationClient(
    BaseAggregationClient, AlgorithmUdfAggregationClient
):
    """Used by workers only."""

    def aggregate(
        self, aggregation_type: AggregationType, values: List[float]
    ) -> List[float]:
        original = np.asarray(values)
        flat = original.ravel().tolist()

        result = self._aggregate_request(aggregation_type, flat)

        if original.ndim > 1 and len(result) == original.size:
            return np.array(result).reshape(original.shape).tolist()

        return result
