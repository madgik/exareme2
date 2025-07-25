from typing import List

import numpy as np

from exareme2.aggregation_clients import AggregationType
from exareme2.aggregation_clients import BaseAggregationClient
from exareme2.algorithms.exaflow.exaflow_udf_aggregation_client_interface import (
    ExaflowUDFAggregationClientI,
)


class ExaflowUDFAggregationClient(BaseAggregationClient, ExaflowUDFAggregationClientI):
    def aggregate(
        self, aggregation_type: AggregationType, values: List[float]
    ) -> List[float]:
        original = np.asarray(values)
        flat = original.ravel().tolist()

        result = self._aggregate_request(aggregation_type, flat)

        if original.ndim > 1 and len(result) == original.size:
            return np.array(result).reshape(original.shape).tolist()

        return result
