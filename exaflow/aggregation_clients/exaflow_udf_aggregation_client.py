import numpy as np

from exaflow.aggregation_clients import AggregationType
from exaflow.aggregation_clients import BaseAggregationClient
from exaflow.algorithms.exaflow.exaflow_udf_aggregation_client_interface import (
    ArrayInput,
)
from exaflow.algorithms.exaflow.exaflow_udf_aggregation_client_interface import (
    ExaflowUDFAggregationClientI,
)


class ExaflowUDFAggregationClient(BaseAggregationClient, ExaflowUDFAggregationClientI):
    def aggregate(
        self, aggregation_type: AggregationType, values: ArrayInput
    ) -> np.ndarray:
        original = np.asarray(values)
        flat = original.ravel()

        aggregated = np.asarray(
            self._aggregate_request(aggregation_type, flat.tolist()), dtype=float
        )

        if original.shape and aggregated.size == original.size:
            return aggregated.reshape(original.shape)

        return aggregated
