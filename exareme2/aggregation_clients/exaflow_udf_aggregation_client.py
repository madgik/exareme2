from typing import Iterable
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

    # ------------------------------------------------------------------
    # Extended aggregation helpers
    # ------------------------------------------------------------------
    def _as_float_list(self, values: Iterable[float]) -> List[float]:
        if isinstance(values, np.ndarray):
            return values.astype(float).ravel().tolist()
        return [float(v) for v in values]

    def _global_sum(self, values: Iterable[float]) -> np.ndarray:
        aggregated = self._aggregate_request(
            AggregationType.SUM, self._as_float_list(values)
        )
        return np.asarray(aggregated, dtype=float)

    def fed_weighted_avg(self, array: np.ndarray, weight: float) -> np.ndarray:
        """Compute federated weighted average of an array across all clients."""

        array = np.asarray(array, dtype=float)
        flat = array.ravel()
        payload = np.append(flat * weight, weight)
        aggregated = self._global_sum(payload)

        total_weight = aggregated[-1]
        if total_weight == 0:
            raise ValueError("Total weight for federated average is zero.")

        averaged = aggregated[:-1] / total_weight
        return averaged.reshape(array.shape)
