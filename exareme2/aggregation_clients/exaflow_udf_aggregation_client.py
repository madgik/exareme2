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

    def fed_avg(self, array: np.ndarray) -> np.ndarray:
        """Compute the simple federated average of an array across all clients."""

        return self.fed_weighted_avg(array, 1.0)

    def fed_sum(self, array: np.ndarray) -> np.ndarray:
        """Compute the federated sum of an array across all clients."""

        array = np.asarray(array, dtype=float)
        aggregated = self._global_sum(array.ravel())
        return aggregated.reshape(array.shape)

    def global_sum(self, array: np.ndarray) -> np.ndarray:
        """Compute sum along axis=0 locally and then federated sum across clients."""

        reduced = np.asarray(array, dtype=float).sum(axis=0)
        shape = np.asarray(reduced).shape
        aggregated = self._global_sum(np.asarray(reduced, dtype=float).ravel())
        return aggregated.reshape(shape)

    def global_count(self, array: np.ndarray) -> np.ndarray:
        """Compute the federated count of samples across all clients."""

        local_count = np.asarray(array).shape[0]
        aggregated = self._global_sum([local_count])
        return aggregated[0]

    def global_avg(self, array: np.ndarray) -> np.ndarray:
        """Compute the federated average using local sums and sample counts."""

        array = np.asarray(array, dtype=float)
        local_sum = array.sum(axis=0)
        shape = np.asarray(local_sum).shape
        flat_sum = np.asarray(local_sum, dtype=float).ravel()
        payload = np.append(flat_sum, array.shape[0])
        aggregated = self._global_sum(payload)

        total_count = aggregated[-1]
        if total_count == 0:
            raise ValueError("Total count for federated average is zero.")

        averaged = aggregated[:-1] / total_count
        return averaged.reshape(shape)

    def global_min(self, array: np.ndarray) -> np.ndarray:
        """Compute min along axis=0 locally and then federated min across clients."""

        reduced = np.asarray(array, dtype=float).min(axis=0)
        shape = np.asarray(reduced).shape
        flat = np.asarray(reduced, dtype=float).ravel().tolist()
        aggregated = self._aggregate_request(AggregationType.MIN, flat)
        return np.asarray(aggregated, dtype=float).reshape(shape)

    def global_max(self, array: np.ndarray) -> np.ndarray:
        """Compute max along axis=0 locally and then federated max across clients."""

        reduced = np.asarray(array, dtype=float).max(axis=0)
        shape = np.asarray(reduced).shape
        flat = np.asarray(reduced, dtype=float).ravel().tolist()
        aggregated = self._aggregate_request(AggregationType.MAX, flat)
        return np.asarray(aggregated, dtype=float).reshape(shape)
