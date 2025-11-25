import logging
from typing import Sequence

import grpc
import numpy as np
from numpy.typing import NDArray

import exaflow.aggregation_clients.aggregation_server_pb2 as pb2
import exaflow.aggregation_clients.aggregation_server_pb2_grpc as pb2_grpc

from .constants import AggregationType
from .serialization import bytes_to_ndarray
from .serialization import ndarray_to_bytes

logger = logging.getLogger(__name__)


ArrayInput = NDArray[np.floating] | Sequence[float]


class BaseAggregationClient:
    def __init__(self, request_id: str, aggregator_address: str = "172.17.0.1:50051"):
        self._request_id = request_id
        self._channel = grpc.insecure_channel(aggregator_address)
        self._stub = pb2_grpc.AggregationServerStub(self._channel)

    def _aggregate_request(
        self, aggregation_type: AggregationType, flat_values: ArrayInput
    ) -> NDArray[np.floating]:
        tensor = ndarray_to_bytes(np.asarray(flat_values, dtype=np.float64))
        req = pb2.AggregateRequest(
            request_id=self._request_id,
            aggregation_type=aggregation_type.value,
            tensor=tensor,
        )
        logger.debug(
            "[AGGREGATE] req_id=%s comp=%s", self._request_id, aggregation_type
        )
        resp = self._stub.Aggregate(req)
        if resp.tensor:
            return bytes_to_ndarray(resp.tensor)
        return np.asarray(resp.result, dtype=np.float64)

    def _aggregate_batch_request(
        self, ops: list[tuple[AggregationType, ArrayInput]]
    ) -> list[NDArray[np.floating]]:
        operations = [
            pb2.Operation(
                aggregation_type=op.value,
                tensor=ndarray_to_bytes(np.asarray(vals, dtype=np.float64)),
            )
            for op, vals in ops
        ]
        req = pb2.AggregateBatchRequest(
            request_id=self._request_id, operations=operations
        )
        resp = self._stub.AggregateBatch(req)
        if resp.tensors:
            return [bytes_to_ndarray(tensor) for tensor in resp.tensors]

        results = resp.results
        offsets = resp.offsets
        reconstructed = []
        for i in range(len(offsets) - 1):
            start, end = offsets[i], offsets[i + 1]
            reconstructed.append(np.asarray(results[start:end], dtype=np.float64))
        return reconstructed

    def close(self) -> None:
        logger.debug("[CHANNEL] Closing gRPC channel.")
        self._channel.close()
