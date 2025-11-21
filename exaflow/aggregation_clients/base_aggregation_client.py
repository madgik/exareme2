import logging
from typing import List

import grpc

import exaflow.aggregation_clients.aggregation_server_pb2 as pb2
import exaflow.aggregation_clients.aggregation_server_pb2_grpc as pb2_grpc

from .constants import AggregationType

logger = logging.getLogger(__name__)


class BaseAggregationClient:
    def __init__(self, request_id: str, aggregator_address: str = "172.17.0.1:50051"):
        self._request_id = request_id
        self._channel = grpc.insecure_channel(aggregator_address)
        self._stub = pb2_grpc.AggregationServerStub(self._channel)

    def _aggregate_request(
        self, aggregation_type: AggregationType, flat_values: List[float]
    ) -> List[float]:
        req = pb2.AggregateRequest(
            request_id=self._request_id,
            aggregation_type=aggregation_type.value,
            vectors=flat_values,
        )
        logger.debug(
            "[AGGREGATE] req_id=%s comp=%s", self._request_id, aggregation_type
        )
        return self._stub.Aggregate(req).result

    def _aggregate_batch_request(
        self, ops: list[tuple[AggregationType, list[float]]]
    ) -> list[list[float]]:
        operations = [
            pb2.Operation(aggregation_type=op.value, vectors=vals) for op, vals in ops
        ]
        req = pb2.AggregateBatchRequest(
            request_id=self._request_id, operations=operations
        )
        resp = self._stub.AggregateBatch(req)
        results = resp.results
        offsets = resp.offsets
        reconstructed = []
        for i in range(len(offsets) - 1):
            start, end = offsets[i], offsets[i + 1]
            reconstructed.append(results[start:end])
        return reconstructed

    def close(self) -> None:
        logger.debug("[CHANNEL] Closing gRPC channel.")
        self._channel.close()
