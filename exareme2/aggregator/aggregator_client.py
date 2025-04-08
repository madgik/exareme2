import logging
from typing import List

import grpc
import numpy as np

from exareme2.aggregator import aggregator_pb2 as pb2
from exareme2.aggregator import aggregator_pb2_grpc as pb2_grpc
from exareme2.aggregator.constants import AGG

logger = logging.getLogger(__name__)


class AggregationClient:
    """
    Client for interacting with the gRPC Aggregator Server.
    It supports basic operations such as sum, min, max, count, and average.
    """

    def __init__(self, request_id: str, aggregator_address: str = "172.17.0.1:50051"):
        self._request_id = request_id
        self.aggregator_address = aggregator_address
        self.channel = grpc.insecure_channel(self.aggregator_address)
        self.stub = pb2_grpc.AggregatorStub(self.channel)

    def configure(self, num_workers: int) -> str:
        """
        Configure the aggregation on the server.
        """
        req = pb2.ConfigureRequest(
            request_id=self._request_id, num_of_workers=num_workers
        )
        response = self.stub.ConfigureAggregations(req)
        logger.info(
            f"[CONFIGURE] request_id='{self._request_id}' workers={num_workers} status={response.status}"
        )
        return response.status

    def cleanup(self) -> str:
        """
        Clean up the aggregation state on the server.
        """
        req = pb2.CleanupRequest(request_id=self._request_id)
        response = self.stub.CleanupAggregations(req)
        logger.info(
            f"[CLEANUP] request_id='{self._request_id}' status={response.status}"
        )
        return response.status

    def aggregate(self, computation_type: AGG, values: List[float]) -> float:
        """
        Send an aggregation request to the server for the given computation type.
        The values are first aggregated locally (if needed) and then sent.
        """
        # Flatten values in case they are nested.
        data = [float(v) for v in np.array(values).flatten()]

        req = pb2.AggregateRequest(
            request_id=str(self._request_id),
            computation_type=computation_type.value,
            data=data,
        )
        logger.info(
            f"[AGGREGATE] req_id='{self._request_id}' comp='{computation_type}'"
        )
        try:
            response = self.stub.Aggregate(req)
            if not response.result:
                raise ValueError(
                    f"No result from server for request_id='{self._request_id}'"
                )
            return response.result[0]
        except grpc.RpcError as e:
            logger.error(f"[AGGREGATE ERROR] request_id='{self._request_id}': {e}")
            raise

    def sum(self, values: List[float]) -> float:
        """
        Compute the global sum by first computing a partial sum locally.
        """
        # Compute a local partial sum and send it to the aggregator.
        return self.aggregate(AGG.SUM, [np.sum(values)])

    def min(self, values: List[float]) -> float:
        """
        Compute the global minimum by first computing a local minimum.
        """
        return self.aggregate(AGG.MIN, [np.min(values)])

    def max(self, values: List[float]) -> float:
        """
        Compute the global maximum by first computing a local maximum.
        """
        return self.aggregate(AGG.MAX, [np.max(values)])

    def count(self, values: List[float]) -> float:
        """
        Compute the global count by sending the local count to be summed.
        """
        return self.aggregate(AGG.SUM, [float(len(values))])

    def avg(self, values: List[float]) -> float:
        """
        Compute the average using the aggregated sum and count.
        """
        logger.info(f"[AVG] Computing average using sum and count")
        total_sum = self.sum(values)
        total_count = self.count(values)
        if total_count == 0:
            logger.warning(
                f"[AVG] Division by zero for request_id='{self._request_id}'"
            )
            return float("nan")
        return total_sum / total_count

    def close(self):
        """
        Close the underlying gRPC channel.
        """
        logger.info("[CHANNEL] Closing gRPC channel.")
        self.channel.close()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, ensuring channel is closed."""
        self.close()
