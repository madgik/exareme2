import logging
import warnings
from typing import List

import grpc
import numpy as np

from exareme2.aggregator import aggregator_pb2 as pb2
from exareme2.aggregator import aggregator_pb2_grpc as pb2_grpc
from exareme2.aggregator.constants import AggregationType

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
        response = self.stub.Configure(req)
        logger.info(
            f"[CONFIGURE] request_id='{self._request_id}' workers={num_workers} status={response.status}"
        )
        return response.status

    def cleanup(self) -> str:
        """
        Clean up the aggregation state on the server.
        """
        req = pb2.CleanupRequest(request_id=self._request_id)
        response = self.stub.Cleanup(req)
        logger.info(
            f"[CLEANUP] request_id='{self._request_id}' status={response.status}"
        )
        return response.status

    def aggregate(
        self, aggregation_type: AggregationType, values: List[float]
    ) -> List[float]:
        """
        Send an aggregation request to the server for the given computation type.
        If values is a nested structure, the values are flattened before sending.
        When the aggregated response is received (always a list), it is reshaped ("unflattened")
        to the original input format if applicable.
        """
        # Convert values to a NumPy array to capture its original shape.
        original_array = np.array(values)
        original_shape = original_array.shape

        # Flatten the input to a 1-D list for the aggregation request.
        flattened_values = original_array.flatten().tolist()
        warnings.warn(f"{values=} {flattened_values=}")

        # Build and send the aggregation request.
        req = pb2.AggregateRequest(
            request_id=str(self._request_id),
            aggregation_type=aggregation_type.value,
            vectors=flattened_values,
        )
        logger.info(
            f"[AGGREGATE] req_id='{self._request_id}' comp='{aggregation_type}'"
        )
        try:
            response = self.stub.Aggregate(req)
            result = response.result  # result is always a list

            # If the input was nested (i.e. more than one dimension) and
            # the result list length matches the total number of flattened elements,
            # reshape the result back to the original shape.
            if original_array.ndim > 1 and len(result) == original_array.size:
                unflattened_result = np.array(result).reshape(original_shape)
                return unflattened_result.tolist()
            else:
                # Otherwise, return the result as received.
                return result

        except grpc.RpcError as e:
            logger.error(f"[AGGREGATE ERROR] request_id='{self._request_id}': {e}")
            raise

    def sum(self, values: List[float]) -> float:
        """
        Compute the global sum by first computing a partial sum locally.
        """
        # Compute a local partial sum and send it to the aggregator.
        return self.aggregate(AggregationType.SUM, [np.sum(values)])

    def min(self, values: List[float]) -> float:
        """
        Compute the global minimum by first computing a local minimum.
        """
        return self.aggregate(AggregationType.MIN, [np.min(values)])

    def max(self, values: List[float]) -> float:
        """
        Compute the global maximum by first computing a local maximum.
        """
        return self.aggregate(AggregationType.MAX, [np.max(values)])

    def count(self, values: List[float]) -> float:
        """
        Compute the global count by sending the local count to be summed.
        """
        return self.aggregate(AggregationType.SUM, [float(len(values))])

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
