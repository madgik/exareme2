import logging
import threading
from concurrent import futures
from typing import List

import grpc
from grpc_health.v1 import health
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from aggregation_server import config

from .aggregation_server_pb2 import AggregateResponse
from .aggregation_server_pb2 import CleanupResponse
from .aggregation_server_pb2 import ConfigureResponse
from .aggregation_server_pb2_grpc import AggregationServerServicer
from .aggregation_server_pb2_grpc import add_AggregationServerServicer_to_server
from .constants import AggregationType

logger = logging.getLogger("AggregationServer")


class AggregationContext:
    """
    Represents the per-request aggregation state.
    Supports multiple aggregation operations by resetting its state after each completed operation.
    """

    def __init__(self, request_id, expected_workers):
        self.request_id = request_id
        self.expected_workers = expected_workers
        self.aggregation_type = None  # Set with the first response.
        self.vectors = []  # Collected worker vectors.
        self.result = None  # Aggregated result.
        self.acquired_count = 0  # Number of workers that have retrieved the result.
        self.error = None  # Stores an error if the computation fails.
        self.lock = threading.Lock()  # Per-request lock.
        self.result_ready = threading.Event()  # Signals when the result is computed.

    def reset(self):
        """
        Reset the internal state to prepare for a new aggregation operation.
        """
        self.aggregation_type = None
        self.vectors = []
        self.result = None
        self.acquired_count = 0
        self.error = None
        self.result_ready = threading.Event()


class AggregationServer(AggregationServerServicer):
    """
    gRPC Aggregation Server that:
      - Collects vectors from workers.
      - Computes the aggregated result (computed only once by the worker that submits the last response).
      - Resets the state to allow multiple aggregations to use the same request id.
    """

    def __init__(self):
        self.aggregation_contexts = {}  # Mapping from request_id to AggregationContext.
        self.global_lock = threading.Lock()

    def Configure(self, request, context):
        """
        Create an AggregationContext for a given request_id if not already configured.
        """
        with self.global_lock:
            if request.request_id in self.aggregation_contexts:
                logger.warning(
                    f"[CONFIGURE] Request context already exists for request_id='{request.request_id}'"
                )
                return ConfigureResponse(
                    status="Already configured for this request_id"
                )
            self.aggregation_contexts[request.request_id] = AggregationContext(
                request.request_id, request.num_of_workers
            )

        logger.info(
            f"[CONFIGURE] Created AggregationContext for request_id='{request.request_id}' "
            f"with expected workers: {request.num_of_workers}"
        )
        return ConfigureResponse(status="Configured")

    def Aggregate(self, request, context):
        """
        Process an aggregation response from a worker:
          1. Retrieve the AggregationContext.
          2. Append the worker’s response (and trigger the computation if this is the last response).
          3. Wait until the result is ready.
          4. Finalize the aggregation and return the result.
        """
        agg_ctx = self._get_aggregation_context(request.request_id, context)
        self._append_response_and_compute(agg_ctx, request, context)
        self._wait_for_result(agg_ctx, request, context)
        result = self._finalize_and_get_result(agg_ctx, context)
        return AggregateResponse(result=result)

    def Cleanup(self, request, context):
        """
        Remove the AggregationContext for this request_id.
        (Useful if the request is complete and no further aggregations are expected.)
        """
        with self.global_lock:
            if request.request_id in self.aggregation_contexts:
                del self.aggregation_contexts[request.request_id]
                logger.info(
                    f"[CLEANUP] Removed AggregationContext for request_id='{request.request_id}'"
                )
                return CleanupResponse(status="Cleaned up")
            else:
                logger.warning(
                    f"[CLEANUP] No AggregationContext found for request_id='{request.request_id}'"
                )
                return CleanupResponse(status="No aggregation found for request_id")

    def _get_aggregation_context(self, request_id, context):
        """
        Retrieve the AggregationContext associated with the given request_id.
        Aborts the RPC if no such context exists.
        """
        with self.global_lock:
            if request_id not in self.aggregation_contexts:
                msg = f"Request ID '{request_id}' not configured."
                logger.error(f"[AGGREGATE] {msg}")
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
            return self.aggregation_contexts[request_id]

    def _append_response_and_compute(self, agg_ctx, request, context):
        """
        Under the AggregationContext’s lock:
          - Sets the computation type on the first response.
          - Appends the worker’s response.
          - If the expected number of vectors have been received,
            computes the aggregated result (only once) and signals that the result is ready.
        """
        with agg_ctx.algorithm_execution_lock:
            if agg_ctx.aggregation_type is None:
                agg_ctx.aggregation_type = request.aggregation_type
            elif agg_ctx.aggregation_type != request.aggregation_type:
                msg = (
                    f"Mismatched computation type for request_id='{request.request_id}'. "
                    f"Expected '{agg_ctx.aggregation_type}', got '{request.aggregation_type}'"
                )
                logger.error(f"[AGGREGATE] {msg}")
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)

            agg_ctx.vectors.append(request.vectors)
            current_count = len(agg_ctx.vectors)
            logger.info(
                f"[AGGREGATE] request_id='{request.request_id}' aggregation_type='{request.aggregation_type}' "
                f"received response {current_count}/{agg_ctx.expected_workers}: vectors={request.vectors}"
            )

            if (
                current_count == agg_ctx.expected_workers
                and not agg_ctx.result_ready.is_set()
            ):
                try:
                    aggregation_function = {
                        AggregationType.SUM.value: self.sum,
                        AggregationType.MIN.value: self.min,
                        AggregationType.MAX.value: self.max,
                    }.get(agg_ctx.aggregation_type)

                    if aggregation_function is None:
                        raise ValueError(
                            f"Unsupported computation type: {agg_ctx.aggregation_type}"
                        )

                    if not agg_ctx.vectors:
                        raise ValueError("No vectors provided for aggregation.")

                    if not all(
                        len(sublist) == len(agg_ctx.vectors[0])
                        for sublist in agg_ctx.vectors
                    ):
                        raise ValueError("All vectors must have the same length.")

                    # This will now be handled inside the aggregation function
                    agg_ctx.result = aggregation_function(agg_ctx.vectors)
                except Exception as e:
                    agg_ctx.error = e
                finally:
                    agg_ctx.result_ready.set()

    def _wait_for_result(self, agg_ctx, request, context):
        """
        Block until the aggregation result is computed and signaled by the event.
        Abort the RPC if waiting times out.
        """
        if not agg_ctx.result_ready.wait(timeout=config.timeout):
            msg = (
                f"Timeout waiting for aggregation result for request_id='{request.request_id}' "
                f"and aggregation_type='{request.aggregation_type}'"
            )
            logger.error(f"[AGGREGATE] {msg}")
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, msg)

    def _finalize_and_get_result(self, agg_ctx, context):
        """
        After the result is ready, acquire the result under the lock.
        Update the count of workers that have obtained the result.
        Once all workers have received the result, reset the context for the next aggregation operation.
        """
        with agg_ctx.algorithm_execution_lock:
            if agg_ctx.error is not None:
                logger.error(f"[AGGREGATE] Error during aggregation: {agg_ctx.error}")
                context.abort(grpc.StatusCode.INTERNAL, str(agg_ctx.error))
            result = agg_ctx.result
            agg_ctx.acquired_count += 1

            if agg_ctx.acquired_count == agg_ctx.expected_workers:
                # Reset the AggregationContext for the next aggregation.
                agg_ctx.reset()

        return result

    @staticmethod
    def validate_vector_lengths(vectors: List[List[float]]):
        if not vectors:
            raise ValueError("No vectors provided.")
        first_length = len(vectors[0])
        if not all(len(v) == first_length for v in vectors):
            raise ValueError("All vectors must have the same length.")

    @staticmethod
    def sum(vectors: List[List[float]]) -> List[float]:
        return [sum(x) for x in zip(*vectors)]

    @staticmethod
    def min(vectors: List[List[float]]) -> List[float]:
        return [min(x) for x in zip(*vectors)]

    @staticmethod
    def max(vectors: List[List[float]]) -> List[float]:
        return [max(x) for x in zip(*vectors)]


def serve():
    """
    Start the gRPC Aggregation Server.
    """

    logging.basicConfig(
        level=config.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))
    add_AggregationServerServicer_to_server(AggregationServer(), server)

    # --- gRPC health ---
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("Aggregation", health_pb2.HealthCheckResponse.SERVING)
    server.add_insecure_port(f"{config.host}:{config.port}")
    server.start()
    logger.info(f"Aggregation server running on {config.host}:{config.port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Aggregation server shutting down...")
        server.stop(0)


if __name__ == "__main__":
    serve()
