import logging
import threading
from concurrent import futures

import grpc

from exareme2.aggregator import aggregator_pb2 as pb2
from exareme2.aggregator import aggregator_pb2_grpc as pb2_grpc
from exareme2.aggregator import config
from exareme2.aggregator.aggregation_server import NumpyAggregationServer
from exareme2.aggregator.constants import AGG

TIMEOUT_SECONDS = 10
logger = logging.getLogger("AggregatorServer")


class AggregationContext:
    """
    Represents the per-request aggregation state.
    Supports multiple aggregation operations by resetting its state after each completed operation.
    """

    def __init__(self, request_id, expected_workers):
        self.request_id = request_id
        self.expected_workers = expected_workers
        self.computation_type = None  # Set with the first response.
        self.responses = []  # Collected worker responses.
        self.result = None  # Aggregated result.
        self.acquired_count = 0  # Number of workers that have retrieved the result.
        self.error = None  # Stores an error if the computation fails.
        self.lock = threading.Lock()  # Per-request lock.
        self.result_ready = threading.Event()  # Signals when the result is computed.

    def reset(self):
        """
        Reset the internal state to prepare for a new aggregation operation.
        """
        self.computation_type = None
        self.responses = []
        self.result = None
        self.acquired_count = 0
        self.error = None
        self.result_ready = threading.Event()


class AggregatorServer(pb2_grpc.AggregatorServicer, NumpyAggregationServer):
    """
    gRPC Aggregator Server that:
      - Collects responses from workers.
      - Computes the aggregated result (computed only once by the worker that submits the last response).
      - Resets the state to allow multiple aggregations to use the same request id.
    """

    def __init__(self):
        self.aggregation_contexts = {}  # Mapping from request_id to AggregationContext.
        self.global_lock = threading.Lock()

    def ConfigureAggregations(self, request, context):
        """
        Create an AggregationContext for a given request_id if not already configured.
        """
        with self.global_lock:
            if request.request_id in self.aggregation_contexts:
                logger.warning(
                    f"[CONFIGURE] Request context already exists for request_id='{request.request_id}'"
                )
                return pb2.ConfigureResponse(
                    status="Already configured for this request_id"
                )
            self.aggregation_contexts[request.request_id] = AggregationContext(
                request.request_id, request.num_of_workers
            )

        logger.info(
            f"[CONFIGURE] Created AggregationContext for request_id='{request.request_id}' "
            f"with expected workers: {request.num_of_workers}"
        )
        return pb2.ConfigureResponse(status="Configured")

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
        result = self._finalize_and_get_result(agg_ctx, request, context)
        return pb2.AggregateResponse(result=result)

    def CleanupAggregations(self, request, context):
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
                return pb2.CleanupResponse(status="Cleaned up")
            else:
                logger.warning(
                    f"[CLEANUP] No AggregationContext found for request_id='{request.request_id}'"
                )
                return pb2.CleanupResponse(status="No aggregation found for request_id")

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
          - If the expected number of responses have been received,
            computes the aggregated result (only once) and signals that the result is ready.
        """
        with agg_ctx.lock:
            if agg_ctx.computation_type is None:
                agg_ctx.computation_type = request.computation_type
            elif agg_ctx.computation_type != request.computation_type:
                msg = (
                    f"Mismatched computation type for request_id='{request.request_id}'. "
                    f"Expected '{agg_ctx.computation_type}', got '{request.computation_type}'"
                )
                logger.error(f"[AGGREGATE] {msg}")
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)

            agg_ctx.responses.append(request.data)
            current_count = len(agg_ctx.responses)
            logger.info(
                f"[AGGREGATE] request_id='{request.request_id}' computation_type='{request.computation_type}' "
                f"received response {current_count}/{agg_ctx.expected_workers}: data={request.data}"
            )

            if (
                current_count == agg_ctx.expected_workers
                and not agg_ctx.result_ready.is_set()
            ):
                try:
                    aggregation_function = {
                        AGG.SUM.value: self.sum,
                        AGG.MIN.value: self.min,
                        AGG.MAX.value: self.max,
                    }.get(agg_ctx.computation_type)
                    if aggregation_function is None:
                        raise ValueError(
                            f"Unsupported computation type: {agg_ctx.computation_type}"
                        )
                    agg_ctx.result = aggregation_function(agg_ctx.responses)
                except Exception as e:
                    agg_ctx.error = e
                finally:
                    agg_ctx.result_ready.set()

    def _wait_for_result(self, agg_ctx, request, context):
        """
        Block until the aggregation result is computed and signaled by the event.
        Abort the RPC if waiting times out.
        """
        if not agg_ctx.result_ready.wait(timeout=TIMEOUT_SECONDS):
            msg = (
                f"Timeout waiting for aggregation result for request_id='{request.request_id}' "
                f"and computation_type='{request.computation_type}'"
            )
            logger.error(f"[AGGREGATE] {msg}")
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, msg)

    def _finalize_and_get_result(self, agg_ctx, request, context):
        """
        After the result is ready, acquire the result under the lock.
        Update the count of workers that have obtained the result.
        Once all workers have received the result, reset the context for the next aggregation operation.
        """
        with agg_ctx.lock:
            if agg_ctx.error is not None:
                logger.error(f"[AGGREGATE] Error during aggregation: {agg_ctx.error}")
                context.abort(grpc.StatusCode.INTERNAL, str(agg_ctx.error))
            result = agg_ctx.result
            agg_ctx.acquired_count += 1

            if agg_ctx.acquired_count == agg_ctx.expected_workers:
                # Reset the AggregationContext for the next aggregation.
                agg_ctx.reset()

        return result


def serve():
    """
    Start the gRPC Aggregator Server.
    """
    log_level = config.log_level
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))
    pb2_grpc.add_AggregatorServicer_to_server(AggregatorServer(), server)
    server.add_insecure_port(f"{config.host}:{config.port}")
    server.start()
    logger.info(f"Aggregator server running on {config.host}:{config.port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Aggregator server shutting down...")
        server.stop(0)


if __name__ == "__main__":
    serve()
