import argparse
import logging
import threading
from concurrent import futures

import grpc

from exareme2.aggregator import aggregator_pb2 as pb2
from exareme2.aggregator import aggregator_pb2_grpc as pb2_grpc
from exareme2.aggregator.aggregation_server import NumpyAggregationServer
from exareme2.aggregator.constants import AGG

TIMEOUT_SECONDS = 10
logger = logging.getLogger("AggregatorServer")


class AggregatorServer(pb2_grpc.AggregatorServicer, NumpyAggregationServer):
    """gRPC Aggregator Server that collects responses from workers and computes the final aggregation result."""

    def __init__(self):
        self.aggregation_states = {}
        self.lock = threading.Lock()

    def ConfigureAggregations(self, request, context):
        """
        Configure aggregation state for a given request_id.
        """
        with self.lock:
            if request.request_id in self.aggregation_states:
                logger.warning(
                    f"[CONFIGURE] Already exists: request_id='{request.request_id}'"
                )
                return pb2.ConfigureResponse(
                    status="Already configured for this request_id"
                )

            self.aggregation_states[request.request_id] = {
                "expected_workers": request.num_of_workers,
                "operations": {},  # mapping from operation key to its state
            }

        logger.info(
            f"[CONFIGURE] request_id='{request.request_id}' workers_expected={request.num_of_workers}"
        )
        return pb2.ConfigureResponse(status="Configured")

    def Aggregate(self, request, context):
        """
        Receive a worker's response for an aggregation operation.
        When all expected responses are received, compute the result and notify waiting workers.
        """
        request_id = request.request_id
        computation_type = request.computation_type
        operation_key = f"{computation_type}"  # Note: only one concurrent aggregation per type is supported

        with self.lock:
            if request_id not in self.aggregation_states:
                msg = f"[AGGREGATE] Request ID '{request_id}' not configured."
                logger.error(msg)
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)

            state = self.aggregation_states[request_id]
            expected_responses = state["expected_workers"]
            op_state = state["operations"].setdefault(
                operation_key,
                {
                    "responses": [],
                    "semaphore": threading.Semaphore(0),
                    "result": None,
                    "acquired_count": 0,
                },
            )
            op_state["responses"].append(request.data)
            current_responses = len(op_state["responses"])

            logger.info(
                f"[AGGREGATE] req_id='{request_id}' comp='{computation_type}' "
                f"responses={current_responses}/{expected_responses} data={request.data}"
            )

        # When the last response is received, compute the aggregated result.
        if current_responses == expected_responses:
            try:
                aggregation_functions = {
                    AGG.SUM.value: self.sum,
                    AGG.MIN.value: self.min,
                    AGG.MAX.value: self.max,
                }
                if computation_type not in aggregation_functions:
                    msg = f"[AGGREGATE] Unsupported comp_type='{computation_type}'"
                    logger.error(msg)
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)

                result = aggregation_functions[computation_type](op_state["responses"])
                with self.lock:
                    op_state["result"] = result

                logger.info(
                    f"[COMPLETE] req_id='{request_id}' comp='{computation_type}' result={result}"
                )
            except Exception as e:
                logger.exception(f"[AGGREGATE] Failed to compute: {str(e)}")
                context.abort(grpc.StatusCode.INTERNAL, str(e))

            # Release the semaphore for each expected response.
            for _ in range(expected_responses):
                op_state["semaphore"].release()

        # Wait for the aggregation result.
        if not op_state["semaphore"].acquire(timeout=TIMEOUT_SECONDS):
            msg = f"[AGGREGATE] TIMEOUT waiting for all responses: req_id='{request_id}' comp='{computation_type}'"
            logger.error(msg)
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, msg)

        with self.lock:
            result = op_state["result"]
            op_state["acquired_count"] += 1
            if op_state["acquired_count"] == expected_responses:
                del state["operations"][operation_key]
                logger.info(
                    f"[CLEANUP] Cleared op_state for req_id='{request_id}' comp='{computation_type}'"
                )

        return pb2.AggregateResponse(result=result)

    def CleanupAggregations(self, request, context):
        """
        Clean up all aggregation state for a given request_id.
        """
        with self.lock:
            if request.request_id in self.aggregation_states:
                del self.aggregation_states[request.request_id]
                logger.info(
                    f"[CLEANUP] Cleared all for request_id='{request.request_id}'"
                )
                return pb2.CleanupResponse(status="Cleaned up")
            else:
                logger.warning(
                    f"[CLEANUP] Nothing to clean for request_id='{request.request_id}'"
                )
                return pb2.CleanupResponse(status="No aggregation found for request_id")


def serve(host: str, port: int, max_workers: int, log_level: str):
    """
    Start the gRPC aggregator server.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb2_grpc.add_AggregatorServicer_to_server(AggregatorServer(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info(f"Aggregator server running on {host}:{port}")
    try:
        # Use gRPC's built-in wait_for_termination for a clean shutdown.
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Aggregator server shutting down...")
        server.stop(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    serve(args.host, args.port, args.max_workers, args.log_level)
