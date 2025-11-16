import logging
import threading
from concurrent import futures
from typing import Dict
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
    def __init__(self, request_id, expected_workers):
        self.request_id = request_id
        self.expected_workers = expected_workers
        self.aggregation_type = None
        self.vectors = []
        self.result = None
        self.acquired_count = 0
        self.error = None
        self.lock = threading.Lock()
        self.result_ready = threading.Event()

    def reset(self):
        self.aggregation_type = None
        self.vectors = []
        self.result = None
        self.acquired_count = 0
        self.error = None
        self.result_ready = threading.Event()


class AggregationServer(AggregationServerServicer):
    def __init__(self):
        self.aggregation_contexts: Dict[str, AggregationContext] = {}
        self.global_lock = threading.Lock()

    def Configure(self, request, context):
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
        agg_ctx = self._get_aggregation_context(request.request_id, context)
        with agg_ctx.lock:
            self._validate_request_type(agg_ctx, request, context)
            current_count = self._store_vectors(agg_ctx, request)

            self._compute_result_if_ready(agg_ctx, current_count)

        self._wait_for_result(agg_ctx, request, context)
        result = self._get_result(agg_ctx, context)
        return AggregateResponse(result=result)

    def Cleanup(self, request, context):
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

    def _get_aggregation_context(self, request_id, context) -> AggregationContext:
        with self.global_lock:
            if request_id not in self.aggregation_contexts:
                available = ", ".join(self.aggregation_contexts.keys()) or "none"
                msg = (
                    f"Request ID '{request_id}' not configured. "
                    f"Available contexts: {available}"
                )
                logger.error(f"[AGGREGATE] {msg}")
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
            return self.aggregation_contexts[request_id]

    def _validate_request_type(
        self, agg_ctx: AggregationContext, request, context
    ) -> None:
        if agg_ctx.aggregation_type is None:
            agg_ctx.aggregation_type = request.aggregation_type
        elif agg_ctx.aggregation_type != request.aggregation_type:
            msg = (
                f"Mismatched computation type for request_id='{request.request_id}'. "
                f"Expected '{agg_ctx.aggregation_type}', got '{request.aggregation_type}'"
            )
            logger.error(f"[AGGREGATE] {msg}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)

    def _store_vectors(self, agg_ctx: AggregationContext, request) -> int:
        agg_ctx.vectors.append(request.vectors)
        current_count = len(agg_ctx.vectors)
        logger.info(
            f"[AGGREGATE] request_id='{request.request_id}' aggregation_type='{request.aggregation_type}' "
            f"received response {current_count}/{agg_ctx.expected_workers}: vectors={request.vectors}"
        )
        return current_count

    def _compute_result_if_ready(
        self, agg_ctx: AggregationContext, current_count: int
    ) -> None:
        if current_count < agg_ctx.expected_workers or agg_ctx.result_ready.is_set():
            return

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

            if not all(len(v) == len(agg_ctx.vectors[0]) for v in agg_ctx.vectors):
                raise ValueError("All vectors must have the same length.")

            agg_ctx.result = aggregation_function(agg_ctx.vectors)
        except Exception as exc:
            agg_ctx.error = exc
        finally:
            agg_ctx.result_ready.set()

    def _wait_for_result(self, agg_ctx: AggregationContext, request, context) -> None:
        if not agg_ctx.result_ready.wait(
            timeout=config.max_wait_for_aggregation_inputs
        ):
            msg = (
                f"Timeout waiting for aggregation result for request_id='{request.request_id}' "
                f"and aggregation_type='{request.aggregation_type}'"
            )
            logger.error(f"[AGGREGATE] {msg}")
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, msg)

    def _get_result(self, agg_ctx: AggregationContext, context):
        with agg_ctx.lock:
            if agg_ctx.error is not None:
                logger.error(f"[AGGREGATE] Error during aggregation: {agg_ctx.error}")
                context.abort(grpc.StatusCode.INTERNAL, str(agg_ctx.error))

            result = agg_ctx.result
            agg_ctx.acquired_count += 1

            if agg_ctx.acquired_count == agg_ctx.expected_workers:
                agg_ctx.reset()

        return result

    @staticmethod
    def sum(vectors: List[List[float]]) -> List[float]:
        return [sum(column) for column in zip(*vectors)]

    @staticmethod
    def min(vectors: List[List[float]]) -> List[float]:
        return [min(column) for column in zip(*vectors)]

    @staticmethod
    def max(vectors: List[List[float]]) -> List[float]:
        return [max(column) for column in zip(*vectors)]


def serve():
    logging.basicConfig(
        level=config.log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.max_grpc_connections)
    )
    add_AggregationServerServicer_to_server(AggregationServer(), server)

    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("Aggregation", health_pb2.HealthCheckResponse.SERVING)

    server.add_insecure_port(f"0.0.0.0:{config.port}")
    server.start()
    logger.info(f"Aggregation server running on 0.0.0.0:{config.port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Aggregation server shutting down…")
        server.stop(0)


if __name__ == "__main__":
    serve()
