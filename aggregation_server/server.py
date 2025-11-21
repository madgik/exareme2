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

from .aggregation_server_pb2 import AggregateBatchResponse
from .aggregation_server_pb2 import AggregateResponse
from .aggregation_server_pb2 import CleanupResponse
from .aggregation_server_pb2 import ConfigureResponse
from .aggregation_server_pb2_grpc import AggregationServerServicer
from .aggregation_server_pb2_grpc import add_AggregationServerServicer_to_server
from .constants import AggregationType

logger = logging.getLogger("AggregationServer")
"""
Aggregation server overview
---------------------------

This service collects partial aggregates from multiple workers for a given
`request_id` and returns combined results. Two RPCs exist:

- `Aggregate`: single operation (SUM/MIN/MAX) over a flat vector of floats.
- `AggregateBatch`: multiple operations in one call; each op specifies type and
  a flat vector. The server aggregates per-op and returns all results in a
  flattened list with offsets.

Workflow:
1) Controller calls Configure(request_id, num_workers) to create context.
2) Workers call Aggregate/AggregateBatch with the same request_id.
3) When all expected workers contribute, the server aggregates and signals
   result readiness. Subsequent calls for the same request receive the same
   result until all workers have consumed it; then the context resets.
4) Controller calls Cleanup(request_id) when done.

State per request_id:
- expected_workers: how many worker responses to wait for.
- aggregation_type: fixed per Aggregate mode (SUM/MIN/MAX).
- vectors: collected payloads (single-op mode).
- batch_ops, batch_vectors: collected payloads per op (batch mode).
- result/result_ready/error/acquired_count: coordination flags/results.

Notes / future improvements:
- For very large arrays, converting to Python lists (`tolist`) is expensive.
  A future performance path could add a binary payload (bytes + shape) in the
  proto to avoid Python scalar expansion.
"""


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
        self.vector_length = None
        # Batch-related state
        self.batch_mode = False
        self.batch_ops = None  # List[str]
        self.batch_vectors = None  # List[List[List[float]]]
        self.batch_vector_lengths = None  # List[int]
        self.batch_result = None

    def reset(self):
        self.aggregation_type = None
        self.vectors = []
        self.result = None
        self.acquired_count = 0
        self.error = None
        self.result_ready.clear()
        self.vector_length = None
        self.batch_mode = False
        self.batch_ops = None
        self.batch_vectors = None
        self.batch_vector_lengths = None
        self.batch_result = None


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
        # TODO: Somehow make this faster maybe send a whole bunch of commands like we did with the tranfer instead of just one calculation:
        # Example: PCA uses this:
        # total_n_obs = agg_client.sum([float(n_obs)])[0]
        # total_sx = np.array(agg_client.sum(sx.tolist()), dtype=float)
        # total_sxx = np.array(agg_client.sum(sxx.tolist()), dtype=float)
        # Maybe we shoule just make aggregation server accept a list of computation: [Computation(<type>,<values>), ...]
        # In the above example it should be [Computation("sum",[float(n_obs)]), Computation("sum",sx.tolist()), Computation("sum",sxx.tolist())]
        # Order should matter so we do not aggregate wrong combinations then maybe have a name for the combination so add like a name right before the type
        # Optimize as much as we can
        agg_ctx = self._get_aggregation_context(request.request_id, context)
        if agg_ctx.batch_mode:
            msg = (
                f"[AGGREGATE] request_id='{request.request_id}' already in batch mode."
            )
            logger.error(msg)
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)
        with agg_ctx.lock:
            self._validate_request_type(agg_ctx, request, context)
            current_count = self._store_vectors(agg_ctx, request)

            self._compute_result_if_ready(agg_ctx, current_count)

        self._wait_for_result(agg_ctx, request, context)
        result = self._get_result(agg_ctx, context)
        return AggregateResponse(result=result)

    def AggregateBatch(self, request, context):
        agg_ctx = self._get_aggregation_context(request.request_id, context)
        with agg_ctx.lock:
            self._init_batch_if_needed(agg_ctx, request, context)
            current_count = self._store_batch_vectors(agg_ctx, request)
            self._compute_batch_result_if_ready(agg_ctx, current_count)

        self._wait_for_result(agg_ctx, request, context)
        results, offsets = self._get_batch_result(agg_ctx, context)
        return AggregateBatchResponse(results=results, offsets=offsets)

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
        vectors = list(request.vectors)
        if not vectors:
            raise ValueError(
                f"Empty vectors received for request_id='{request.request_id}'"
            )

        if agg_ctx.vector_length is None:
            agg_ctx.vector_length = len(vectors)
        elif len(vectors) != agg_ctx.vector_length:
            raise ValueError(
                f"All vectors must have the same length "
                f"(expected {agg_ctx.vector_length}, got {len(vectors)})"
            )

        agg_ctx.vectors.append(vectors)
        current_count = len(agg_ctx.vectors)
        logger.info(
            f"[AGGREGATE] request_id='{request.request_id}' aggregation_type='{request.aggregation_type}' "
            f"received response {current_count}/{agg_ctx.expected_workers} (vector_len={len(vectors)})"
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

            agg_ctx.result = aggregation_function(agg_ctx.vectors)
        except Exception as exc:
            agg_ctx.error = exc
        finally:
            agg_ctx.result_ready.set()

    def _wait_for_result(self, agg_ctx: AggregationContext, request, context) -> None:
        if not agg_ctx.result_ready.wait(
            timeout=config.max_wait_for_aggregation_inputs
        ):
            agg_type = getattr(request, "aggregation_type", "batch")
            received = (
                len(agg_ctx.vectors)
                if not agg_ctx.batch_mode
                else len(agg_ctx.batch_vectors[0] if agg_ctx.batch_vectors else [])
            )
            msg = (
                f"Timeout waiting for aggregation result for request_id='{request.request_id}' "
                f"and aggregation_type='{agg_type}' "
                f"(received {received}/{agg_ctx.expected_workers})"
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

    def _init_batch_if_needed(self, agg_ctx: AggregationContext, request, context):
        if not request.operations:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Batch request must include at least one operation.",
            )
        if not agg_ctx.batch_mode:
            agg_ctx.batch_mode = True
            agg_ctx.batch_ops = [op.aggregation_type for op in request.operations]
            agg_ctx.batch_vectors = [[] for _ in request.operations]
            agg_ctx.batch_vector_lengths = [None for _ in request.operations]
            return

        if len(request.operations) != len(agg_ctx.batch_ops):
            msg = (
                f"Batch op count mismatch for request_id='{request.request_id}' "
                f"(expected {len(agg_ctx.batch_ops)}, got {len(request.operations)})"
            )
            logger.error(f"[AGGREGATE] {msg}")
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)

        for idx, op in enumerate(request.operations):
            if op.aggregation_type != agg_ctx.batch_ops[idx]:
                msg = (
                    f"Batch aggregation_type mismatch at index {idx} for request_id='{request.request_id}' "
                    f"(expected '{agg_ctx.batch_ops[idx]}', got '{op.aggregation_type}')"
                )
                logger.error(f"[AGGREGATE] {msg}")
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, msg)

    def _store_batch_vectors(self, agg_ctx: AggregationContext, request) -> int:
        # assume operations already validated
        for idx, op in enumerate(request.operations):
            vectors = list(op.vectors)
            if not vectors:
                raise ValueError(
                    f"Empty vectors in batch op {idx} for request_id='{request.request_id}'"
                )
            expected_len = agg_ctx.batch_vector_lengths[idx]
            if expected_len is None:
                agg_ctx.batch_vector_lengths[idx] = len(vectors)
            elif len(vectors) != expected_len:
                raise ValueError(
                    f"All vectors in batch op {idx} must have the same length "
                    f"(expected {expected_len}, got {len(vectors)})"
                )
            agg_ctx.batch_vectors[idx].append(vectors)

        current_count = len(agg_ctx.batch_vectors[0])
        logger.info(
            f"[AGGREGATE] request_id='{request.request_id}' batch received response "
            f"{current_count}/{agg_ctx.expected_workers}"
        )
        return current_count

    def _compute_batch_result_if_ready(
        self, agg_ctx: AggregationContext, current_count: int
    ) -> None:
        if current_count < agg_ctx.expected_workers or agg_ctx.result_ready.is_set():
            return

        try:
            results = []
            offsets = [0]
            for idx, op_type in enumerate(agg_ctx.batch_ops):
                aggregation_function = {
                    AggregationType.SUM.value: self.sum,
                    AggregationType.MIN.value: self.min,
                    AggregationType.MAX.value: self.max,
                }.get(op_type)
                if aggregation_function is None:
                    raise ValueError(f"Unsupported computation type: {op_type}")
                vectors = agg_ctx.batch_vectors[idx]
                res = aggregation_function(vectors)
                results.extend(res)
                offsets.append(len(results))
            agg_ctx.batch_result = (results, offsets)
        except Exception as exc:
            agg_ctx.error = exc
        finally:
            agg_ctx.result_ready.set()

    def _get_batch_result(self, agg_ctx: AggregationContext, context):
        with agg_ctx.lock:
            if agg_ctx.error is not None:
                logger.error(
                    f"[AGGREGATE] Error during batch aggregation: {agg_ctx.error}"
                )
                context.abort(grpc.StatusCode.INTERNAL, str(agg_ctx.error))
            if not agg_ctx.batch_result:
                context.abort(grpc.StatusCode.INTERNAL, "Batch result missing.")
            results, offsets = agg_ctx.batch_result
            agg_ctx.acquired_count += 1
            if agg_ctx.acquired_count == agg_ctx.expected_workers:
                agg_ctx.reset()
            return results, offsets

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
