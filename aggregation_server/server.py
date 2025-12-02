import logging
import threading
from concurrent import futures
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import grpc
import numpy as np
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
from .serialization import bytes_to_ndarray
from .serialization import ndarray_to_bytes

logger = logging.getLogger("AggregationServer")
"""
Aggregation server, threading model, and data flow
--------------------------------------------------

Overview
~~~~~~~~
Each `request_id` represents a round of vector aggregation across `expected_workers`
callers. Workers first register via `Configure(request_id, num_workers)` and then
call either:
- `Aggregate`: single operation (SUM / MIN / MAX) over a flat vector.
- `AggregateBatch`: multiple operations in one call; each operation defines its
  own aggregation type and vector.

Payloads can arrive either as legacy repeated doubles (`vectors`) or as Arrow
tensors (`tensor`). Responses always include both a Python list and the binary
tensor form for efficiency.

Synchronization model
~~~~~~~~~~~~~~~~~~~~~
For every `request_id` we keep an `AggregationContext` guarded by a single
`Condition`. The context follows a small state machine:
- `collecting`: accepting inputs. Once `expected_workers` inputs arrive we
  compute the result and move to `ready`.
- `ready` or `failed`: result (or error) is available. Each worker request that
  hits this state increments `acquired_count`. When all `expected_workers`
  have observed the result or error, the context is reset back to `collecting`.

The same state machine is shared by single and batch RPCs; the first RPC to
arrive fixes the mode for that round, preventing cross-mode mixing. All waits
are bounded by `config.max_wait_for_aggregation_inputs` to avoid deadlocks from
missing workers.

Error handling
~~~~~~~~~~~~~~
Any validation or computation error sets the context to `failed`, notifies
waiters, and counts the triggering request as consumed so the round can drain.
Subsequent workers immediately receive the same error instead of waiting
indefinitely.
"""


class AggregationContext:
    def __init__(self, request_id: str, expected_workers: int):
        self.request_id = request_id
        self.expected_workers = expected_workers
        self.mode: Optional[str] = None  # "single" or "batch"
        self.aggregation_type: Optional[str] = None
        self.vectors: List[np.ndarray] = []
        self.vector_length: Optional[int] = None
        self.result: Optional[np.ndarray] = None
        self.error: Optional[Exception] = None
        self.acquired_count = 0
        self.state = "collecting"  # collecting | ready | failed
        self.condition = threading.Condition()
        # Batch-specific fields
        self.batch_ops: Optional[List[str]] = None
        self.batch_vectors: Optional[List[List[np.ndarray]]] = None
        self.batch_vector_lengths: Optional[List[Optional[int]]] = None
        self.batch_result: Optional[Tuple[List[float], List[int], List[np.ndarray]]] = (
            None
        )

    def reset(self) -> None:
        self.mode = None
        self.aggregation_type = None
        self.vectors = []
        self.vector_length = None
        self.result = None
        self.error = None
        self.acquired_count = 0
        self.state = "collecting"
        self.batch_ops = None
        self.batch_vectors = None
        self.batch_vector_lengths = None
        self.batch_result = None
        self.condition.notify_all()

    def mark_ready(self) -> None:
        self.state = "ready"
        self.condition.notify_all()

    def fail(self, exc: Exception) -> None:
        self.error = exc
        self.state = "failed"
        self.condition.notify_all()


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
        with agg_ctx.condition:
            if agg_ctx.state != "collecting":
                self._wait_for_result_ready_locked(agg_ctx, request, context)
                result = self._consume_single_result_locked(agg_ctx, context)
            else:
                self._prepare_single_mode(agg_ctx, request, context)
                try:
                    current_count = self._store_single_vector_locked(agg_ctx, request)
                    if current_count == agg_ctx.expected_workers:
                        self._compute_single_locked(agg_ctx)
                except Exception as exc:
                    self._handle_failure(agg_ctx, exc, context)
                self._wait_for_result_ready_locked(agg_ctx, request, context)
                result = self._consume_single_result_locked(agg_ctx, context)

        return AggregateResponse(
            result=result.tolist(), tensor=ndarray_to_bytes(result)
        )

    def AggregateBatch(self, request, context):
        agg_ctx = self._get_aggregation_context(request.request_id, context)
        with agg_ctx.condition:
            if agg_ctx.state != "collecting":
                self._wait_for_result_ready_locked(agg_ctx, request, context)
                results, offsets, tensors = self._consume_batch_result_locked(
                    agg_ctx, context
                )
            else:
                self._prepare_batch_mode(agg_ctx, request, context)
                try:
                    current_count = self._store_batch_vectors_locked(agg_ctx, request)
                    if current_count == agg_ctx.expected_workers:
                        self._compute_batch_locked(agg_ctx)
                except Exception as exc:
                    self._handle_failure(agg_ctx, exc, context)

                self._wait_for_result_ready_locked(agg_ctx, request, context)
                results, offsets, tensors = self._consume_batch_result_locked(
                    agg_ctx, context
                )

        tensor_payloads = [ndarray_to_bytes(arr) for arr in tensors]
        return AggregateBatchResponse(
            results=results, offsets=offsets, tensors=tensor_payloads
        )

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

    def _prepare_single_mode(self, agg_ctx: AggregationContext, request, context):
        if agg_ctx.mode is None:
            agg_ctx.mode = "single"
        elif agg_ctx.mode != "single":
            msg = (
                f"[AGGREGATE] request_id='{request.request_id}' already in batch mode."
            )
            logger.error(msg)
            self._handle_failure(agg_ctx, ValueError(msg), context)

        if agg_ctx.aggregation_type is None:
            agg_ctx.aggregation_type = request.aggregation_type
        elif agg_ctx.aggregation_type != request.aggregation_type:
            msg = (
                f"Mismatched computation type for request_id='{request.request_id}'. "
                f"Expected '{agg_ctx.aggregation_type}', got '{request.aggregation_type}'"
            )
            logger.error(f"[AGGREGATE] {msg}")
            self._handle_failure(agg_ctx, ValueError(msg), context)

    def _prepare_batch_mode(self, agg_ctx: AggregationContext, request, context):
        if not request.operations:
            self._handle_failure(
                agg_ctx,
                ValueError("Batch request must include at least one operation."),
                context,
            )

        if agg_ctx.mode is None:
            agg_ctx.mode = "batch"
            agg_ctx.batch_ops = [op.aggregation_type for op in request.operations]
            agg_ctx.batch_vectors = [[] for _ in request.operations]
            agg_ctx.batch_vector_lengths = [None for _ in request.operations]
            return

        if agg_ctx.mode != "batch":
            msg = f"[AGGREGATE] request_id='{request.request_id}' already in single-op mode."
            logger.error(msg)
            self._handle_failure(agg_ctx, ValueError(msg), context)

        if len(request.operations) != len(agg_ctx.batch_ops):
            msg = (
                f"Batch op count mismatch for request_id='{request.request_id}' "
                f"(expected {len(agg_ctx.batch_ops)}, got {len(request.operations)})"
            )
            logger.error(f"[AGGREGATE] {msg}")
            self._handle_failure(agg_ctx, ValueError(msg), context)

        for idx, op in enumerate(request.operations):
            if op.aggregation_type != agg_ctx.batch_ops[idx]:
                msg = (
                    f"Batch aggregation_type mismatch at index {idx} for request_id='{request.request_id}' "
                    f"(expected '{agg_ctx.batch_ops[idx]}', got '{op.aggregation_type}')"
                )
                logger.error(f"[AGGREGATE] {msg}")
                self._handle_failure(agg_ctx, ValueError(msg), context)

    def _store_single_vector_locked(self, agg_ctx: AggregationContext, request) -> int:
        vector = self._decode_vector(
            request.tensor, request.vectors, request.request_id
        )

        vector_length = len(vector)
        if agg_ctx.vector_length is None:
            agg_ctx.vector_length = vector_length
        elif vector_length != agg_ctx.vector_length:
            raise ValueError(
                f"All vectors must have the same length "
                f"(expected {agg_ctx.vector_length}, got {vector_length})"
            )

        agg_ctx.vectors.append(vector)
        current_count = len(agg_ctx.vectors)
        logger.info(
            f"[AGGREGATE] request_id='{request.request_id}' aggregation_type='{request.aggregation_type}' "
            f"received response {current_count}/{agg_ctx.expected_workers} (vector_len={vector_length})"
        )
        return current_count

    def _compute_single_locked(self, agg_ctx: AggregationContext) -> None:
        aggregation_function = self._aggregation_fn(agg_ctx.aggregation_type)
        agg_ctx.result = aggregation_function(agg_ctx.vectors)
        agg_ctx.mark_ready()

    def _store_batch_vectors_locked(self, agg_ctx: AggregationContext, request) -> int:
        for idx, op in enumerate(request.operations):
            vector = self._decode_vector(op.tensor, op.vectors, request.request_id)
            vector_length = len(vector)
            expected_len = agg_ctx.batch_vector_lengths[idx]
            if expected_len is None:
                agg_ctx.batch_vector_lengths[idx] = vector_length
            elif vector_length != expected_len:
                raise ValueError(
                    f"All vectors in batch op {idx} must have the same length "
                    f"(expected {expected_len}, got {vector_length})"
                )
            agg_ctx.batch_vectors[idx].append(vector)

        current_count = len(agg_ctx.batch_vectors[0])
        logger.info(
            f"[AGGREGATE] request_id='{request.request_id}' batch received response "
            f"{current_count}/{agg_ctx.expected_workers}"
        )
        return current_count

    def _compute_batch_locked(self, agg_ctx: AggregationContext) -> None:
        flat_results: List[float] = []
        offsets = [0]
        tensor_results: List[np.ndarray] = []
        for idx, op_type in enumerate(agg_ctx.batch_ops):
            aggregation_function = self._aggregation_fn(op_type)
            vectors = agg_ctx.batch_vectors[idx]
            res = aggregation_function(vectors)
            tensor_results.append(res)
            res_list = res.tolist()
            flat_results.extend(res_list)
            offsets.append(len(flat_results))

        agg_ctx.batch_result = (flat_results, offsets, tensor_results)
        agg_ctx.mark_ready()

    def _wait_for_result_ready_locked(
        self, agg_ctx: AggregationContext, request, context
    ) -> None:
        ready = agg_ctx.condition.wait_for(
            lambda: agg_ctx.state in {"ready", "failed"},
            timeout=config.max_wait_for_aggregation_inputs,
        )
        if ready:
            return

        agg_type = getattr(request, "aggregation_type", "batch")
        received = self._received_count(agg_ctx)
        msg = (
            f"Timeout waiting for aggregation result for request_id='{request.request_id}' "
            f"and aggregation_type='{agg_type}' "
            f"(received {received}/{agg_ctx.expected_workers})"
        )
        logger.error(f"[AGGREGATE] {msg}")
        timeout_exc = TimeoutError(msg)
        agg_ctx.fail(timeout_exc)
        self._mark_acquired_and_maybe_reset_locked(agg_ctx)
        context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, msg)

    def _consume_single_result_locked(
        self, agg_ctx: AggregationContext, context
    ) -> np.ndarray:
        if agg_ctx.error is not None:
            error = agg_ctx.error
            status = self._status_for_exception(error)
            self._mark_acquired_and_maybe_reset_locked(agg_ctx)
            logger.error(f"[AGGREGATE] Error during aggregation: {error}")
            context.abort(status, str(error))

        if agg_ctx.result is None:
            context.abort(grpc.StatusCode.INTERNAL, "Aggregation result missing.")

        result = agg_ctx.result
        self._mark_acquired_and_maybe_reset_locked(agg_ctx)
        return result

    def _consume_batch_result_locked(self, agg_ctx: AggregationContext, context):
        if agg_ctx.error is not None:
            error = agg_ctx.error
            status = self._status_for_exception(error)
            self._mark_acquired_and_maybe_reset_locked(agg_ctx)
            logger.error(f"[AGGREGATE] Error during batch aggregation: {error}")
            context.abort(status, str(error))

        if not agg_ctx.batch_result:
            context.abort(grpc.StatusCode.INTERNAL, "Batch result missing.")

        results, offsets, tensors = agg_ctx.batch_result
        self._mark_acquired_and_maybe_reset_locked(agg_ctx)
        return results, offsets, tensors

    def _mark_acquired_and_maybe_reset_locked(
        self, agg_ctx: AggregationContext
    ) -> None:
        agg_ctx.acquired_count += 1
        if agg_ctx.acquired_count >= agg_ctx.expected_workers:
            agg_ctx.reset()

    def _handle_failure(
        self, agg_ctx: AggregationContext, exc: Exception, context
    ) -> None:
        agg_ctx.fail(exc)
        self._mark_acquired_and_maybe_reset_locked(agg_ctx)
        status = self._status_for_exception(exc)
        logger.error(f"[AGGREGATE] {exc}", exc_info=exc)
        context.abort(status, str(exc))

    def _status_for_exception(self, exc: Exception) -> grpc.StatusCode:
        if isinstance(exc, TimeoutError):
            return grpc.StatusCode.DEADLINE_EXCEEDED
        if isinstance(exc, ValueError):
            return grpc.StatusCode.INVALID_ARGUMENT
        return grpc.StatusCode.INTERNAL

    def _received_count(self, agg_ctx: AggregationContext) -> int:
        if agg_ctx.mode == "batch":
            return len(agg_ctx.batch_vectors[0]) if agg_ctx.batch_vectors else 0
        return len(agg_ctx.vectors)

    def _aggregation_fn(self, aggregation_type: str):
        aggregation_function = {
            AggregationType.SUM.value: self.sum,
            AggregationType.MIN.value: self.min,
            AggregationType.MAX.value: self.max,
        }.get(aggregation_type)

        if aggregation_function is None:
            raise ValueError(f"Unsupported computation type: {aggregation_type}")

        return aggregation_function

    def _decode_vector(
        self, tensor: bytes, legacy_vectors, request_id: str
    ) -> np.ndarray:
        if tensor:
            return np.asarray(bytes_to_ndarray(tensor), dtype=np.float64).reshape(-1)

        vectors = list(legacy_vectors)
        if not vectors:
            raise ValueError(f"Empty vectors received for request_id='{request_id}'")
        return np.asarray(vectors, dtype=np.float64)

    @staticmethod
    def sum(vectors: List[np.ndarray]) -> np.ndarray:
        return np.add.reduce(vectors)

    @staticmethod
    def min(vectors: List[np.ndarray]) -> np.ndarray:
        return np.minimum.reduce(vectors)

    @staticmethod
    def max(vectors: List[np.ndarray]) -> np.ndarray:
        return np.maximum.reduce(vectors)


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
