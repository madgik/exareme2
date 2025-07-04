import concurrent.futures
import uuid

import grpc
import numpy as np
import pandas as pd
import pytest

from aggregation_server.constants import AggregationType
from exareme2.aggregation_clients.controller_aggregation_client import (
    ControllerAggregationClient,
)
from exareme2.aggregation_clients.exaflow_udf_aggregation_client import (
    ExaflowUDFAggregationClient,
)


@pytest.fixture(scope="module")
def agg_client(aggregation_server_service):
    """
    * Configure an aggregation controller client so expects 2 workers.
    * Yield a **worker** client that the tests will call .sum/.min/.max on.
    * Tear everything down afterward.
    """
    request_id = str(uuid.uuid4())

    controller = ControllerAggregationClient(request_id=request_id)
    status = controller.configure(2)
    assert status == "Configured"

    worker = ExaflowUDFAggregationClient(request_id=request_id)

    yield worker  # <<< The tests receive only the worker-side client.

    controller.cleanup()
    worker.close()
    controller.close()


# --------------------------------------------------------------------------- #
# Single-column helpers
# --------------------------------------------------------------------------- #
def _run_concurrently(fn, times: int):
    with concurrent.futures.ThreadPoolExecutor(max_workers=times) as executor:
        futures = [executor.submit(fn) for _ in range(times)]
        return [f.result() for f in futures]


# --------------------------------------------------------------------------- #
# === Single-Column Tests ===
# --------------------------------------------------------------------------- #
def test_sum_single_column(agg_client):
    df = pd.DataFrame({"col": [1, 2, 3]})
    expected = 2 * np.sum(df["col"])

    results = _run_concurrently(lambda: agg_client.sum(df["col"]), 2)

    for r in results:
        assert r == expected


def test_min_single_column(agg_client):
    df = pd.DataFrame({"col": [1, 2, 3]})
    expected = np.min(df["col"])

    results = _run_concurrently(lambda: agg_client.min(df["col"]), 2)

    for r in results:
        assert r == expected


def test_max_single_column(agg_client):
    df = pd.DataFrame({"col": [1, 2, 3]})
    expected = np.max(df["col"])

    results = _run_concurrently(lambda: agg_client.max(df["col"]), 2)

    for r in results:
        assert r == expected


# --------------------------------------------------------------------------- #
# === Multi-Column Tests (direct aggregate) ===
# --------------------------------------------------------------------------- #
def test_sum_multi_column(agg_client):
    df = pd.DataFrame({"col1": [10.0], "col2": [20.0], "col3": [30.0]})
    expected_first = 2 * df["col1"].iloc[0]

    results = _run_concurrently(
        lambda: agg_client.aggregate(AggregationType.SUM, df["col1"]), 2
    )

    for r in results:
        assert r == expected_first


def test_min_multi_column(agg_client):
    df = pd.DataFrame({"col1": [5.0], "col2": [50.0], "col3": [500.0]})
    expected_first = df["col1"].iloc[0]

    results = _run_concurrently(
        lambda: agg_client.aggregate(AggregationType.MIN, df["col1"]), 2
    )

    for r in results:
        assert r == expected_first


def test_max_multi_column(agg_client):
    df = pd.DataFrame({"col1": [15.0], "col2": [25.0], "col3": [35.0]})
    expected_first = df["col1"].iloc[0]

    results = _run_concurrently(
        lambda: agg_client.aggregate(AggregationType.MAX, df["col1"]), 2
    )

    for r in results:
        assert r == expected_first


# --------------------------------------------------------------------------- #
# Sequential aggregations (same request id, same client)
# --------------------------------------------------------------------------- #
def test_sequential_aggregations_same_request(agg_client):
    df = pd.DataFrame({"col": [1, 2, 3]})

    expected_sum = 2 * np.sum(df["col"])
    expected_min = np.min(df["col"])

    # SUM
    sum_results = _run_concurrently(lambda: agg_client.sum(df["col"]), 2)
    for r in sum_results:
        assert r == expected_sum

    # MIN
    min_results = _run_concurrently(lambda: agg_client.min(df["col"]), 2)
    for r in min_results:
        assert r == expected_min


# --------------------------------------------------------------------------- #
# Parallel aggregations (different request ids)
# --------------------------------------------------------------------------- #
def _make_pair(req_id: str, n_workers: int = 2):
    """Utility: return (controller, worker) configured for *n_workers*."""
    ctrl = ControllerAggregationClient(request_id=req_id)
    ctrl.configure(n_workers)
    return ctrl, ExaflowUDFAggregationClient(request_id=req_id)


def test_parallel_aggregations_different_clients():
    df = pd.DataFrame({"col": [1, 2, 3]})

    expected_sum = 2 * np.sum(df["col"])
    expected_min = np.min(df["col"])
    expected_max = np.max(df["col"])

    ctrl_sum, worker_sum = _make_pair("parallel-test-sum")
    ctrl_min, worker_min = _make_pair("parallel-test-min")
    ctrl_max, worker_max = _make_pair("parallel-test-max")

    def call_sum():
        return worker_sum.sum(df["col"])

    def call_min():
        return worker_min.min(df["col"])

    def call_max():
        return worker_max.max(df["col"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        tasks = []
        for _ in range(2):
            tasks.extend(
                [
                    executor.submit(call_sum),
                    executor.submit(call_min),
                    executor.submit(call_max),
                ]
            )
        results = [f.result() for f in tasks]

    sum_results = results[0::3]
    min_results = results[1::3]
    max_results = results[2::3]

    for r in sum_results:
        assert r == expected_sum
    for r in min_results:
        assert r == expected_min
    for r in max_results:
        assert r == expected_max

    # cleanup
    for c in (ctrl_sum, ctrl_min, ctrl_max):
        c.cleanup()
        c.close()
    for w in (worker_sum, worker_min, worker_max):
        w.close()


# --------------------------------------------------------------------------- #
# High-concurrency and reset
# --------------------------------------------------------------------------- #
def test_high_concurrency_and_reset():
    df = pd.DataFrame({"col": list(range(1, 11))})
    expected_sum = 10 * np.sum(df["col"])

    ctrl, worker = _make_pair("high-concurrency", n_workers=10)

    for _ in range(2):  # two cycles
        results = _run_concurrently(lambda: worker.sum(df["col"]), 10)
        for r in results:
            assert r == expected_sum

    ctrl.cleanup()
    worker.close()
    ctrl.close()


# --------------------------------------------------------------------------- #
# Failure scenarios
# --------------------------------------------------------------------------- #
def test_worker_timeout(agg_client):
    """
    Expect DEADLINE_EXCEEDED because only one of the two required workers
    calls the aggregation client.
    """
    df = pd.DataFrame({"col": [1, 2, 3]})

    with pytest.raises(grpc.RpcError) as exc_info:
        agg_client.sum(df["col"])

    assert exc_info.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED


def test_mixed_computation_types(agg_client):
    df = pd.DataFrame({"col": [1, 2, 3]})

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        fut_sum = executor.submit(agg_client.sum, df["col"])
        fut_min = executor.submit(agg_client.min, df["col"])

        errors, results = [], []
        for f in (fut_sum, fut_min):
            try:
                results.append(f.result())
            except Exception as e:
                errors.append(e)

    assert len(errors) == 1, "Exactly one call should fail due to mixed types."


def test_error_propagation_on_unsupported_type():
    """
    Call .aggregate with a bogus aggregation type ('AVG').
    Both concurrent calls must raise.
    """
    ctrl, worker = _make_pair("error-propagation")

    df = pd.DataFrame({"col": [1, 2, 3]})

    def call_avg():
        # Intentionally pass an invalid value – should raise locally
        # (AttributeError) or remotely (grpc.INTERNAL), either is fine.
        return worker.aggregate("AVG", df["col"])  # type: ignore[arg-type]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call_avg) for _ in range(2)]
        errors = []
        for f in futures:
            try:
                f.result()
            except Exception as e:
                errors.append(e)

    assert len(errors) == 2

    ctrl.cleanup()
    worker.close()
    ctrl.close()
