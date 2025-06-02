import concurrent.futures
import uuid

import grpc
import numpy as np
import pandas as pd
import pytest

from exareme2.aggregator.constants import AggregationType
from exareme2.algorithms.exaflow.aggregator_client import AggregationClient


# Fixture to create an AggregatorClient and configure it to expect 2 workers.
@pytest.fixture(scope="session")
def aggregator_client(aggregator_service):
    client = AggregationClient(str(uuid.uuid4()))
    status = client.configure(2)
    assert status == "Configured", "Failed to configure aggregator with 2 workers."
    yield client
    client.cleanup()
    client.close()


# === Single-Column Tests ===
def test_sum_single_column(aggregator_client):
    # Create a DataFrame and pass its column (Series) to the client.
    df = pd.DataFrame({"col": [1, 2, 3]})
    # Each worker returns the sum of the column (i.e. 6) and the aggregated sum is 2 * 6 = 12.
    expected = 2 * np.sum(df["col"])

    def call_sum():
        return aggregator_client.sum(df["col"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call_sum) for _ in range(2)]
        results = [f.result() for f in futures]

    for result in results:
        assert result == expected, "Single column sum aggregation failed."


def test_min_single_column(aggregator_client):
    df = pd.DataFrame({"col": [1, 2, 3]})
    expected = np.min(df["col"])

    def call_min():
        return aggregator_client.min(df["col"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call_min) for _ in range(2)]
        results = [f.result() for f in futures]

    for result in results:
        assert result == expected, "Single column min aggregation failed."


def test_max_single_column(aggregator_client):
    df = pd.DataFrame({"col": [1, 2, 3]})
    expected = np.max(df["col"])

    def call_max():
        return aggregator_client.max(df["col"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call_max) for _ in range(2)]
        results = [f.result() for f in futures]

    for result in results:
        assert result == expected, "Single column max aggregation failed."


def test_count_single_column(aggregator_client):
    df = pd.DataFrame({"col": [1, 2, 3]})
    expected = 2 * float(len(df["col"]))

    def call_count():
        return aggregator_client.count(df["col"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call_count) for _ in range(2)]
        results = [f.result() for f in futures]

    for result in results:
        assert result == expected, "Single column count aggregation failed."


def test_avg_single_column(aggregator_client):
    # For the column with values [1,2,3,4]:
    #   Each worker computes sum=10 and count=4, so aggregated sum=20 and count=8, average=2.5.
    df = pd.DataFrame({"col": [1, 2, 3, 4]})
    expected = 2.5

    def call_avg():
        return aggregator_client.avg(df["col"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call_avg) for _ in range(2)]
        results = [f.result() for f in futures]

    for result in results:
        np.testing.assert_almost_equal(
            result, expected, err_msg="Single column avg aggregation failed."
        )


# === Multi-Column Tests (Directly Calling aggregate) ===


def test_sum_multi_column(aggregator_client):
    # Create a DataFrame with multiple columns.
    # In this example, we assume that the aggregator client aggregates the first column.
    df = pd.DataFrame({"col1": [10.0], "col2": [20.0], "col3": [30.0]})
    # Expected aggregated result for the first column: 10.0 + 10.0 = 20.0.
    expected_first = 2 * df["col1"].iloc[0]

    def call_sum():
        return aggregator_client.aggregate(AggregationType.SUM, df["col1"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call_sum) for _ in range(2)]
        results = [f.result() for f in futures]

    for result in results:
        assert (
            result == expected_first
        ), "Multi-column sum aggregation failed for first column."


def test_min_multi_column(aggregator_client):
    df = pd.DataFrame({"col1": [5.0], "col2": [50.0], "col3": [500.0]})
    expected_first = df["col1"].iloc[0]

    def call_min():
        return aggregator_client.aggregate(AggregationType.MIN, df["col1"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call_min) for _ in range(2)]
        results = [f.result() for f in futures]

    for result in results:
        assert (
            result == expected_first
        ), "Multi-column min aggregation failed for first column."


def test_max_multi_column(aggregator_client):
    df = pd.DataFrame({"col1": [15.0], "col2": [25.0], "col3": [35.0]})
    expected_first = df["col1"].iloc[0]

    def call_max():
        return aggregator_client.aggregate(AggregationType.MAX, df["col1"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call_max) for _ in range(2)]
        results = [f.result() for f in futures]

    for result in results:
        assert (
            result == expected_first
        ), "Multi-column max aggregation failed for first column."


def test_sequential_aggregations_same_request(aggregator_client):
    """
    Sequential aggregations using a single aggregator client.

    The client is initialized with a fixed request id and is configured to expect 2 worker responses.
    First, it performs a 'sum' aggregation with 2 concurrent calls simulating 2 workers.
    Then, it performs a 'min' aggregation with 2 concurrent calls.
    Each operation must complete successfully and yield the expected result.
    """
    # Prepare a sample DataFrame.
    df = pd.DataFrame({"col": [1, 2, 3]})

    # Expected results:
    # Sum aggregation: assume the logic returns 2 * sum when 2 worker responses are expected.
    expected_sum = 2 * np.sum(df["col"])
    # Min aggregation: simply the minimum value in the column.
    expected_min = np.min(df["col"])

    # --- SUM AGGREGATION ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_sum1 = executor.submit(aggregator_client.sum, df["col"])
        future_sum2 = executor.submit(aggregator_client.sum, df["col"])
        sum_results = [future_sum1.result(), future_sum2.result()]

    for r in sum_results:
        assert (
            r == expected_sum
        ), f"Sequential aggregation (sum): Expected {expected_sum}, got {r}."

    # --- MIN AGGREGATION ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_min1 = executor.submit(aggregator_client.min, df["col"])
        future_min2 = executor.submit(aggregator_client.min, df["col"])
        min_results = [future_min1.result(), future_min2.result()]

    for r in min_results:
        assert (
            r == expected_min
        ), f"Sequential aggregation (min): Expected {expected_min}, got {r}."


# --- Parallel Aggregations (Different Clients) ---
def test_parallel_aggregations_different_clients():
    """
    Parallel aggregations using different aggregator clients.

    Each aggregator client is instantiated with a unique request id and configured to expect 2 worker responses.
    Three clients are created: one for 'sum', one for 'min', and one for 'max'.
    For each client, 2 concurrent calls are made (simulating 2 workers for that operation).
    All results must match their respective expected values.
    """
    # Prepare a sample DataFrame.
    df = pd.DataFrame({"col": [1, 2, 3]})

    # Expected results:
    # For sum aggregation, assume the logic returns 2 * sum when 2 worker responses are expected.
    expected_sum = 2 * np.sum(df["col"])
    expected_min = np.min(df["col"])
    expected_max = np.max(df["col"])

    # Create separate aggregator clients (each with its own unique request id).
    client_sum = AggregationClient(request_id="parallel-test-sum")
    client_min = AggregationClient(request_id="parallel-test-min")
    client_max = AggregationClient(request_id="parallel-test-max")

    # Configure each client to expect 2 worker responses.
    client_sum.configure(2)
    client_min.configure(2)
    client_max.configure(2)

    def call_sum():
        return client_sum.sum(df["col"])

    def call_min():
        return client_min.min(df["col"])

    def call_max():
        return client_max.max(df["col"])

    # For each operation, simulate 2 workers concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        tasks = []
        for _ in range(2):
            tasks.append(executor.submit(call_sum))
            tasks.append(executor.submit(call_min))
            tasks.append(executor.submit(call_max))
        results = [future.result() for future in tasks]

    # Given the tasks were added as follows (per loop iteration):
    #   task0: sum, task1: min, task2: max, task3: sum, task4: min, task5: max.
    sum_results = [results[0], results[3]]
    min_results = [results[1], results[4]]
    max_results = [results[2], results[5]]

    for r in sum_results:
        assert (
            r == expected_sum
        ), f"Parallel aggregation (sum): Expected {expected_sum}, got {r}."
    for r in min_results:
        assert (
            r == expected_min
        ), f"Parallel aggregation (min): Expected {expected_min}, got {r}."
    for r in max_results:
        assert (
            r == expected_max
        ), f"Parallel aggregation (max): Expected {expected_max}, got {r}."


def test_high_concurrency_and_reset():
    """
    Create an aggregator client configured with 10 expected responses.
    Launch 10 concurrent calls for an aggregation operation (e.g. sum),
    verify that all get the correct result, and then ensure that after reset,
    another aggregation cycle works as expected.
    """
    # Prepare a DataFrame.
    df = pd.DataFrame({"col": list(range(1, 11))})  # Values 1 to 10.
    # Expected aggregated result (for sum, for example).
    expected_sum = 10 * np.sum(df["col"])

    # Create a client with a unique request id and 10 worker responses expected.
    client = AggregationClient(request_id="high-concurrency")
    client.configure(10)

    # First aggregation cycle.
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(client.sum, df["col"]) for _ in range(10)]
        results_cycle_1 = [f.result() for f in futures]
    for res in results_cycle_1:
        assert (
            res == expected_sum
        ), f"High concurrency test (cycle 1): Expected {expected_sum}, got {res}."

    # Second aggregation cycle (after reset).
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(client.sum, df["col"]) for _ in range(10)]
        results_cycle_2 = [f.result() for f in futures]
    for res in results_cycle_2:
        assert (
            res == expected_sum
        ), f"High concurrency test (cycle 2): Expected {expected_sum}, got {res}."


# --- Failed Aggregations---
def test_worker_timeout(aggregator_client):
    """
    Configure a client to expect 2 responses, but simulate only one worker responding.
    Expect the aggregation call to eventually time out with a DEADLINE_EXCEEDED error.
    """
    # Prepare input data.
    df = pd.DataFrame({"col": [1, 2, 3]})

    # Only one call is made even though 2 responses are expected.
    with pytest.raises(grpc.RpcError) as exc_info:
        aggregator_client.sum(df["col"])
    # Check that the error is a timeout.
    assert exc_info.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED


def test_mixed_computation_types(aggregator_client):
    """
    Two concurrent calls on the same client: one requesting 'sum' and one 'min'.
    Expect one call to fail because mixing aggregation types within a single aggregation cycle is disallowed.
    """
    df = pd.DataFrame({"col": [1, 2, 3]})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_sum = executor.submit(aggregator_client.sum, df["col"])
        future_min = executor.submit(aggregator_client.min, df["col"])
        errors = []
        results = []
        for future in (future_sum, future_min):
            try:
                results.append(future.result())
            except Exception as e:
                errors.append(e)

    # We expect exactly one error (the mismatched computation type).
    assert (
        len(errors) == 1
    ), "Expected one call to fail due to mismatched aggregation types."


def test_error_propagation_on_unsupported_type():
    """
    Create a client with 2 expected responses and invoke an unsupported aggregation operation.
    Both concurrent calls should receive an error (e.g. grpc.INTERNAL).
    """
    client = AggregationClient(request_id="error-propagation")
    client.configure(2)
    df = pd.DataFrame({"col": [1, 2, 3]})

    # Assume that 'avg' is not supported.
    def call_avg():
        # Here we assume an API call like aggregate(operation, data)
        return client.aggregate("AVG", df["col"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call_avg) for _ in range(2)]
        errors = []
        for future in futures:
            try:
                future.result()
            except Exception as e:
                errors.append(e)

    assert (
        len(errors) == 2
    ), "Expected both worker calls to fail for an unsupported aggregation type."
