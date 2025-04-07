import concurrent.futures
import uuid

import numpy as np
import pandas as pd
import pytest

from exareme2.aggregator.aggregator_client import AggregationClient
from exareme2.aggregator.constants import AGG


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
        return aggregator_client.aggregate(AGG.SUM, df["col1"])

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
        return aggregator_client.aggregate(AGG.MIN, df["col1"])

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
        return aggregator_client.aggregate(AGG.MAX, df["col1"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(call_max) for _ in range(2)]
        results = [f.result() for f in futures]

    for result in results:
        assert (
            result == expected_first
        ), "Multi-column max aggregation failed for first column."


# === Multiple Aggregations in a Single Test ===


def test_multiple_aggregations(aggregator_client):
    # In this test we run three different aggregation operations concurrently.
    # For each operation we create a DataFrame column.
    df_sum = pd.DataFrame({"col": [1, 2, 3]})
    df_min = pd.DataFrame({"col": [1, 2, 3]})
    df_max = pd.DataFrame({"col": [1, 2, 3]})

    expected_sum = 2 * np.sum(df_sum["col"])
    expected_min = np.min(df_min["col"])
    expected_max = np.max(df_max["col"])

    def call_sum():
        return aggregator_client.sum(df_sum["col"])

    def call_min():
        return aggregator_client.min(df_min["col"])

    def call_max():
        return aggregator_client.max(df_max["col"])

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # Submit two tasks per aggregation operation.
        for _ in range(2):
            tasks.append(executor.submit(call_sum))
            tasks.append(executor.submit(call_min))
            tasks.append(executor.submit(call_max))
        results = [future.result() for future in tasks]

    # Group results in sets of three (each set corresponds to sum, min, max).
    sum_results = results[0::3]
    min_results = results[1::3]
    max_results = results[2::3]

    for r in sum_results:
        assert r == expected_sum, "Multiple aggregation: Sum result mismatch."
    for r in min_results:
        assert r == expected_min, "Multiple aggregation: Min result mismatch."
    for r in max_results:
        assert r == expected_max, "Multiple aggregation: Max result mismatch."
