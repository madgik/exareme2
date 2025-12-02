import numpy as np

from exaflow.algorithms.exareme3.exaflow_registry import exaflow_udf
from exaflow.algorithms.exareme3.library.lazy_aggregation import RecordingAggClient


def test_exaflow_udf_enables_lazy_by_default():
    calls = RecordingAggClient()

    @exaflow_udf(with_aggregation_server=True)
    def udf(agg_client):
        a = agg_client.sum(np.array([1.0], dtype=float))
        b = agg_client.sum(np.array([2.0], dtype=float))
        return float(np.asarray(a, dtype=float)[0] + np.asarray(b, dtype=float)[0])

    total = udf(calls)
    assert total == 3.0
    assert calls.calls == [("batch", 2)]


def test_exaflow_udf_lazy_can_be_disabled():
    calls = RecordingAggClient()

    @exaflow_udf(with_aggregation_server=True, enable_lazy_aggregation=False)
    def udf(agg_client):
        a = agg_client.sum(np.array([1.0], dtype=float))
        b = agg_client.sum(np.array([2.0], dtype=float))
        return float(np.asarray(a, dtype=float)[0] + np.asarray(b, dtype=float)[0])

    total = udf(calls)
    assert total == 3.0
    assert calls.calls == [("sum", 1), ("sum", 1)]


def test_exaflow_udf_custom_client_name():
    calls = RecordingAggClient()

    @exaflow_udf(with_aggregation_server=True, agg_client_name="client")
    def udf(client):
        x = client.sum(np.array([4.0], dtype=float))
        y = client.sum(np.array([5.0], dtype=float))
        return float(np.asarray(x, dtype=float)[0] + np.asarray(y, dtype=float)[0])

    total = udf(calls)
    assert total == 9.0
    assert calls.calls == [("batch", 2)]
