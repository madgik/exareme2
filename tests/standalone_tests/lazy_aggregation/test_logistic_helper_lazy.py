import numpy as np

from exaflow.algorithms.exareme3.lazy_aggregation import RecordingAggClient
from exaflow.algorithms.exareme3.library.logistic_common import (
    run_distributed_logistic_regression,
)


def test_logistic_helper_batches_globals():
    agg = RecordingAggClient()
    # Simple balanced case to avoid BadUserInput
    X = np.array([[0.0], [0.0], [0.0], [0.0]], dtype=float)
    y = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=float)

    result = run_distributed_logistic_regression(agg, X, y)

    assert result["n_obs"] == 4
    assert result["y_sum"] == 2.0
    # Expect batch of two initial sums (n_obs, y_sum) and a batch of three inside the loop (grad/H/ll).
    assert agg.calls == [("batch", 2), ("batch", 3)]
