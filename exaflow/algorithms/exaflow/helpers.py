from __future__ import annotations

"""
Backward-compatible helper utilities inspired by exareme2.

These are intentionally lightweight and adapted to exaflow:
- `get_transfer_data` simply returns the aggregated payload produced by
  `engine.run_algorithm_udf` (the first element of the results list).
- `sum_secure_transfers` provides a secure-sum UDF for numerical arrays via
  the aggregation server.
"""

from typing import Any

import numpy as np

from exaflow.algorithms.exaflow.exaflow_registry import exaflow_udf


def get_transfer_data(udf_results: list[Any]) -> Any:
    """
    In exaflow, `run_algorithm_udf` already returns decoded Python objects.
    For aggregation-server UDFs the payload is identical on all workers, so
    grabbing the first element matches the exareme2 helper semantics.
    """

    if not udf_results:
        raise ValueError("Empty UDF results; cannot extract transfer data.")
    return udf_results[0]


@exaflow_udf(with_aggregation_server=True)
def sum_secure_transfers(data, inputdata, agg_client, loctransf):
    """
    Securely sum a numeric vector or matrix across workers using the aggregation server.

    Parameters
    ----------
    loctransf : array-like
        Per-worker numeric payload (1D or 2D) to be summed element-wise.

    Returns
    -------
    list
        The summed array as a (possibly nested) Python list.
    """

    arr = np.asarray(loctransf, dtype=float)
    summed = agg_client.sum(arr)
    return np.asarray(summed).tolist()
