from __future__ import annotations

from typing import Iterable

import duckdb
import numpy as np
import pyarrow as pa
from duckdb import functional
from duckdb import typing as sqltypes

from exaflow.algorithms.exaflow.exaflow_udf_aggregation_client_interface import (
    ExaflowUDFAggregationClientI,
)
from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.worker import config as worker_config
from exaflow.worker.exaflow.duckdb._utils import build_where_clause
from exaflow.worker.exaflow.duckdb._utils import empty_struct_list_literal
from exaflow.worker.exaflow.duckdb._utils import primary_table_name
from exaflow.worker.exaflow.duckdb._utils import struct_list_to_matrix
from exaflow.worker.exaflow.duckdb._utils import struct_pack_expression
from exaflow.worker_communication import BadUserInput

_ARROW_RESULT_TYPE = pa.struct(
    [
        ("n_obs", pa.int64()),
        ("eigenvalues", pa.list_(pa.float64())),
        ("eigenvectors", pa.list_(pa.list_(pa.float64()))),
    ]
)

_DUCKDB_RESULT_TYPE = duckdb.struct_type(
    {
        "n_obs": sqltypes.BIGINT,
        "eigenvalues": duckdb.list_type(sqltypes.DOUBLE),
        "eigenvectors": duckdb.list_type(duckdb.list_type(sqltypes.DOUBLE)),
    }
)


def run_pca(inputdata: Inputdata, agg_client: ExaflowUDFAggregationClientI) -> dict:
    if not agg_client:
        raise RuntimeError("Aggregation client is required for PCA execution.")

    if not inputdata.y:
        raise BadUserInput("PCA requires at least one variable in 'y'.")

    query = _build_duckdb_query(inputdata)
    with duckdb.connect(worker_config.duckdb.path, read_only=True) as conn:
        register_pca_aggregation_arrow_udf(conn, agg_client=agg_client)
        row = conn.execute(query).fetchone()

    if not row or any(value is None for value in row):
        raise BadUserInput(
            "No rows matched the provided datasets and filters for PCA computation."
        )

    n_obs, eigenvalues, eigenvectors = row
    return {
        "n_obs": int(n_obs),
        "eigenvalues": list(eigenvalues),
        "eigenvectors": [list(vector) for vector in eigenvectors],
    }


def register_pca_aggregation_arrow_udf(
    conn: duckdb.DuckDBPyConnection,
    *,
    agg_client: ExaflowUDFAggregationClientI,
    function_name: str = "pca_udf_agg_arrow",
) -> None:
    def _udf(samples: pa.ChunkedArray) -> pa.Array:
        matrix = struct_list_to_matrix(samples)
        result = _pca_with_aggregation_client(matrix, agg_client)
        return pa.array([result], type=_ARROW_RESULT_TYPE)

    conn.create_function(
        function_name,
        _udf,
        return_type=_DUCKDB_RESULT_TYPE,
        type=functional.ARROW,
        side_effects=True,
    )


def _pca_with_aggregation_client(
    matrix: np.ndarray, agg_client: ExaflowUDFAggregationClientI
) -> dict:
    if matrix.size:
        finite_mask = np.all(np.isfinite(matrix), axis=1)
        matrix = matrix[finite_mask]

    n_obs = int(len(matrix))
    num_features = matrix.shape[1] if matrix.ndim == 2 else 0

    if num_features == 0:
        raise BadUserInput("PCA requires at least one numerical variable.")

    sx = np.einsum("ij->j", matrix) if n_obs else np.zeros(num_features, dtype=float)
    sxx = (
        np.einsum("ij,ij->j", matrix, matrix)
        if n_obs
        else np.zeros(num_features, dtype=float)
    )

    total_n_obs = agg_client.sum([float(n_obs)])[0]
    total_sx = np.asarray(agg_client.sum(sx.tolist()), dtype=float)
    total_sxx = np.asarray(agg_client.sum(sxx.tolist()), dtype=float)

    if total_n_obs <= 1:
        raise BadUserInput(
            "PCA requires at least two valid rows across all workers after filtering."
        )

    means = total_sx / total_n_obs
    variances = (total_sxx - total_n_obs * means**2) / (total_n_obs - 1)
    variances = np.maximum(variances, 0.0)
    sigmas = np.sqrt(variances)
    zero_sigma = sigmas == 0
    if np.any(zero_sigma):
        sigmas = sigmas.copy()
        sigmas[zero_sigma] = 1.0

    standardized = (
        (matrix - means) / sigmas if n_obs else np.empty((0, len(sigmas)), dtype=float)
    )
    gramian = np.einsum("ji,jk->ik", standardized, standardized)
    total_gramian = np.asarray(agg_client.sum(gramian.tolist()), dtype=float)
    covariance = total_gramian / (total_n_obs - 1)

    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real.T
    return {
        "n_obs": int(total_n_obs),
        "eigenvalues": eigenvalues.tolist(),
        "eigenvectors": eigenvectors.tolist(),
    }


def _build_duckdb_query(inputdata: Inputdata) -> str:
    table_name = primary_table_name(inputdata.data_model)
    sample_expr = struct_pack_expression(inputdata.y)
    where_clause = build_where_clause(inputdata, required_columns=inputdata.y or [])
    empty_literal = empty_struct_list_literal(inputdata.y)
    samples_expr = f"coalesce(array_agg(sample), {empty_literal})"

    return f"""
WITH filtered AS (
    SELECT {sample_expr} AS sample
    FROM {table_name}
    {where_clause}
),
aggregated AS (
    SELECT pca_udf_agg_arrow({samples_expr}) AS result
    FROM filtered
)
SELECT result.n_obs, result.eigenvalues, result.eigenvectors
FROM aggregated
LIMIT 1
"""
