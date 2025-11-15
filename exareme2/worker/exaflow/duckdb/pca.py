from __future__ import annotations

from typing import Iterable
from typing import List

import duckdb
import numpy as np
import pyarrow as pa
from duckdb import functional
from duckdb import typing as sqltypes

from exareme2.algorithms.exaflow.exaflow_udf_aggregation_client_interface import (
    ExaflowUDFAggregationClientI,
)
from exareme2.algorithms.utils.inputdata_utils import Inputdata
from exareme2.data_filters import build_filter_clause
from exareme2.worker import config as worker_config
from exareme2.worker_communication import BadUserInput

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
        matrix = _struct_list_to_matrix(samples)
        result = _pca_with_aggregation_client(matrix, agg_client)
        return pa.array([result], type=_ARROW_RESULT_TYPE)

    conn.create_function(
        function_name,
        _udf,
        return_type=_DUCKDB_RESULT_TYPE,
        type=functional.ARROW,
        side_effects=True,
    )


def _struct_list_to_matrix(struct_lists: pa.ChunkedArray) -> np.ndarray:
    """
    Convert a DuckDB list of structs into a 2D numpy array.
    """
    arr = struct_lists.combine_chunks()
    if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        struct_array = arr.flatten()
    else:
        struct_array = arr

    num_rows = len(struct_array)
    num_fields = struct_array.type.num_fields
    if num_fields == 0:
        return np.empty((num_rows, 0), dtype=float)

    columns: List[np.ndarray] = []
    for idx in range(num_fields):
        field_array = struct_array.field(idx)
        columns.append(
            np.asarray(field_array.to_numpy(zero_copy_only=False), dtype=float)
        )

    if not columns:
        return np.empty((num_rows, 0), dtype=float)
    return np.column_stack(columns)


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
    table_name = _primary_table_name(inputdata.data_model)
    sample_expr = _struct_pack_expression(inputdata.y)
    where_clause = _build_where_clause(inputdata)
    empty_literal = _empty_struct_list_literal(inputdata.y)
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


def _build_where_clause(inputdata: Inputdata) -> str:
    clauses: List[str] = []
    datasets = sorted(
        {
            *(inputdata.datasets or []),
            *(inputdata.validation_datasets or []),
        }
    )
    if datasets:
        datasets_clause = ", ".join(_quote_literal(value) for value in datasets)
        clauses.append(f'{_quote_identifier("dataset")} IN ({datasets_clause})')

    if inputdata.filters:
        clauses.append(build_filter_clause(inputdata.filters))

    if not clauses:
        return ""
    return "WHERE " + " AND ".join(clauses)


def _primary_table_name(data_model: str) -> str:
    sanitized = data_model
    for ch in (":", "-", "."):
        sanitized = sanitized.replace(ch, "_")
    return f'"{sanitized}__primary_data"'


def _struct_pack_expression(columns: Iterable[str]) -> str:
    assignments = ", ".join(
        f"{_quote_identifier(column)} := {_quote_identifier(column)}"
        for column in columns
    )
    return f"struct_pack({assignments})"


def _empty_struct_list_literal(columns: Iterable[str]) -> str:
    struct_fields = ", ".join(
        f"{_quote_identifier(column)} DOUBLE" for column in columns
    )
    return f"array[]::STRUCT({struct_fields})[]"


def _quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _quote_literal(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"
