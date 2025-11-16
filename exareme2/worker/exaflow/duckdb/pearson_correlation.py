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
from exareme2.algorithms.exaflow.library.stats.stats import pearson_correlation
from exareme2.algorithms.utils.inputdata_utils import Inputdata
from exareme2.worker import config as worker_config
from exareme2.worker.exaflow.duckdb._utils import build_where_clause
from exareme2.worker.exaflow.duckdb._utils import empty_struct_list_literal
from exareme2.worker.exaflow.duckdb._utils import primary_table_name
from exareme2.worker.exaflow.duckdb._utils import struct_list_to_matrix
from exareme2.worker.exaflow.duckdb._utils import struct_pack_expression
from exareme2.worker_communication import BadUserInput

_UDF_FUNCTION_NAME = "pearson_udf_agg_arrow"

_ARROW_RESULT_TYPE = pa.struct(
    [
        ("n_obs", pa.int64()),
        ("correlations", pa.list_(pa.list_(pa.float64()))),
        ("p_values", pa.list_(pa.list_(pa.float64()))),
        ("ci_hi", pa.list_(pa.list_(pa.float64()))),
        ("ci_lo", pa.list_(pa.list_(pa.float64()))),
    ]
)

_DUCKDB_RESULT_TYPE = duckdb.struct_type(
    {
        "n_obs": sqltypes.BIGINT,
        "correlations": duckdb.list_type(duckdb.list_type(sqltypes.DOUBLE)),
        "p_values": duckdb.list_type(duckdb.list_type(sqltypes.DOUBLE)),
        "ci_hi": duckdb.list_type(duckdb.list_type(sqltypes.DOUBLE)),
        "ci_lo": duckdb.list_type(duckdb.list_type(sqltypes.DOUBLE)),
    }
)


def run_pearson(
    inputdata: Inputdata,
    agg_client: ExaflowUDFAggregationClientI,
    alpha: float,
) -> dict:
    if not agg_client:
        raise RuntimeError("Aggregation client is required for Pearson correlation.")

    if not inputdata.y:
        raise BadUserInput("Pearson correlation needs target variables in 'y'.")

    x_vars: List[str] = inputdata.x if inputdata.x else inputdata.y
    y_vars: List[str] = inputdata.y

    query = _build_duckdb_query(inputdata, x_vars=x_vars, y_vars=y_vars)
    with duckdb.connect(worker_config.duckdb.path, read_only=True) as conn:
        register_pearson_aggregation_arrow_udf(conn, agg_client=agg_client, alpha=alpha)
        row = conn.execute(query).fetchone()

    if not row or any(value is None for value in row):
        raise BadUserInput(
            "No rows matched the provided datasets and filters for Pearson correlation."
        )

    n_obs, correlations, p_values, ci_hi, ci_lo = row
    return {
        "n_obs": int(n_obs),
        "correlations": [list(values) for values in correlations],
        "p_values": [list(values) for values in p_values],
        "ci_hi": [list(values) for values in ci_hi],
        "ci_lo": [list(values) for values in ci_lo],
    }


def register_pearson_aggregation_arrow_udf(
    conn: duckdb.DuckDBPyConnection,
    *,
    agg_client: ExaflowUDFAggregationClientI,
    alpha: float,
    function_name: str = _UDF_FUNCTION_NAME,
) -> None:
    def _udf(x_samples: pa.ChunkedArray, y_samples: pa.ChunkedArray) -> pa.Array:
        x_matrix = struct_list_to_matrix(x_samples)
        y_matrix = struct_list_to_matrix(y_samples)
        x_matrix, y_matrix = _filter_finite_rows(x_matrix, y_matrix)

        try:
            result = pearson_correlation(
                agg_client=agg_client,
                x=x_matrix,
                y=y_matrix,
                alpha=alpha,
            )
        except ValueError as exc:
            raise BadUserInput(str(exc)) from exc

        return pa.array([result], type=_ARROW_RESULT_TYPE)

    conn.create_function(
        function_name,
        _udf,
        return_type=_DUCKDB_RESULT_TYPE,
        type=functional.ARROW,
        side_effects=True,
    )


def _filter_finite_rows(x_matrix: np.ndarray, y_matrix: np.ndarray):
    if x_matrix.shape[0] != y_matrix.shape[0]:
        raise BadUserInput(
            "Mismatched sample counts between x and y variables for Pearson correlation."
        )

    if x_matrix.size and y_matrix.size:
        finite_mask = np.all(np.isfinite(x_matrix), axis=1) & np.all(
            np.isfinite(y_matrix), axis=1
        )
        x_matrix = x_matrix[finite_mask]
        y_matrix = y_matrix[finite_mask]

    return x_matrix, y_matrix


def _build_duckdb_query(
    inputdata: Inputdata,
    *,
    x_vars: Iterable[str],
    y_vars: Iterable[str],
    function_name: str = _UDF_FUNCTION_NAME,
) -> str:
    table_name = primary_table_name(inputdata.data_model)
    x_sample_expr = struct_pack_expression(x_vars)
    y_sample_expr = struct_pack_expression(y_vars)
    where_clause = build_where_clause(inputdata, required_columns=[*x_vars, *y_vars])
    empty_x_literal = empty_struct_list_literal(x_vars)
    empty_y_literal = empty_struct_list_literal(y_vars)
    x_samples_expr = f"coalesce(array_agg(x_sample), {empty_x_literal})"
    y_samples_expr = f"coalesce(array_agg(y_sample), {empty_y_literal})"

    return f"""
WITH filtered AS (
    SELECT {x_sample_expr} AS x_sample, {y_sample_expr} AS y_sample
    FROM {table_name}
    {where_clause}
),
aggregated AS (
    SELECT {function_name}({x_samples_expr}, {y_samples_expr}) AS result
    FROM filtered
)
SELECT result.n_obs, result.correlations, result.p_values, result.ci_hi, result.ci_lo
FROM aggregated
LIMIT 1
"""
