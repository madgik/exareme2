from __future__ import annotations

from typing import Iterable
from typing import List

import numpy as np
import pyarrow as pa

from exareme2.algorithms.utils.inputdata_utils import Inputdata
from exareme2.data_filters import build_filter_clause


def struct_list_to_matrix(struct_lists: pa.ChunkedArray) -> np.ndarray:
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


def primary_table_name(data_model: str) -> str:
    sanitized = data_model
    for ch in (":", "-", "."):
        sanitized = sanitized.replace(ch, "_")
    return f'"{sanitized}__primary_data"'


def struct_pack_expression(columns: Iterable[str]) -> str:
    assignments = ", ".join(
        f"{quote_identifier(column)} := {quote_identifier(column)}"
        for column in columns
    )
    return f"struct_pack({assignments})"


def empty_struct_list_literal(columns: Iterable[str]) -> str:
    struct_fields = ", ".join(
        f"{quote_identifier(column)} DOUBLE" for column in columns
    )
    return f"array[]::STRUCT({struct_fields})[]"


def build_where_clause(inputdata: Inputdata, *, required_columns: Iterable[str]) -> str:
    clauses: List[str] = []
    datasets = sorted(
        {
            *(inputdata.datasets or []),
            *(inputdata.validation_datasets or []),
        }
    )
    if datasets:
        datasets_clause = ", ".join(quote_literal(value) for value in datasets)
        clauses.append(f'{quote_identifier("dataset")} IN ({datasets_clause})')

    if inputdata.filters:
        clauses.append(build_filter_clause(inputdata.filters))

    not_null_clause = " AND ".join(
        f"{quote_identifier(column)} IS NOT NULL" for column in required_columns
    )
    if not_null_clause:
        clauses.append(not_null_clause)

    if not clauses:
        return ""
    return "WHERE " + " AND ".join(clauses)


def quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def quote_literal(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"
