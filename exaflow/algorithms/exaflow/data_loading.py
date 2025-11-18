from __future__ import annotations

from typing import Iterable
from typing import Sequence
from typing import Set

import pandas as pd

from exaflow.algorithms.utils.inputdata_utils import Inputdata
from exaflow.algorithms.utils.inputdata_utils import fetch_data
from exaflow.data_filters import build_filter_clause


def load_algorithm_dataframe(
    inputdata: Inputdata,
    csv_paths: Sequence[str],
    *,
    dropna: bool = True,
    include_dataset: bool = False,
    extra_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Shared data loader for exaflow algorithms.

    It mirrors the existing fetch_data behaviour when `use_duckdb` is False
    and optionally pulls rows via DuckDB when `use_duckdb` is True. On any
    DuckDB error it falls back to fetch_data to maintain current behaviour.
    """
    required_columns: Set[str] = set(inputdata.x or []) | set(inputdata.y or [])
    if include_dataset:
        required_columns.add("dataset")
    if extra_columns:
        required_columns.update(extra_columns)

    return _fetch_with_duckdb(inputdata, required_columns, dropna=dropna)


def _fetch_with_duckdb(
    inputdata: Inputdata, required_columns: Set[str], *, dropna: bool
) -> pd.DataFrame:
    import duckdb

    from exaflow.worker import config as worker_config
    from exaflow.worker.exaflow.duckdb._utils import primary_table_name
    from exaflow.worker.exaflow.duckdb._utils import quote_identifier
    from exaflow.worker.exaflow.duckdb._utils import quote_literal

    datasets = (inputdata.datasets or []) + (
        inputdata.validation_datasets if inputdata.validation_datasets else []
    )

    table_name = primary_table_name(inputdata.data_model)
    columns = sorted(required_columns) if required_columns else ["*"]
    select_columns = ", ".join(quote_identifier(col) for col in columns)

    where_clauses = []
    datasets_clause = ", ".join(quote_literal(value) for value in datasets)
    where_clauses.append(f'{quote_identifier("dataset")} IN ({datasets_clause})')

    if inputdata.filters:
        where_clauses.append(build_filter_clause(inputdata.filters))

    if dropna and required_columns:
        not_null_clause = " AND ".join(
            f"{quote_identifier(col)} IS NOT NULL" for col in required_columns
        )
        where_clauses.append(not_null_clause)

    where_sql = ""
    if where_clauses:
        where_sql = " WHERE " + " AND ".join(f"({clause})" for clause in where_clauses)

    query = f"SELECT {select_columns} FROM {table_name}{where_sql}"
    with duckdb.connect(worker_config.duckdb.path, read_only=True) as conn:
        df = conn.execute(query).fetch_df()

    # Drop duplicate column names to mirror fetch_data behaviour when reading CSVs.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    return df
