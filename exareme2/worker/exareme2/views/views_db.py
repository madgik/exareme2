from typing import List
from typing import Optional

from exareme2.data_filters import build_filter_clause
from exareme2.worker.exareme2.monetdb.guard import is_list_of_identifiers
from exareme2.worker.exareme2.monetdb.guard import is_primary_data_table
from exareme2.worker.exareme2.monetdb.guard import is_valid_filter
from exareme2.worker.exareme2.monetdb.guard import sql_injection_guard
from exareme2.worker.exareme2.monetdb.monetdb_facade import db_execute_and_fetchall
from exareme2.worker.exareme2.monetdb.monetdb_facade import db_execute_query
from exareme2.worker.exareme2.tables.tables_db import get_table_names
from exareme2.worker.exareme2.tables.tables_db import get_table_schema
from exareme2.worker_communication import ColumnInfo
from exareme2.worker_communication import InsufficientDataError
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import TableType


def get_view_names(context_id: str) -> List[str]:
    return get_table_names(TableType.VIEW, context_id)


@sql_injection_guard(
    view_name=str.isidentifier,
    table_name=is_primary_data_table,
    columns=is_list_of_identifiers,
    filters=is_valid_filter,
    minimum_row_count=None,
    check_min_rows=None,
)
def create_view(
    view_name: str,
    table_name: str,
    columns: List[str],
    filters: Optional[dict],
    minimum_row_count: int,
    check_min_rows=False,
) -> TableInfo:
    filter_clause = ""
    if filters:
        filter_clause = f"WHERE {build_filter_clause(filters)}"
    columns_clause = ", ".join([f'"{column}"' for column in columns])

    view_creation_query = f"""
        CREATE VIEW {view_name}
        AS SELECT {columns_clause}
        FROM {table_name}
        {filter_clause}
        """

    db_execute_query(view_creation_query)

    view_rows_query_result = db_execute_and_fetchall(
        f"""
        SELECT COUNT(*)
        FROM {view_name}
        """
    )
    view_rows_result_row = view_rows_query_result[0]
    view_rows_count = view_rows_result_row[0]

    if view_rows_count < 1 or (check_min_rows and view_rows_count < minimum_row_count):
        db_execute_query(f"""DROP VIEW {view_name}""")
        raise InsufficientDataError(
            f"Query: {view_creation_query} creates an "
            f"insufficient data view. ({view_name=} has been dropped)"
        )

    view_schema = get_table_schema(view_name)
    ordered_view_schema = _get_ordered_table_schema(view_schema, columns)

    return TableInfo(
        name=view_name,
        schema_=ordered_view_schema,
        type_=TableType.VIEW,
    )


def _get_ordered_table_schema(
    table_schema: TableSchema, ordered_columns: List[str]
) -> TableSchema:
    ordered_column_infos = [
        _get_column_info_from_schema(table_schema, column_name)
        for column_name in ordered_columns
    ]
    return TableSchema(columns=ordered_column_infos)


def _get_column_info_from_schema(
    table_schema: TableSchema, column_name: str
) -> ColumnInfo:
    for column in table_schema.columns:
        if column.name == column_name:
            return column
    else:
        raise ValueError(f"{column_name=} does not exist in {table_schema=}.")
