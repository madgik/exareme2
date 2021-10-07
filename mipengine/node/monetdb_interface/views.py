from typing import List

from mipengine.filters import build_filter_clause
from mipengine.node.monetdb_interface.common_actions import get_table_names
from mipengine.node.monetdb_interface.monet_db_connection import MonetDB

PRIVACY_MAGIC_NUMBER = 10


def get_view_names(context_id: str) -> List[str]:
    return get_table_names("view", context_id)


def create_view(
    view_name: str,
    table_name: str,
    columns: List[str],
    filters: dict,
):
    filter_clause = ""
    if filters:
        filter_clause = f"WHERE {build_filter_clause(filters)}"
    columns_clause = ", ".join(columns)

    MonetDB().execute(
        f"""
        CREATE VIEW {view_name}
        AS SELECT {columns_clause}
        FROM {table_name}
        {filter_clause}
        """
    )

    view_rows_query_result = MonetDB().execute_and_fetchall(
        f"""
        SELECT COUNT(*)
        FROM {view_name}
        """
    )
    view_rows_result_row = view_rows_query_result[0]
    view_rows_count = view_rows_result_row[0]

    if view_rows_count < PRIVACY_MAGIC_NUMBER:
        raise ValueError("Privacy Error!!!")
