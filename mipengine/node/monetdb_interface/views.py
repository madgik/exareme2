from typing import List

from mipengine.filters import build_filter_clause
from mipengine.node import config as node_config
from mipengine.node.monetdb_interface.common_actions import get_table_names
from mipengine.node.monetdb_interface.monet_db_connection import MonetDBPool
from mipengine.node_exceptions import InsufficientDataError
from mipengine.node_tasks_DTOs import TableType

MINIMUM_ROW_COUNT = node_config.privacy.minimum_row_count


def get_view_names(context_id: str) -> List[str]:
    return get_table_names(TableType.VIEW, context_id)


def create_view(
    view_name: str,
    table_name: str,
    columns: List[str],
    filters: dict,
    check_min_rows=False,
):
    filter_clause = ""
    if filters:
        filter_clause = f"WHERE {build_filter_clause(filters)}"
    columns_clause = ", ".join(columns)

    view_creation_query = f"""
        CREATE VIEW {view_name}
        AS SELECT {columns_clause}
        FROM {table_name}
        {filter_clause}
        """

    MonetDBPool().execute(view_creation_query)

    if check_min_rows:
        view_rows_query_result = MonetDBPool().execute_and_fetchall(
            f"""
            SELECT COUNT(*)
            FROM {view_name}
            """
        )
        view_rows_result_row = view_rows_query_result[0]
        view_rows_count = view_rows_result_row[0]

        if view_rows_count < MINIMUM_ROW_COUNT:
            MonetDBPool().execute(f"""DROP VIEW {view_name}""")
            raise InsufficientDataError(
                f"The following view has less rows than the PRIVACY_THRESHOLD({MINIMUM_ROW_COUNT}):  {view_creation_query}"
            )
