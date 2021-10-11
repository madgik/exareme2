from typing import List

from mipengine.filters import build_filter_clause
from mipengine.node import config as node_config
from mipengine.node.monetdb_interface.common_actions import get_table_names
from mipengine.node.monetdb_interface.monet_db_connection import MonetDB
from mipengine.node_tasks_DTOs import PrivacyError

PRIVACY_THRESHOLD = node_config.privacy_threshold


def get_view_names(context_id: str) -> List[str]:
    return get_table_names("view", context_id)


def create_view(
    view_name: str,
    table_name: str,
    columns: List[str],
    filters: dict,
    privacy_protection=False,
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

    MonetDB().execute(view_creation_query)

    if privacy_protection:
        view_rows_query_result = MonetDB().execute_and_fetchall(
            f"""
            SELECT COUNT(*)
            FROM {view_name}
            """
        )
        view_rows_result_row = view_rows_query_result[0]
        view_rows_count = view_rows_result_row[0]

        if view_rows_count < PRIVACY_THRESHOLD:
            raise PrivacyError(
                f"The following view has less rows than the PRIVACY_THRESHOLD({PRIVACY_THRESHOLD}):  {view_creation_query}"
            )
