from typing import List

from mipengine.filters import build_filter_clause
from mipengine.node.monetdb_interface.common_actions import get_table_names
from mipengine.node.monetdb_interface.monet_db_connection import MonetDB


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
