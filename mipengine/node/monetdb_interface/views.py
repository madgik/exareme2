from typing import List

from mipengine.common.sql_injection_guard import sql_injection_guard
from mipengine.node.monetdb_interface import common_actions
from mipengine.node.monetdb_interface.common_actions import connection
from mipengine.node.monetdb_interface.common_actions import cursor

DATA_TABLE_PRIMARY_KEY = "row_id"


def get_views_names(context_id: str) -> List[str]:
    return common_actions.get_tables_names("view", context_id)


@sql_injection_guard
def create_view(
    view_name: str, pathology: str, datasets: List[str], columns: List[str]
):
    # TODO: Add filters argument
    dataset_names = ",".join(f"'{dataset}'" for dataset in datasets)
    columns = ", ".join(columns)
    cursor.execute(
        f"""CREATE VIEW {view_name}
        AS SELECT {DATA_TABLE_PRIMARY_KEY}, {columns}
        FROM {pathology}_data
        WHERE dataset IN ({dataset_names})"""
    )
    connection.commit()
