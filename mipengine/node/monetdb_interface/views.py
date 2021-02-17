from typing import List

from mipengine.node.monetdb_interface import common
from mipengine.node.monetdb_interface.common import connection
from mipengine.node.monetdb_interface.common import cursor
from mipengine.node.tasks.data_classes import ColumnInfo


def get_views_names(context_id: str) -> List[str]:
    return common.get_tables_names("view", context_id)


def get_view_schema(view_name: str) -> List[ColumnInfo]:
    return common.get_table_schema('view', view_name)


def get_view_data(context_id: str) -> List[str]:
    return common.get_table_data("view", context_id)


def create_view(view_name: str, columns: List[str], datasets: List[str]):
    cursor.execute(
        f"CREATE VIEW {view_name} AS SELECT {', '.join(columns)} FROM data WHERE dataset IN ({str(datasets)[1:-1]})")
    connection.commit()


def clean_up_views(context_id: str = None):
    context_clause = ""
    if context_id is not None:
        context_clause = f"name LIKE '%{context_id.lower()}%' AND"

    cursor.execute(
        "SELECT name FROM tables "
        "WHERE"
        f" {context_clause} "
        f" type = 1 AND "
        "system = false")

    tables_names = [table[0] for table in cursor]
    for name in tables_names:
        cursor.execute(f"DROP VIEW if exists sys.{name}")
    connection.commit()
