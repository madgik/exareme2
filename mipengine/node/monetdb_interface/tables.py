from typing import List

from mipengine.common.sql_injection_guard import sql_injection_guard
from mipengine.node.monetdb_interface import common_actions
from mipengine.node.monetdb_interface.common_actions import connection
from mipengine.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from mipengine.node.monetdb_interface.common_actions import cursor
from mipengine.common.node_tasks_DTOs import TableInfo


def get_tables_names(context_id: str) -> List[str]:
    return common_actions.get_tables_names("normal", context_id)


@sql_injection_guard
def create_table(table_info: TableInfo):
    columns_schema = convert_schema_to_sql_query_format(table_info.schema)
    cursor.execute(f"CREATE TABLE {table_info.name} ( {columns_schema} )")
    connection.commit()
