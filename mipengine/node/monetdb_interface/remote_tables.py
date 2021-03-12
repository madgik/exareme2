from typing import List

from pymonetdb import Connection
from pymonetdb.sql.cursors import Cursor

from mipengine.node.monetdb_interface import common_actions
from mipengine.node.monetdb_interface.common_actions import convert_schema_to_sql_query_format
from mipengine.common.node_tasks_DTOs import TableInfo
from mipengine.common.validate_identifier_names import validate_identifier_names
from mipengine.node.monetdb_interface.connection_pool import execute_with_occ


def get_remote_tables_names(cursor: Cursor, context_id: str) -> List[str]:
    return common_actions.get_tables_names(cursor, "remote", context_id)


@validate_identifier_names
def create_remote_table(connection:Connection, cursor: Cursor, table_info: TableInfo, url: str):
    columns_schema = convert_schema_to_sql_query_format(table_info.schema)
    execute_with_occ(connection, cursor,
        f"CREATE REMOTE TABLE {table_info.name} ( {columns_schema}) on '{url}' WITH USER 'monetdb' PASSWORD 'monetdb'")
