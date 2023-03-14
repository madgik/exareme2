from typing import List

from mipengine.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from mipengine.node.monetdb_interface.common_actions import get_table_names
from mipengine.node.monetdb_interface.guard import is_socket_address
from mipengine.node.monetdb_interface.guard import is_valid_table_schema
from mipengine.node.monetdb_interface.guard import sql_injection_guard
from mipengine.node.monetdb_interface.monet_db_facade import db_execute_query
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import TableType


def get_remote_table_names(context_id: str) -> List[str]:
    return get_table_names(TableType.REMOTE, context_id)


@sql_injection_guard(
    name=str.isidentifier,
    monetdb_socket_address=is_socket_address,
    schema=is_valid_table_schema,
    username=str.isidentifier,
    password=str.isidentifier,
)
def create_remote_table(
    name: str,
    schema: TableSchema,
    monetdb_socket_address: str,
    username: str,
    password: str,
):
    columns_schema = convert_schema_to_sql_query_format(schema)
    db_execute_query(
        f"""
        CREATE REMOTE TABLE {username}.{name}
        ( {columns_schema}) ON 'mapi:monetdb://{monetdb_socket_address}/db/{username}/{name}'
        WITH USER '{username}' PASSWORD '{password}'
        """
    )
