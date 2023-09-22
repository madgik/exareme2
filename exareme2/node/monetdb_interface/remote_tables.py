from typing import List

from exareme2.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from exareme2.node.monetdb_interface.common_actions import get_table_names
from exareme2.node.monetdb_interface.guard import is_socket_address
from exareme2.node.monetdb_interface.guard import is_valid_table_schema
from exareme2.node.monetdb_interface.guard import sql_injection_guard
from exareme2.node.monetdb_interface.monet_db_facade import db_execute_query
from exareme2.node_tasks_DTOs import TableSchema
from exareme2.node_tasks_DTOs import TableType


def get_remote_table_names(context_id: str) -> List[str]:
    return get_table_names(TableType.REMOTE, context_id)


@sql_injection_guard(
    table_name=str.isidentifier,
    monetdb_socket_address=is_socket_address,
    schema=is_valid_table_schema,
    table_creator_username=str.isidentifier,
    public_username=str.isidentifier,
    public_password=str.isidentifier,
)
def create_remote_table(
    table_name: str,
    schema: TableSchema,
    monetdb_socket_address: str,
    table_creator_username: str,
    public_username: str,
    public_password: str,
):
    columns_schema = convert_schema_to_sql_query_format(schema)
    db_execute_query(
        f"""
        CREATE REMOTE TABLE {table_name}
        ( {columns_schema}) ON 'mapi:monetdb://{monetdb_socket_address}/db/{table_creator_username}/{table_name}'
        WITH USER '{public_username}' PASSWORD '{public_password}'
        """
    )
