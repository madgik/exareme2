from typing import List

from mipengine.node import config as node_config
from mipengine.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from mipengine.node.monetdb_interface.common_actions import get_table_names
from mipengine.node.monetdb_interface.monet_db_connection import MonetDBPool
from mipengine.node_tasks_DTOs import TableType


def get_remote_table_names(context_id: str) -> List[str]:
    return get_table_names(TableType.REMOTE, context_id)


def create_remote_table(name, schema, monetdb_socket_address: str):
    columns_schema = convert_schema_to_sql_query_format(schema)
    MonetDBPool().execute(
        f"""
        CREATE REMOTE TABLE {name}
        ( {columns_schema}) ON 'mapi:monetdb://{monetdb_socket_address}/{node_config.monetdb.database}'
        WITH USER 'monetdb' PASSWORD 'monetdb'
        """
    )
