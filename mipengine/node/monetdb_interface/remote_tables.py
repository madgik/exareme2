from typing import List

from mipengine.common.node_tasks_DTOs import TableInfo
from mipengine.common.validate_identifier_names import validate_identifier_names
from mipengine.node.monetdb_interface.common_actions import get_table_names
from mipengine.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from mipengine.node.monetdb_interface.monet_db_connection import MonetDB


def get_remote_table_names(context_id: str) -> List[str]:
    return get_table_names("remote", context_id)


@validate_identifier_names
def create_remote_table(table_info: TableInfo, url: str):
    columns_schema = convert_schema_to_sql_query_format(table_info.schema)
    MonetDB().execute(
        f"""
        CREATE REMOTE TABLE {table_info.name}
        ( {columns_schema}) on '{url}'
        WITH USER 'monetdb' PASSWORD 'monetdb'
        """
    )
