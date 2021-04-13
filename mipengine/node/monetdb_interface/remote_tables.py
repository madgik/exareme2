from typing import List

from mipengine.node.monetdb_interface import common_actions
from mipengine.node.monetdb_interface.common_actions import connection
from mipengine.node.monetdb_interface.common_actions import (
    convert_schema_to_sql_query_format,
)
from mipengine.node.monetdb_interface.common_actions import cursor
from mipengine.common.node_tasks_DTOs import TableInfo
from mipengine.common.sql_injection_guard import sql_injection_guard


def get_remote_tables_names(context_id: str) -> List[str]:
    return common_actions.get_tables_names("remote", context_id)


# @protect_from_sql_injection
def create_remote_table(table_info: TableInfo, db_location: str, db_name: str):

    columns_schema = convert_schema_to_sql_query_format(table_info.schema)
    query = f"CREATE REMOTE TABLE {table_info.name} ( {columns_schema}) ON 'mapi:monetdb://{db_location}/{db_name}' WITH USER 'monetdb' PASSWORD 'monetdb' "

    cursor.execute(query)

    connection.commit()
