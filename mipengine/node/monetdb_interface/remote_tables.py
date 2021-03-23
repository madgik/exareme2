from typing import List

from mipengine.node.monetdb_interface import common_action
from mipengine.node.monetdb_interface.common_action import connection
from mipengine.node.monetdb_interface.common_action import convert_schema_to_sql_query_format
from mipengine.node.monetdb_interface.common_action import cursor
from mipengine.common.node_tasks_DTOs import TableInfo
from mipengine.common.validate_identifier_names import validate_identifier_names


def get_remote_tables_names(context_id: str) -> List[str]:
    return common_action.get_tables_names("remote", context_id)


# @validate_identifier_names
def create_remote_table(table_info: TableInfo, url: str, db_name: str):

    columns_schema = convert_schema_to_sql_query_format(table_info.schema)
    query = f"CREATE REMOTE TABLE {table_info.name} ( {columns_schema}) ON 'mapi:monetdb://{url}/{db_name}' WITH USER 'monetdb' PASSWORD 'monetdb'"

    cursor.execute(query)

    connection.commit()
