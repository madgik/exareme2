from typing import List

from mipengine.node.monetdb_interface import common
from mipengine.node.monetdb_interface.common import connection
from mipengine.node.monetdb_interface.common import convert_schema_to_sql_query_format
from mipengine.node.monetdb_interface.common import cursor
from mipengine.node.tasks.data_classes import TableInfo
from mipengine.utils.validate_identifier_names import validate_identifier_names


def get_remote_tables_names(context_id: str) -> List[str]:
    return common.get_tables_names("remote", context_id)


@validate_identifier_names
def create_remote_table(table_info: TableInfo, url: str):
    columns_schema = convert_schema_to_sql_query_format(table_info.schema)
    cursor.execute(
        f"CREATE REMOTE TABLE {table_info.name} ( {columns_schema}) on '{url}' WITH USER 'monetdb' PASSWORD 'monetdb'")
    connection.commit()
