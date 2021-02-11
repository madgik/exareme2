from typing import List
from mipengine.node.monetdb_interface import common
from mipengine.node.monetdb_interface.common import convert_table_info_to_sql_query_format, cursor, connection
from mipengine.node.tasks.data_classes import TableInfo, ColumnInfo


def get_remote_tables_names(context_id: str) -> List[str]:
    return common.get_tables_names("remote", context_id)


def create_remote_table(table_info: TableInfo, url: str):
    columns_schema = convert_table_info_to_sql_query_format(table_info)
    cursor.execute(f"CREATE REMOTE TABLE {table_info.name} ( {columns_schema}) on '{url}'")
    connection.commit()