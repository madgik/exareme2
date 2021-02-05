from typing import List

from mipengine.worker.monetdb_interface import common
from mipengine.worker.monetdb_interface.common import convert_table_info_to_sql_query_format, cursor, connection
from mipengine.worker.tasks.data_classes import TableInfo, ColumnInfo


def get_table_schema(table_name: str = None) -> List[ColumnInfo]:
    return get_tables_info([table_name])[0].schema


def get_tables_info(table_names: List[str] = None) -> List[TableInfo]:
    return common.get_tables_info("normal", table_names)


def create_table(table_info: TableInfo):
    columns_schema = convert_table_info_to_sql_query_format(table_info)

    cursor.execute(f"CREATE TABLE {table_info.name} ( {columns_schema} )")
    connection.commit()


def delete_table(table_name: str):
    common.delete_table(table_name, "normal")
