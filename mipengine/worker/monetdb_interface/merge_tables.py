from typing import List

from mipengine.worker.monetdb_interface.common import convert_table_info_to_sql_query_format, add_tables_to_merge_table, cursor, connection
from mipengine.worker.tasks.data_classes import TableInfo, ColumnInfo
from mipengine.worker_tests.tables import get_tables_info, delete_table


def get_merge_table_schema(table_name: str = None) -> List[ColumnInfo]:
    return get_merge_tables_info([table_name])[0].schema


def get_merge_tables_info(table_names: List[str] = None) -> List[TableInfo]:
    return get_tables_info("merge", table_names)


def create_merge_table(table_info: TableInfo, partition_tables_names: List[str]):
    columns_schema = convert_table_info_to_sql_query_format(table_info)
    print(f"CREATE MERGE TABLE ({table_info.name} {columns_schema} );")
    cursor.execute(f"CREATE MERGE TABLE {table_info.name} ( {columns_schema} );")
    connection.commit()
    add_tables_to_merge_table(table_info.name, partition_tables_names)


def delete_merge_table(table_name: str):
    delete_table(table_name, "merge")
