from typing import List

from celery import shared_task

from mipengine.worker import monetdb_interface
from mipengine.worker.tasks.data_classes import TableInfo, TableData, ColumnInfo


@shared_task
def get_remote_tables_info(table_name: List[str] = None) -> List[TableInfo]:
    list_of_tables = monetdb_interface.get_remote_tables_info(table_name)
    return TableInfo.schema().dumps(list_of_tables, many=True)


@shared_task
def create_remote_table(columns_info: List[ColumnInfo], name: str, url: str) -> TableInfo:
    table_info = TableInfo(name, columns_info)
    monetdb_interface.create_remote_table(table_info, url)
    return table_info.to_json()


@shared_task
def get_remote_table_data(table_name: str) -> TableData:
    schema = monetdb_interface.get_remote_table_schema(table_name)
    data = monetdb_interface.get_table_data(table_name)
    return TableData(data, schema).to_json()


@shared_task
def delete_remote_table(table_name: str):
    monetdb_interface.delete_remote_table(table_name)
