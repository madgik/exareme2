from typing import List

import pymonetdb
from celery import shared_task

from tasks import monetdb_interface
from tasks.data_classes import TableInfo, TableData, ColumnInfo


@shared_task
def get_tables_info(table_name: List[str] = None) -> List[TableInfo]:
    list_of_tables = monetdb_interface.get_tables_info(table_name)
    return TableInfo.schema().dumps(list_of_tables, many=True)


@shared_task
def create_table(columns_info: List[ColumnInfo], execution_id: str) -> TableInfo:
    # TODO , a table name cannot start with number, what do we put in front?
    table_name = 'table_' + str(pymonetdb.uuid.uuid1()).replace("-", "") + "_" + execution_id
    table_info = TableInfo(table_name, columns_info)

    monetdb_interface.create_table(table_info)

    return table_info.to_json()


@shared_task
def get_table_data(table_name: str) -> TableData:
    schema = monetdb_interface.get_table_schema(table_name)
    data = monetdb_interface.get_table_data(table_name)
    return TableData(data, schema).to_json()


@shared_task
def delete_table(table_name: str):
    monetdb_interface.delete_table(table_name)
