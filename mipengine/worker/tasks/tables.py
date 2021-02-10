import json
from typing import List
from celery import shared_task

from mipengine.worker.monetdb_interface import tables, common
from mipengine.worker.monetdb_interface.common import tables_naming_convention, config
from mipengine.worker.tasks.data_classes import TableInfo, TableData, ColumnInfo


@shared_task
def get_tables(context_id: str) -> List[str]:
    return json.dumps(tables.get_tables_names(context_id))


@shared_task
def get_table_schema(table_name: str) -> List[ColumnInfo]:
    schema = tables.get_table_schema(table_name)
    return ColumnInfo.schema().dumps(schema, many=True)


@shared_task
def get_table_data(table_name: str) -> TableData:
    schema = tables.get_table_schema(table_name)
    data = tables.get_table_data(table_name)
    return TableData(schema, data).to_json()


@shared_task
def create_table(context_Id: str, schema: List[ColumnInfo]) -> str:
    schema_object = ColumnInfo.schema().load(schema, many=True)
    table_name = tables_naming_convention("table", context_Id, config["node"]["identifier"])
    table_info = TableInfo(table_name.lower(), schema_object)
    tables.create_table(table_info)
    return table_name.lower()


@shared_task
def clean_up(context_Id: str = None):
    common.clean_up(context_Id)
