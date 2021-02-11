import json
from typing import List
from celery import shared_task

from mipengine.node.monetdb_interface import merge_tables, common, tables
from mipengine.node.monetdb_interface.common import tables_naming_convention, config, connection
from mipengine.node.tasks.data_classes import TableInfo, ColumnInfo


@shared_task
def get_merge_tables(context_id: str) -> List[str]:
    return json.dumps(merge_tables.get_merge_tables_names(context_id))


@shared_task
def create_merge_table(context_Id: str, partition_tables_names: List[str]) -> str:
    merge_table_name = tables_naming_convention("merge", context_Id, config["node"]["identifier"])
    schema = tables.get_table_schema(partition_tables_names[0])
    table_info = TableInfo(merge_table_name.lower(), schema)
    merge_tables.create_merge_table(table_info)
    merge_tables.add_to_merge_table(merge_table_name, partition_tables_names)
    return merge_table_name.lower()


@shared_task
def clean_up(context_Id: str = None):
    common.clean_up(context_Id)
