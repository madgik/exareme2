import json
from typing import List

from celery import shared_task

from mipengine.node.monetdb_interface import common
from mipengine.node.monetdb_interface import remote_tables
from mipengine.node.tasks.data_classes import TableInfo


@shared_task
def get_remote_tables(context_id: str) -> List[str]:
    return json.dumps(remote_tables.get_remote_tables_names(context_id))


@shared_task
def create_remote_table(table_info_json: str, url: str):

    table_info = TableInfo.from_json(table_info_json)
    remote_tables.create_remote_table(table_info, url)
    return 0


@shared_task
def clean_up(context_Id: str = None):
    common.clean_up(context_Id)
    return 0
