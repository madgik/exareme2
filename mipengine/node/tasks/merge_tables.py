from typing import List

from celery import shared_task
from tasks.tables import TableData
from tasks.tables import TableInfo


@shared_task
def get_merge_tables() -> List[TableInfo]:
    pass


@shared_task
def create_merge_table(schema: str):
    pass


@shared_task
def get_merge_table(merge_table_name: str) -> TableData:
    pass


@shared_task
def update_merge_table(merge_table_name: str):
    pass

