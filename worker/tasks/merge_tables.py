from typing import List

from celery import shared_task

from worker.tasks.data_classes import TableInfo, TableData


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
