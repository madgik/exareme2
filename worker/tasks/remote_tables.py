from typing import List

from celery import shared_task

from worker.tasks.data_classes import TableInfo, TableData


@shared_task
def get_remote_tables() -> List[TableInfo]:
    pass


@shared_task
def create_remote_table(table_name: str):
    pass


@shared_task
def get_remote_table(remote_table_name: str) -> TableData:
    pass
