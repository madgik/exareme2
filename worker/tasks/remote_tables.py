from typing import Tuple, List
from celery import shared_task

from tasks.tables import TableInfo, TableData


@shared_task
def get_remote_tables() -> List[TableInfo]:
    pass


@shared_task
def create_remote_table(table_name: str):
    pass


@shared_task
def get_remote_table(remote_table_name: str) -> TableData:
    pass