from typing import List

from celery import shared_task

from tasks.dtos import TableInfo, TableSchema


@shared_task
def get_tables() -> List[TableInfo]:
    pass


@shared_task
def create_table(table_schemas: List[TableSchema]) -> TableInfo:
    pass


@shared_task
def get_local_non_private_tables() -> List[TableInfo]:
    pass


@shared_task
def get_table(table_name: str) -> TableInfo:
    pass


@shared_task
def delete_table(table_name: str):
    pass
