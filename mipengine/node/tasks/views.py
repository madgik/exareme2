from typing import Tuple, List

from celery import shared_task

from tasks.tables import TableView, TableData, TableInfo


@shared_task
def get_views() -> List[TableInfo]:
    pass


@shared_task
def create_view(view: TableView) -> TableInfo:
    pass


@shared_task
def get_view(view_name: str) -> TableData:
    pass


@shared_task
def delete_view(view_name: str):
    pass
