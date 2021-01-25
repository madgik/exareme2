from typing import Tuple, List

from celery import shared_task

from tasks.dtos import ViewTable, TableData, TableInfo


@shared_task
def get_views() -> List[TableInfo]:
    pass


@shared_task
def create_view(view: ViewTable) -> List[TableInfo]:
    pass


@shared_task
def get_view(view_name: str) -> Tuple[TableInfo, TableData]:
    pass


@shared_task
def delete_view(view_name: str):
    pass
