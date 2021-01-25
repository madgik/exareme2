from typing import Tuple, List

from celery import shared_task

from tasks.tables import TableInfo


@shared_task
def get_udfs() -> Tuple[str, str]:
    pass


@shared_task
def create_udf(udf_name: str, input: List[str]) -> TableInfo:
    pass


@shared_task
def get_udf(udf_name: str) -> Tuple[str, str]:
    pass
