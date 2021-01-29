from typing import List

from celery import shared_task

from tasks.data_classes import UDFInfo, Parameter
from tasks.tables import TableInfo


@shared_task
def get_udfs() -> List[UDFInfo]:
    pass


@shared_task
def run_udf(udf_name: str, input: List[Parameter]) -> TableInfo:
    pass


@shared_task
def get_udf(udf_name: str) -> UDFInfo:
    pass
