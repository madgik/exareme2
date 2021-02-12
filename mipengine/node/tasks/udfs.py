from typing import List

from celery import shared_task

from mipengine.node.tasks.data_classes import Parameter
from mipengine.node.tasks.data_classes import TableInfo
from mipengine.node.tasks.data_classes import UDFInfo


@shared_task
def get_udfs() -> List[UDFInfo]:
    pass


@shared_task
def run_udf(udf_name: str, input: List[Parameter]) -> TableInfo:
    pass


@shared_task
def get_udf(udf_name: str) -> UDFInfo:
    pass
