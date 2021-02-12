from typing import List

from celery import shared_task
from tasks.tables import TableInfo


class Parameter:
    def __init__(self, name, value):
        self.name: str = name
        self.value = value


class UDFInfo:
    def __init__(self, name, header):
        self.name: str = name
        self.header: str = header


@shared_task
def get_udfs() -> List[UDFInfo]:
    pass


@shared_task
def run_udf(udf_name: str, input: List[Parameter]) -> TableInfo:
    pass


@shared_task
def get_udf(udf_name: str) -> UDFInfo:
    pass
