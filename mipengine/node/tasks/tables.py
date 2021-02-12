from enum import Enum
from typing import List
from typing import Union

from celery import shared_task


class ColumnType(Enum):
    INT = 'INT'
    FLOAT = 'FLOAT'
    TEXT = 'TEXT'


class ColumnInfo:
    def __init__(self, column_name, column_type):
        self.column_name: str = column_name
        self.column_type: ColumnType = column_type


class TableInfo:
    def __init__(self, name, schema):
        self.name: str = name
        self.schema: List[ColumnInfo] = schema


class TableView:
    def __init__(self, datasets, columns, filters):
        self.datasets: List[str] = datasets
        self.columns: List[str] = columns
        self.filters = filters


class TableData:
    def __init__(self, data, schema):
        self.data: List[
            List[
                Union[
                    str,
                    int,
                    float,
                    bool]]] = data
        self.schema: List[ColumnInfo] = schema


@shared_task
def get_tables() -> List[TableInfo]:
    pass


@shared_task
def create_table(column_infos: List[ColumnInfo], execution_id: str) -> TableInfo:
    pass


@shared_task
def get_table(table_name: str) -> TableData:
    pass


@shared_task
def delete_table(table_name: str):
    pass
