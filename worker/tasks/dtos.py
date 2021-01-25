from enum import Enum
from typing import List, Union


class ColumnType(Enum):
    INT = 'INT'
    FLOAT = 'FLOAT'
    TEXT = 'TEXT'


class TableSchema:
    def __init__(self, column_name, column_type):
        self.column_name: str = column_name
        self.type: ColumnType = column_type


class TableInfo:
    def __init__(self, table_name, schema):
        self.table_name: str = table_name
        self.schema: TableSchema = schema


class ViewTable:
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
        self.schema: TableSchema = schema

