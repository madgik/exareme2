from dataclasses import dataclass
from typing import List, Union

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ColumnInfo:
    name: str
    type: str

    def __init__(self, name, type):
        self.name = name
        print(type)
        try:
            self.type = {
                "int": "int",
                "varchar": "text",
                "double": "float"
            }[type]
        except Exception:
            raise ValueError("Column can have one of the following types: INT,FLOAT,TEXT")


@dataclass_json
@dataclass
class TableInfo:
    name: str
    schema: List[ColumnInfo]


@dataclass_json
@dataclass
class TableView:
    datasets: List[str]
    columns: List[str]
    filters: str


@dataclass_json
@dataclass
class TableData:
    data: List[
        List[
            Union[
                str,
                int,
                float,
                bool]]]
    schema: List[ColumnInfo]


@dataclass_json
@dataclass
class Parameter:
    name: str
    value: 'typing.Any'


@dataclass_json
@dataclass
class UDFInfo:
    name: str
    header: str
