from dataclasses import dataclass
from typing import List
from typing import Union

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ColumnInfo:
    name: str
    type: str

    def __init__(self, name, type):
        self.name = name
        allowed_types = {"int", "text", "float", "bool", "clob"}
        if str.lower(type) in allowed_types:
            self.type = str.lower(type)
        else:
            raise ValueError(f"Column can have one of the following types: {allowed_types}")


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
    filter: str


@dataclass_json
@dataclass
class TableData:
    schema: List[ColumnInfo]
    data: List[
        List[
            Union[
                str,
                int,
                float,
                bool]]]


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
