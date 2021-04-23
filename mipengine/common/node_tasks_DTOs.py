from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ColumnInfo:
    name: str
    data_type: str


@dataclass_json
@dataclass
class TableSchema:
    columns: List[ColumnInfo]


@dataclass_json
@dataclass
class TableInfo:
    name: str
    schema: TableSchema


@dataclass_json
@dataclass
class TableView:
    datasets: List[str]
    columns: List[str]
    filter: Dict


@dataclass_json
@dataclass
class TableData:
    schema: TableSchema
    data: List[List[Union[str, int, float, bool]]]


@dataclass_json
@dataclass
class UDFArgument:
    type: str
    value: Any
