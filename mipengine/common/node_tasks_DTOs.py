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

    def __post_init__(self):
        allowed_types = {"int", "text", "real"}
        self.data_type = str.lower(self.data_type)
        if self.data_type not in allowed_types:
            raise TypeError(
                f"Column can have one of the following types: {allowed_types}"
            )

        if not self.name.isidentifier():
            raise ValueError(
                f"Name : {self.name} has inappropriate characters for a sql query."
            )


@dataclass_json
@dataclass
class TableSchema:
    columns: List[ColumnInfo]


@dataclass_json
@dataclass
class TableInfo:
    name: str
    schema: TableSchema

    def __post_init__(self):
        if not self.name.isidentifier():
            raise ValueError(
                f"Name : {self.name} has inappropriate characters for a sql query."
            )


@dataclass_json
@dataclass
class TableView:
    datasets: List[str]
    columns: List[str]
    filter: Dict

    def __post_init__(self):
        for dataset in self.datasets:
            if not self.dataset.isidentifier():
                raise ValueError(
                    f"Dataset : {dataset} has inappropriate characters for a sql query."
                )
        for column in self.columns:
            if not self.column.isidentifier():
                raise ValueError(
                    f"Column : {column} has inappropriate characters for a sql query."
                )


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

    def __post_init__(self):
        allowed_types = {"table", "literal"}
        if self.type not in allowed_types:
            raise TypeError(
                f"UDFArgument type  can have one of the following types: {allowed_types}"
            )
