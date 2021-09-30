import enum
from typing import Any
from typing import Dict
from typing import List
from typing import Union
from warnings import warn

from pydantic import (
    BaseModel,
    validator,
)


# ~~~~~~~~~~~~~~~~~~~~ Enums ~~~~~~~~~~~~~~~~~~~~ #


class DBDataType(enum.Enum):
    INT = enum.auto()
    FLOAT = enum.auto()
    TEXT = enum.auto()


class UDFArgumentKind(enum.Enum):
    TABLE = enum.auto()
    LITERAL = enum.auto()


# ~~~~~~~~~~~~~~~~~~ Validators ~~~~~~~~~~~~~~~~~ #


def validate_allowed_types(type: str, allowed_types: List[str]):
    if type.lower() not in allowed_types:
        raise TypeError(f"The allowed types are the following : {allowed_types}")
    return type.lower()


def validate_name(name):
    if not name.isidentifier():
        raise ValueError(f"Expected valid identifier, got {name}")
    if not name.islower():
        warn(f"Names must be lowercase, got {name}")
        return name.lower()
    return name


# ~~~~~~~~~~~~~~~~~~~ DTOs ~~~~~~~~~~~~~~~~~~~~~~ #


class ColumnInfo(BaseModel):
    name: str
    data_type: DBDataType

    _validate_name = validator("name", allow_reuse=True)(validate_name)


class TableSchema(BaseModel):
    columns: List[ColumnInfo]


class TableInfo(BaseModel):
    name: str
    table_schema: TableSchema

    _validate_name = validator("name", allow_reuse=True)(validate_name)


class TableView(BaseModel):
    datasets: List[str]
    columns: List[str]
    filter: Dict

    _validate_names = validator(
        "datasets",
        "columns",
        each_item=True,
        allow_reuse=True,
    )(validate_name)


class TableData(BaseModel):
    table_schema: TableSchema
    data: List[List[Union[float, int, bool, str]]]


class UDFArgument(BaseModel):
    kind: UDFArgumentKind
    value: Any
