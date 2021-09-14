from typing import Any
from typing import Dict
from typing import List
from typing import Union
from warnings import warn

from pydantic import (
    BaseModel,
    validator,
)


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


class ColumnInfo(BaseModel):
    name: str
    data_type: str

    _validate_allowed_types = validator(
        "data_type", {"int", "text", "real"}, allow_reuse=True
    )(validate_allowed_types)
    _validate_name = validator("name", allow_reuse=True)(validate_name)


class TableSchema(BaseModel):
    columns: List[ColumnInfo]


class TableInfo(BaseModel):
    name: str
    schema: TableSchema

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
    schema: TableSchema
    data: List[List[Union[str, int, float, bool]]]


class UDFArgument(BaseModel):
    type: str
    value: Any

    _validate_allowed_types = validator("type", {"table", "literal"}, allow_reuse=True)(
        validate_allowed_types
    )
