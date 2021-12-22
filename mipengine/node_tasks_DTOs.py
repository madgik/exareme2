import enum
from abc import ABC
from typing import Any
from typing import List
from typing import Union

from pydantic import (
    BaseModel,
    validator,
)

from mipengine import DType

# ~~~~~~~~~~~~~~~~~~~~ Enums ~~~~~~~~~~~~~~~~~~~~ #


class UDFArgumentKind(enum.Enum):
    TABLE = enum.auto()
    LITERAL = enum.auto()

    def __str__(self):
        return self.name


class TableType(enum.Enum):
    NORMAL = enum.auto()
    REMOTE = enum.auto()
    MERGE = enum.auto()
    VIEW = enum.auto()

    def __str__(self):
        return self.name


# ~~~~~~~~~~~~~~~~~~ Validator ~~~~~~~~~~~~~~~~~ #


def validate_identifier(identifier):
    if not identifier.isidentifier():
        raise ValueError(f"Expected valid identifier, got {identifier}")
    return identifier


# ~~~~~~~~~~~~~~~~~~~ DTOs ~~~~~~~~~~~~~~~~~~~~~~ #


class ImmutableBaseModel(BaseModel, ABC):
    class Config:
        allow_mutation = False


class ColumnInfo(ImmutableBaseModel):
    name: str
    dtype: DType

    _validate_identifier = validator("name", allow_reuse=True)(validate_identifier)


class TableSchema(ImmutableBaseModel):
    columns: List[ColumnInfo]


class TableInfo(ImmutableBaseModel):
    name: str
    schema_: TableSchema
    type_: TableType

    _validate_identifier = validator("name", allow_reuse=True)(validate_identifier)


class _ColumnData(ImmutableBaseModel):
    name: str
    type: DType

    @validator("type")
    def validate_type(cls, tp):
        if cls.__name__ == "ColumnData":
            raise TypeError(
                "ColumnData should not be instantiated. "
                "Use ColumnDataInt, ColumnDataStr or ColumnDataFloat instead."
            )
        column_type = cls.__fields__["type"].default
        if tp != column_type:
            raise ValueError(
                f"Objects of type {cls.__name__} have a fixed type {column_type}, "
                f"you cannot use {tp} in the constructor."
            )
        return tp


class ColumnDataInt(_ColumnData):
    data: List[Union[None, int]]
    type = DType.INT


class ColumnDataStr(_ColumnData):
    data: List[Union[None, str]]
    type = DType.STR


class ColumnDataFloat(_ColumnData):
    data: List[Union[None, float]]
    type = DType.FLOAT


class TableData(ImmutableBaseModel):
    name: str
    columns: List[Union[ColumnDataInt, ColumnDataStr, ColumnDataFloat]]


class UDFArgument(ImmutableBaseModel):
    kind: UDFArgumentKind
    value: Any


# ~~~~~~~~~~~~~~~~~~~ Exceptions ~~~~~~~~~~~~~~~~~~~~~~ #


class InsufficientDataError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
