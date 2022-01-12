import enum
from abc import ABC
from typing import Any
from typing import List
from typing import Dict
from typing import Union

from pydantic import (
    BaseModel,
    validator,
)

from mipengine import DType

# ~~~~~~~~~~~~~~~~~~~~ Enums ~~~~~~~~~~~~~~~~~~~~ #
from mipengine.table_data_DTOs import ColumnDataBinary
from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataInt
from mipengine.table_data_DTOs import ColumnDataJSON
from mipengine.table_data_DTOs import ColumnDataStr


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


class TableData(ImmutableBaseModel):
    name: str
    columns: List[
        Union[
            ColumnDataInt,
            ColumnDataStr,
            ColumnDataFloat,
            ColumnDataJSON,
            ColumnDataBinary,
        ]
    ]


class UDFArgument(ImmutableBaseModel):
    kind: UDFArgumentKind
    value: Any


# ~~~~~~~~~~~~~~~~~~~ Exceptions ~~~~~~~~~~~~~~~~~~~~~~ #


class InsufficientDataError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
