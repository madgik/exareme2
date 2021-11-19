import enum
from abc import ABC
from typing import Any
from typing import Dict
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


class TableView(ImmutableBaseModel):
    datasets: List[str]
    columns: List[str]
    filter: Dict

    _validate_identifiers = validator(
        "datasets",
        "columns",
        each_item=True,
        allow_reuse=True,
    )(validate_identifier)


class TableData(ImmutableBaseModel):
    schema_: TableSchema
    data_: List[List[Union[float, int, str, None]]]
    # Union is problematic in pydantic we keep track on that with bug report
    # https://team-1617704806227.atlassian.net/browse/MIP-245


class UDFArgument(ImmutableBaseModel):
    kind: UDFArgumentKind
    value: Any


# ~~~~~~~~~~~~~~~~~~~ Exceptions ~~~~~~~~~~~~~~~~~~~~~~ #


class InsufficientDataError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message
