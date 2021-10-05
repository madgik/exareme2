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

from mipengine import DType

# ~~~~~~~~~~~~~~~~~~~~ Enums ~~~~~~~~~~~~~~~~~~~~ #


class UDFArgumentKind(enum.Enum):
    TABLE = enum.auto()
    LITERAL = enum.auto()


# ~~~~~~~~~~~~~~~~~~ Validator ~~~~~~~~~~~~~~~~~ #


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
    dtype: DType

    _validate_name = validator("name", allow_reuse=True)(validate_name)


class TableSchema(BaseModel):
    columns: List[ColumnInfo]


class TableInfo(BaseModel):
    name: str
    schema_: TableSchema

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
    schema_: TableSchema
    data_: List[List[Union[float, int, str, None]]]
    # Union is problematic in pydantic we keep track on that with bug report
    # https://team-1617704806227.atlassian.net/browse/MIP-245


class UDFArgument(BaseModel):
    kind: UDFArgumentKind
    value: Any
