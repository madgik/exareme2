import enum
from abc import ABC
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import validator

from mipengine import DType

# ~~~~~~~~~~~~~~~~~~~~ Enums ~~~~~~~~~~~~~~~~~~~~ #
from mipengine.table_data_DTOs import ColumnDataBinary
from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataInt
from mipengine.table_data_DTOs import ColumnDataJSON
from mipengine.table_data_DTOs import ColumnDataStr


class _NodeUDFDTOType(enum.Enum):
    TABLE = "TABLE"
    LITERAL = "LITERAL"
    SMPC = "SMPC"

    def __str__(self):
        return self.name


class TableType(enum.Enum):
    NORMAL = "NORMAL"
    REMOTE = "REMOTE"
    MERGE = "MERGE"
    VIEW = "VIEW"

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


class CommonDataElement(ImmutableBaseModel):
    code: str
    label: str
    sql_type: str
    is_categorical: bool
    enumerations: Optional[Dict[str, str]] = None
    min: Optional[float] = None
    max: Optional[float] = None

    def __eq__(self, other):
        if isinstance(other, CommonDataElement):
            return (
                self.code == other.code
                and self.label == other.label
                and self.sql_type == other.sql_type
                and self.is_categorical == other.is_categorical
                and self.enumerations == other.enumerations
                and self.max == other.max
                and self.min == other.min
            )
        return False


# ~~~~~~~~~~~~~~~~~~~ UDFs IO ~~~~~~~~~~~~~~~~~~~~~~ #


class NodeUDFDTO(ImmutableBaseModel):
    type: _NodeUDFDTOType
    value: Any

    @validator("type")
    def validate_type(cls, tp):
        if cls.__name__ == "NodeUDFDTO":
            raise TypeError(
                "NodeUDFDTO should not be instantiated. "
                "Use NodeLiteralDTO, NodeTableDTO or NodeSMPCDTO instead."
            )
        udf_argument_type = cls.__fields__["type"].default
        if tp != udf_argument_type:
            raise ValueError(
                f"Objects of type {cls.__name__} have a fixed type {udf_argument_type}, "
                f"you cannot use {tp} in the constructor."
            )
        return tp


class NodeLiteralDTO(NodeUDFDTO):
    type = _NodeUDFDTOType.LITERAL
    value: Any


class NodeTableDTO(NodeUDFDTO):
    type = _NodeUDFDTOType.TABLE
    value: str


class NodeSMPCValueDTO(ImmutableBaseModel):
    template: NodeTableDTO
    sum_op_values: NodeTableDTO = None
    min_op_values: NodeTableDTO = None
    max_op_values: NodeTableDTO = None
    union_op_values: NodeTableDTO = None


class NodeSMPCDTO(NodeUDFDTO):
    type = _NodeUDFDTOType.SMPC
    value: NodeSMPCValueDTO


class UDFPosArguments(ImmutableBaseModel):
    args: List[Union[NodeLiteralDTO, NodeTableDTO, NodeSMPCDTO]]


class UDFKeyArguments(ImmutableBaseModel):
    args: Dict[str, Union[NodeLiteralDTO, NodeTableDTO, NodeSMPCDTO]]


class UDFResults(ImmutableBaseModel):
    results: List[Union[NodeTableDTO, NodeSMPCDTO]]
