import enum
from abc import ABC
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from pydantic import BaseModel
from pydantic import validator

from mipengine import DType
from mipengine.table_data_DTOs import ColumnDataBinary
from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataInt
from mipengine.table_data_DTOs import ColumnDataJSON
from mipengine.table_data_DTOs import ColumnDataStr


# ~~~~~~~~~~~~~~~~~~~~ Enums ~~~~~~~~~~~~~~~~~~~~ #
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


# ~~~~~~~~~~~~~~~~~~~ DTOs ~~~~~~~~~~~~~~~~~~~~~~ #


class ImmutableBaseModel(BaseModel, ABC):
    class Config:
        allow_mutation = False


class ColumnInfo(ImmutableBaseModel):
    name: str
    dtype: DType


class TableSchema(ImmutableBaseModel):
    columns: List[ColumnInfo]

    @property
    def column_names(self):
        return [column_info.name for column_info in self.columns]


class TableInfo(ImmutableBaseModel):
    name: str
    schema_: TableSchema
    type_: TableType

    @property
    def column_names(self):
        return self.schema_.column_names

    @property
    def _tablename_parts(self) -> Tuple[str, str, str, str]:
        table_type, node_id, context_id, command_id, command_subid = self.name.split(
            "_"
        )
        return node_id, context_id, command_id, command_subid

    @property
    def node_id(self) -> str:
        node_id, _, _, _ = self._tablename_parts
        return node_id

    @property
    def context_id(self) -> str:
        _, context_id, _, _ = self._tablename_parts
        return context_id

    @property
    def command_id(self) -> str:
        _, _, command_id, _ = self._tablename_parts
        return command_id

    @property
    def command_subid(self) -> str:
        _, _, _, command_subid = self._tablename_parts
        return command_subid

    @property
    def name_without_node_id(self) -> str:
        return (
            str(self.type_)
            + "_"
            + self.context_id
            + "_"
            + self.command_id
            + "_"
            + self.command_subid
        )


class DataModelAttributes(ImmutableBaseModel):
    tags: List
    properties: Dict


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
        return (
            isinstance(other, CommonDataElement)
            and self.code == other.code
            and self.label == other.label
            and self.sql_type == other.sql_type
            and self.is_categorical == other.is_categorical
            and self.enumerations == other.enumerations
            and self.max == other.max
            and self.min == other.min
        )


class CommonDataElements(BaseModel):
    values: Dict[str, CommonDataElement]

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    def __eq__(self, other):
        """
        We are overriding the equals function to check that the two cdes have identical fields except one edge case.
        The edge case is that the two comparing cdes can only contain a difference in the field of enumerations in
        the cde with code 'dataset' and still be considered compatible.
        """
        if set(self.values.keys()) != set(other.values.keys()):
            return False
        for cde_code in self.values.keys():
            cde1 = self.values[cde_code]
            cde2 = other.values[cde_code]
            if not cde1 == cde2 and not self._are_equal_dataset_cdes(cde1, cde2):
                return False
        return True

    def _are_equal_dataset_cdes(
        self, cde1: CommonDataElement, cde2: CommonDataElement
    ) -> bool:
        if cde1.code != "dataset" or cde2.code != "dataset":
            return False

        if (
            cde1.label != cde2.label
            or cde1.sql_type != cde2.sql_type
            or cde1.is_categorical != cde2.is_categorical
            or cde1.max != cde2.max
            or cde1.min != cde2.min
        ):
            return False

        return True


# ~~~~~~~~~~~~~~~~~~~ UDFs IO ~~~~~~~~~~~~~~~~~~~~~~ #


class SMPCTablesInfo(ImmutableBaseModel):
    template: TableInfo
    sum_op: Optional[TableInfo]
    min_op: Optional[TableInfo]
    max_op: Optional[TableInfo]


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
    value: TableInfo


class NodeSMPCDTO(NodeUDFDTO):
    type = _NodeUDFDTOType.SMPC
    value: SMPCTablesInfo


class NodeUDFPosArguments(ImmutableBaseModel):
    # The NodeSMPCDTO cannot be used here instead of the Union due to pydantic json deserialization.
    args: List[Union[NodeLiteralDTO, NodeTableDTO, NodeSMPCDTO]]


class NodeUDFKeyArguments(ImmutableBaseModel):
    # The NodeSMPCDTO cannot be used here instead of the Union due to pydantic json deserialization.
    args: Dict[str, Union[NodeLiteralDTO, NodeTableDTO, NodeSMPCDTO]]


class NodeUDFResults(ImmutableBaseModel):
    # The NodeSMPCDTO cannot be used here instead of the Union due to pydantic json deserialization.
    results: List[Union[NodeLiteralDTO, NodeTableDTO, NodeSMPCDTO]]
