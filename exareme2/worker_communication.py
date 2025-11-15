from abc import ABC
from enum import Enum
from enum import unique
from ipaddress import IPv4Address
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
from pydantic import BaseModel
from pydantic import Field
from pydantic import validator

from exareme2 import DType

"""
!!!!!!!!!!!!!!!!!!!!!! ATTENTION !!!!!!!!!!!!!!!!!!!!!!!!
In some cases an exception thrown by the WORKER(celery) will be received
in the CONTROLLER(celery get method) as a generic Exception and catching
it by its definition won't be possible.

This is happening due to a celery problem: https://github.com/celery/celery/issues/3586

There are some workarounds possible that are case specific.

For example in the DataModelUnavailable using the `super().__init__(self.message)`
was creating many problems in deserializing the exception.

When adding a new exception, the task throwing it should be tested:
1) That you can catch the exception by its name,
2) the contained message, if exists, is shown properly.
"""


class TablesNotFound(Exception):
    """
    Exception raised for errors while retrieving a table from a database.

    Attributes:
        tables -- tables which caused the error
        message -- explanation of the error
    """

    def __init__(self, tables: List[str]):
        self.tables = tables
        self.message = f"The following tables were not found : {tables}"
        super().__init__(self.message)


class IncompatibleSchemasMergeException(Exception):
    """Exception raised for errors while trying to merge tables with incompatible schemas.

    Attributes:
        table -- table which caused the error
        message -- explanation of the error
    """

    def __init__(self, table_names: List[str]):
        self.table_names = table_names
        self.message = (
            f"Tables to be added don't match MERGE TABLE schema : {table_names}"
        )
        super().__init__(self.message)


class IncompatibleTableTypes(Exception):
    """Exception raised for errors while trying to merge tables with incompatible table types.

    Attributes:
        table_types --  the types of the table which caused the error
        message -- explanation of the error
    """

    def __init__(self, table_types: set):
        self.table_types = table_types
        self.message = f"Tables have more than one distinct types : {self.table_types}"
        super().__init__(self.message)


class RequestIDNotFound(Exception):
    """Exception raised while checking the presence of request_id in task's arguments.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self):
        self.message = "Request id is missing from task's arguments."
        super().__init__(self.message)


class DataModelUnavailable(Exception):
    """
    Exception raised when a data model is not available in the WORKER db.

    Attributes:
        worker_id -- the worker id that threw the exception
        data_model --  the unavailable data model
        message -- explanation of the error
    """

    def __init__(self, worker_id: str, data_model: str):
        self.worker_id = worker_id
        self.data_model = data_model
        self.message = f"Data model '{self.data_model}' is not available in worker: '{self.worker_id}'."


class DatasetUnavailable(Exception):
    """
    Exception raised when a dataset is not available in the WORKER db.

    Attributes:
        worker_id -- the worker id that threw the exception
        dataset --  the unavailable dataset
        message -- explanation of the error
    """

    def __init__(self, worker_id: str, dataset: str):
        self.worker_id = worker_id
        self.dataset = dataset
        self.message = (
            f"Dataset '{self.dataset}' is not available in worker: '{self.worker_id}'."
        )


class InsufficientDataError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class BadUserInput(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


# ~~~~~~~~~~~~~~~~~~~~ Enums ~~~~~~~~~~~~~~~~~~~~ #


class _WorkerUDFDTOType(Enum):
    TABLE = "TABLE"
    LITERAL = "LITERAL"
    SMPC = "SMPC"

    def __str__(self):
        return self.name


class TableType(Enum):
    NORMAL = "NORMAL"
    REMOTE = "REMOTE"
    MERGE = "MERGE"
    VIEW = "VIEW"

    def __str__(self):
        return self.name


@unique
class WorkerRole(str, Enum):
    GLOBALWORKER = "GLOBALWORKER"
    LOCALWORKER = "LOCALWORKER"


# ~~~~~~~~~~~~~~~~~~~ DTOs ~~~~~~~~~~~~~~~~~~~~~~ #


class ImmutableBaseModel(BaseModel, ABC):
    class Config:
        allow_mutation = False


class WorkerInfo(BaseModel):
    id: str
    role: WorkerRole
    ip: IPv4Address
    port: int
    data_folder: Optional[str] = None
    auto_load_data: bool = False

    @property
    def socket_addr(self) -> str:
        return f"{self.ip}:{self.port}"


class ColumnInfo(ImmutableBaseModel):
    name: str
    dtype: DType


class TableSchema(ImmutableBaseModel):
    columns: List[ColumnInfo]

    @property
    def column_names(self):
        return [column_info.name for column_info in self.columns]

    @classmethod
    def from_list(cls, lst: List[Tuple[str, DType]]):
        return cls(columns=[ColumnInfo(name=name, dtype=dtype) for name, dtype in lst])

    def to_list(self) -> List[Tuple[str, DType]]:
        return [(col.name, col.dtype) for col in self.columns]


class TableInfo(ImmutableBaseModel):
    name: str
    schema_: TableSchema
    type_: TableType

    @property
    def column_names(self):
        return self.schema_.column_names

    @property
    def _tablename_parts(self) -> Tuple[str, str, str, str]:
        table_type, worker_id, context_id, command_id, result_id = self.name.split("_")
        return worker_id, context_id, command_id, result_id

    @property
    def worker_id(self) -> str:
        worker_id, _, _, _ = self._tablename_parts
        return worker_id

    @property
    def context_id(self) -> str:
        _, context_id, _, _ = self._tablename_parts
        return context_id

    @property
    def command_id(self) -> str:
        _, _, command_id, _ = self._tablename_parts
        return command_id

    @property
    def result_id(self) -> str:
        _, _, _, result_id = self._tablename_parts
        return result_id

    @property
    def name_without_worker_id(self) -> str:
        return (
            str(self.type_)
            + "_"
            + self.context_id
            + "_"
            + self.command_id
            + "_"
            + self.result_id
        )


class DatasetProperties(ImmutableBaseModel):
    tags: List
    properties: Dict[str, List[str]]

    @validator("properties")
    def validate_variables(cls, properties: Dict[str, List[str]]):
        variables = properties.get("variables")
        if not isinstance(variables, list):
            raise ValueError("Dataset properties must include a 'variables' list.")
        if not all(isinstance(variable, str) for variable in variables):
            raise ValueError("Dataset variables must be strings.")
        return properties

    @property
    def variables(self) -> List[str]:
        return self.properties["variables"]


class DataModelAttributes(ImmutableBaseModel):
    tags: List
    properties: Dict


class DatasetInfo(ImmutableBaseModel):
    code: str
    label: str
    variables: List[str] = Field(default_factory=list)


class DatasetsInfoPerDataModel(ImmutableBaseModel):
    datasets_info_per_data_model: Dict[str, List[DatasetInfo]]


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


# ~~~~~~~~~~~~~~~~~~~ Table Data DTOs ~~~~~~~~~~~~~~~~~~~~~~ #


class ColumnData(ImmutableBaseModel):
    name: str
    data: List[Any]
    type: DType

    @validator("type")
    def validate_type(cls, tp):
        if cls.__name__ == "ColumnData":
            raise TypeError(
                "ColumnData should not be instantiated. "
                "Use ColumnDataInt, ColumnDataStr, ColumnDataFloat, ColumnDataJSON  or ColumnDataBinary instead."
            )
        column_type = cls.__fields__["type"].default
        if tp != column_type:
            raise ValueError(
                f"Objects of type {cls.__name__} have a fixed type {column_type}, "
                f"you cannot use {tp} in the constructor."
            )
        return tp


class ColumnDataInt(ColumnData):
    data: List[Union[None, int]]
    type = DType.INT


class ColumnDataStr(ColumnData):
    data: List[Union[None, str]]
    type = DType.STR


class ColumnDataFloat(ColumnData):
    data: List[Union[None, float]]
    type = DType.FLOAT


class ColumnDataJSON(ColumnData):
    data: List[Union[None, str]]
    type = DType.JSON


class ColumnDataBinary(ColumnData):
    data: List[Union[None, int]]
    type = DType.BINARY


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

    def to_pandas(self) -> pd.DataFrame:
        data = {column.name: column.data for column in self.columns}
        return pd.DataFrame(data)


class TabularDataResult(ImmutableBaseModel):
    title: str
    columns: List[Union[ColumnDataInt, ColumnDataStr, ColumnDataFloat]]


# ~~~~~~~~~~~~~~~~~~~ UDFs IO ~~~~~~~~~~~~~~~~~~~~~~ #


class SMPCTablesInfo(ImmutableBaseModel):
    template: TableInfo
    sum_op: Optional[TableInfo]
    min_op: Optional[TableInfo]
    max_op: Optional[TableInfo]


class WorkerUDFDTO(ImmutableBaseModel):
    type: _WorkerUDFDTOType
    value: Any

    @validator("type")
    def validate_type(cls, tp):
        if cls.__name__ == "WorkerUDFDTO":
            raise TypeError(
                "WorkerUDFDTO should not be instantiated. "
                "Use WorkerLiteralDTO, WorkerTableDTO or WorkerSMPCDTO instead."
            )
        udf_argument_type = cls.__fields__["type"].default
        if tp != udf_argument_type:
            raise ValueError(
                f"Objects of type {cls.__name__} have a fixed type {udf_argument_type}, "
                f"you cannot use {tp} in the constructor."
            )
        return tp


class WorkerLiteralDTO(WorkerUDFDTO):
    type = _WorkerUDFDTOType.LITERAL
    value: Any


class WorkerTableDTO(WorkerUDFDTO):
    type = _WorkerUDFDTOType.TABLE
    value: TableInfo


class WorkerSMPCDTO(WorkerUDFDTO):
    type = _WorkerUDFDTOType.SMPC
    value: SMPCTablesInfo


class WorkerUDFPosArguments(ImmutableBaseModel):
    # The WorkerSMPCDTO cannot be used here instead of the Union due to pydantic json deserialization.
    args: List[Union[WorkerLiteralDTO, WorkerTableDTO, WorkerSMPCDTO]]


class WorkerUDFKeyArguments(ImmutableBaseModel):
    # The WorkerSMPCDTO cannot be used here instead of the Union due to pydantic json deserialization.
    args: Dict[str, Union[WorkerLiteralDTO, WorkerTableDTO, WorkerSMPCDTO]]


class WorkerUDFResults(ImmutableBaseModel):
    # The WorkerSMPCDTO cannot be used here instead of the Union due to pydantic json deserialization.
    results: List[Union[WorkerLiteralDTO, WorkerTableDTO, WorkerSMPCDTO]]
