from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Any

from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import DType
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import ImmutableBaseModel
from mipengine.node_tasks_DTOs import UDFArgumentKind

from mipengine.controller.node_tasks_handler_interface import IQueuedUDFAsyncResult


class TableName:
    def __init__(self, table_name):
        self._full_name = table_name
        full_name_split = self._full_name.split("_")
        self._table_type = full_name_split[0]
        self._node_id = full_name_split[1]
        self._context_id = full_name_split[2]
        self._command_id = full_name_split[3]
        self._command_subid = full_name_split[4]

    @property
    def full_table_name(self):
        return self._full_name

    @property
    def table_type(self):
        return self._table_type

    @property
    def node_id(self):
        return self._node_id

    @property
    def context_id(self):
        return self._context_id

    @property
    def command_id(self):
        return self._command_id

    @property
    def command_subid(self):
        return self._command_subid

    def without_node_id(self):
        return (
            self._table_type
            + "_"
            + self._context_id
            + "_"
            + self._command_id
            + "_"
            + self._command_subid
        )

    def __repr__(self):
        return self.full_table_name


class Literal(ImmutableBaseModel):
    value: Any
    kind = UDFArgumentKind.LITERAL

    class Config:
        allow_mutation = False


class _INode(ABC):
    @abstractmethod
    def get_tables(self) -> List[TableName]:
        pass

    @abstractmethod
    def get_table_schema(self, table_name: TableName):
        pass

    @abstractmethod
    def get_table_data(self, table_name: TableName) -> TableData:
        pass

    @abstractmethod
    def create_table(self, command_id: str, schema: TableSchema) -> TableName:
        pass

    @abstractmethod
    def get_views(self) -> List[TableName]:
        pass

    @abstractmethod
    def create_pathology_view(
        self,
        command_id: str,
        pathology: str,
        columns: List[str],
        filters: List[str],
    ) -> TableName:
        pass

    @abstractmethod
    def get_merge_tables(self) -> List[TableName]:
        pass

    @abstractmethod
    def create_merge_table(self, command_id: str, table_names: List[TableName]):
        pass

    @abstractmethod
    def get_remote_tables(self) -> List[str]:
        pass

    @abstractmethod
    def create_remote_table(
        self, table_name: str, table_schema: TableSchema, native_node: "_INode"
    ):
        pass

    @abstractmethod
    def queue_run_udf(
        self, command_id: str, func_name: str, positional_args, keyword_args
    ) -> IQueuedUDFAsyncResult:
        pass

    @abstractmethod
    def get_queued_udf_result(
        self, async_result: IQueuedUDFAsyncResult
    ) -> List[TableName]:
        pass

    @abstractmethod
    def get_udfs(self, algorithm_name) -> List[str]:
        pass
