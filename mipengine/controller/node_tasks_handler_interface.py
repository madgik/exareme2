from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Tuple
from typing import Optional
from typing import Dict

from pydantic import BaseModel

from mipengine.node_tasks_DTOs import ImmutableBaseModel
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFArgument


class IAsyncResult(BaseModel, ABC):
    async_result: Any

    @abstractmethod
    def get(self, timeout=None):
        pass


class UDFPosArguments(ImmutableBaseModel):
    args: List[UDFArgument]


class UDFKeyArguments(ImmutableBaseModel):
    kwargs: Dict[str, UDFArgument]


class IQueuedUDFAsyncResult(IAsyncResult, ABC):
    node_id: str
    command_id: str
    context_id: str
    func_name: str
    positional_args: Optional[UDFPosArguments] = None
    keyword_args: Optional[UDFKeyArguments] = None


class INodeTasksHandler(ABC):
    @property
    @abstractmethod
    def node_id(self) -> str:
        pass

    @property
    @abstractmethod
    def node_data_address(self) -> str:
        pass

    # @abstractmethod
    # def get_node_role(self):#TODO does that make sense???
    #     pass

    # TABLES functionality
    @abstractmethod
    def get_tables(self, context_id: str) -> List[str]:
        pass

    @abstractmethod
    def get_table_schema(self, table_name: str):
        pass

    @abstractmethod
    def get_table_data(self, table_name: str) -> TableData:
        pass

    @abstractmethod
    def create_table(
        self, context_id: str, command_id: str, schema: TableSchema
    ) -> str:
        pass

    # VIEWS functionality
    @abstractmethod
    def get_views(self, context_id: str) -> List[str]:
        pass

    # TODO: this is very specific to mip, very inconsistent with the rest, has to be
    # abstracted somehow
    @abstractmethod
    def create_pathology_view(
        self,
        context_id: str,
        command_id: str,
        pathology: str,
        columns: List[str],
        filters: List[str],
    ) -> str:
        pass

    # MERGE TABLES functionality
    @abstractmethod
    def get_merge_tables(self, context_id: str) -> List[str]:
        pass

    @abstractmethod
    def create_merge_table(
        self, context_id: str, command_id: str, table_names: List[str]
    ):
        pass

    # REMOTE TABLES functionality
    @abstractmethod
    def get_remote_tables(self, context_id: str) -> List[str]:
        pass

    @abstractmethod
    def create_remote_table(
        self, table_name: str, table_schema: TableSchema, original_db_url: str
    ) -> str:  # TODO create
        pass

    # UDFs functionality
    @abstractmethod
    def queue_run_udf(
        self,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: Optional[UDFPosArguments] = None,
        keyword_args: Optional[UDFKeyArguments] = None,
    ) -> IQueuedUDFAsyncResult:
        pass

    @abstractmethod
    def get_queued_udf_result(self, async_result: IQueuedUDFAsyncResult) -> List[str]:
        pass

    @abstractmethod
    def get_udfs(self, algorithm_name) -> List[str]:
        pass

    # return the generated monetdb pythonudf
    @abstractmethod
    def get_run_udf_query(
        self,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: List[str],
    ) -> Tuple[str, str]:
        pass

    # CLEANUP functionality
    @abstractmethod
    def clean_up(self, context_id: str):
        pass
