from abc import ABC
from abc import abstractmethod
from builtins import str
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import BaseModel

from mipengine.node_tasks_DTOs import NodeUDFDTO
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFKeyArguments
from mipengine.node_tasks_DTOs import UDFPosArguments
from mipengine.node_tasks_DTOs import UDFResults


class IAsyncResult(BaseModel, ABC):
    async_result: Any

    @abstractmethod
    def get(self, timeout=None):
        pass


class IQueuedUDFAsyncResult(IAsyncResult, ABC):
    node_id: str
    command_id: str
    request_id: str
    context_id: str
    func_name: str
    positional_args: Optional[UDFPosArguments] = None
    keyword_args: Optional[UDFKeyArguments] = None
    use_smpc: bool = False


class INodeTasksHandler(ABC):
    @property
    @abstractmethod
    def node_id(self) -> str:
        pass

    @property
    @abstractmethod
    def node_data_address(self) -> str:
        pass

    @property
    @abstractmethod
    def tasks_timeout(self) -> int:
        pass

    # @abstractmethod
    # def get_node_role(self):#TODO does that make sense???
    #     pass

    # TABLES functionality
    @abstractmethod
    def get_tables(self, request_id: str, context_id: str) -> List[str]:
        pass

    @abstractmethod
    def get_table_schema(self, request_id: str, table_name: str):
        pass

    @abstractmethod
    def get_table_data(self, request_id: str, table_name: str) -> TableData:
        pass

    @abstractmethod
    def create_table(
        self, request_id: str, context_id: str, command_id: str, schema: TableSchema
    ) -> str:
        pass

    # VIEWS functionality
    @abstractmethod
    def get_views(self, request_id: str, context_id: str) -> List[str]:
        pass

    @abstractmethod
    def create_data_model_views(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        data_model: str,
        datasets: List[str],
        columns_per_view: List[List[str]],
        filters: dict,
        dropna: bool = True,
        check_min_rows: bool = True,
    ) -> List[str]:
        pass

    # MERGE TABLES functionality
    @abstractmethod
    def get_merge_tables(self, request_id: str, context_id: str) -> List[str]:
        pass

    @abstractmethod
    def create_merge_table(
        self, request_id: str, context_id: str, command_id: str, table_names: List[str]
    ):
        pass

    # REMOTE TABLES functionality
    @abstractmethod
    def get_remote_tables(self, request_id: str, context_id: str) -> List[str]:
        pass

    @abstractmethod
    def create_remote_table(
        self,
        request_id: str,
        table_name: str,
        table_schema: TableSchema,
        original_db_url: str,
    ) -> str:
        pass

    # UDFs functionality
    @abstractmethod
    def queue_run_udf(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: Optional[UDFPosArguments] = None,
        keyword_args: Optional[UDFKeyArguments] = None,
        use_smpc: bool = False,
    ) -> IQueuedUDFAsyncResult:
        pass

    @abstractmethod
    def get_queued_udf_result(self, async_result: IQueuedUDFAsyncResult) -> UDFResults:
        pass

    @abstractmethod
    def get_udfs(self, request_id: str, algorithm_name) -> List[str]:
        pass

    # return the generated monetdb python udf
    @abstractmethod
    def get_run_udf_query(
        self,
        request_id: str,
        context_id: str,
        command_id: str,
        func_name: str,
        positional_args: List[NodeUDFDTO],
    ) -> Tuple[str, str]:
        pass

    # CLEANUP functionality
    @abstractmethod
    def clean_up(self, request_id: str, context_id: str):
        pass

    # ------------- SMPC functionality ---------------
    @abstractmethod
    def validate_smpc_templates_match(
        self,
        context_id: str,
        table_name: str,
    ):
        pass

    @abstractmethod
    def load_data_to_smpc_client(
        self, context_id: str, table_name: str, jobid: str
    ) -> int:
        pass

    @abstractmethod
    def get_smpc_result(
        self,
        jobid: str,
        context_id: str,
        command_id: str,
        command_subid: Optional[str] = "0",
    ) -> str:
        pass
