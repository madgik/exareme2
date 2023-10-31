from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional
from typing import Tuple

from exareme2.controller.celery.node_tasks_handler import INodeAlgorithmTasksHandler
from exareme2.node_communication import NodeSMPCDTO
from exareme2.node_communication import NodeTableDTO
from exareme2.node_communication import NodeUDFDTO
from exareme2.node_communication import NodeUDFKeyArguments
from exareme2.node_communication import NodeUDFPosArguments
from exareme2.node_communication import TableData
from exareme2.node_communication import TableInfo
from exareme2.node_communication import TableSchema


class AsyncResult:
    def get(self, timeout=None):
        pass


class _INode(ABC):
    @abstractmethod
    def get_tables(self) -> List[str]:
        pass

    @abstractmethod
    def get_table_data(self, table_name: str) -> TableData:
        pass

    @abstractmethod
    def create_table(self, command_id: str, schema: TableSchema) -> TableInfo:
        pass

    @abstractmethod
    def get_views(self) -> List[str]:
        pass

    @abstractmethod
    def get_merge_tables(self) -> List[str]:
        pass

    @abstractmethod
    def create_merge_table(
        self, command_id: str, table_infos: List[TableInfo]
    ) -> TableInfo:
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
        self,
        command_id: str,
        func_name: str,
        positional_args: NodeUDFPosArguments,
        keyword_args: NodeUDFKeyArguments,
    ) -> AsyncResult:
        pass

    @abstractmethod
    def get_queued_udf_result(self, async_result: AsyncResult) -> List[NodeUDFDTO]:
        pass

    @abstractmethod
    def get_udfs(self, algorithm_name) -> List[str]:
        pass


class _Node(_INode, ABC):
    def __init__(
        self,
        request_id: str,
        context_id: str,
        node_tasks_handler: INodeAlgorithmTasksHandler,
    ):
        self._node_tasks_handler: INodeAlgorithmTasksHandler = node_tasks_handler
        self.node_id: str = self._node_tasks_handler.node_id
        self.request_id: str = request_id
        self.context_id: str = context_id

    def __repr__(self):
        return f"{self.node_id}"

    @property
    def node_address(self) -> str:
        return self._node_tasks_handler.node_data_address

    # TABLES functionality
    def get_tables(self) -> List[str]:
        return self._node_tasks_handler.get_tables(
            context_id=self.context_id,
        )

    def get_table_data(self, table_name: str) -> TableData:
        return self._node_tasks_handler.get_table_data(
            table_name=table_name,
        )

    def create_table(self, command_id: str, schema: TableSchema) -> TableInfo:
        return self._node_tasks_handler.create_table(
            context_id=self.context_id,
            command_id=command_id,
            schema=schema,
        )

    # VIEWS functionality
    def get_views(self) -> List[str]:
        return self._node_tasks_handler.get_views(context_id=self.context_id)

    # MERGE TABLES functionality
    def get_merge_tables(self) -> List[str]:
        return self._node_tasks_handler.get_merge_tables(context_id=self.context_id)

    def create_merge_table(
        self, command_id: str, table_infos: List[TableInfo]
    ) -> TableInfo:
        return self._node_tasks_handler.create_merge_table(
            context_id=self.context_id,
            command_id=command_id,
            table_infos=table_infos,
        )

    # REMOTE TABLES functionality
    def get_remote_tables(self) -> List[str]:
        return self._node_tasks_handler.get_remote_tables(context_id=self.context_id)

    def create_remote_table(
        self,
        table_name: str,
        table_schema: TableSchema,
        native_node: "_Node",
    ) -> TableInfo:
        monetdb_socket_addr = native_node.node_address
        return self._node_tasks_handler.create_remote_table(
            table_name=table_name,
            table_schema=table_schema,
            original_db_url=monetdb_socket_addr,
        )

    # UDFs functionality
    def queue_run_udf(
        self,
        command_id: str,
        func_name: str,
        positional_args: NodeUDFPosArguments,
        keyword_args: NodeUDFKeyArguments,
        use_smpc: bool = False,
        output_schema: Optional[TableSchema] = None,
    ) -> AsyncResult:
        return self._node_tasks_handler.queue_run_udf(
            context_id=self.context_id,
            command_id=command_id,
            func_name=func_name,
            positional_args=positional_args,
            keyword_args=keyword_args,
            use_smpc=use_smpc,
            output_schema=output_schema,
        )

    def get_udfs(self, algorithm_name) -> List[str]:
        return self._node_tasks_handler.get_udfs(algorithm_name=algorithm_name)

    def get_run_udf_query(
        self, command_id: str, func_name: str, positional_args: List[NodeUDFDTO]
    ) -> Tuple[str, str]:
        return self._node_tasks_handler.get_run_udf_query(
            context_id=self.context_id,
            command_id=command_id,
            func_name=func_name,
            positional_args=positional_args,
        )


class LocalNode(_Node):
    def __init__(
        self,
        request_id: str,
        context_id: str,
        node_tasks_handler: INodeAlgorithmTasksHandler,
        data_model: str,
        datasets: List[str],
    ):
        super().__init__(request_id, context_id, node_tasks_handler)
        self._data_model = data_model
        self._datasets = datasets

    @property
    def data_model(self):
        return self._data_model

    @property
    def datasets(self):
        return self._datasets

    def create_data_model_views(
        self,
        command_id: str,
        columns_per_view: List[List[str]],
        filters: dict = None,
        dropna: bool = True,
        check_min_rows: bool = True,
    ) -> List[TableInfo]:
        """
        Creates views on a specific data model.

        Parameters
        ----------
        command_id : str
            The id of the command.
        data_model : str
            The data model of the view.
        datasets : str
            The datasets that will be included in the view.
        columns_per_view : List[List[str]]
            A list of column names' for each view to be created.
        filters : dict
            A dict representation of a jQuery QueryBuilder json. (https://querybuilder.js.org/)
        dropna : bool
            Remove NAs from the view.
        check_min_rows : bool
            Raise an exception if there are not enough rows in the view.

        Returns
        ------
        List[TableInfo]
            A list of views(TableInfo) created, corresponding to the columns_per_view list.
        """
        return self._node_tasks_handler.create_data_model_views(
            context_id=self.context_id,
            command_id=command_id,
            data_model=self._data_model,
            datasets=self._datasets,
            columns_per_view=columns_per_view,
            filters=filters,
            dropna=dropna,
            check_min_rows=check_min_rows,
        )

    def get_queued_udf_result(self, async_result: AsyncResult) -> List[NodeUDFDTO]:
        return self._node_tasks_handler.get_queued_udf_result(async_result=async_result)

    def load_data_to_smpc_client(self, table_name: str, jobid: str) -> str:
        return self._node_tasks_handler.load_data_to_smpc_client(table_name, jobid)


class GlobalNode(_Node):
    def get_queued_udf_result(self, async_result: AsyncResult) -> List[NodeTableDTO]:
        node_udf_dtos = self._node_tasks_handler.get_queued_udf_result(
            async_result=async_result
        )
        for dto in node_udf_dtos:
            if isinstance(dto, NodeSMPCDTO):
                raise TypeError("A global node should not return an SMPC DTO.")
        return node_udf_dtos

    def validate_smpc_templates_match(
        self,
        table_name: str,
    ):
        self._node_tasks_handler.validate_smpc_templates_match(table_name)

    def get_smpc_result(
        self,
        jobid: str,
        command_id: str,
        command_subid: Optional[str] = "0",
    ) -> TableInfo:
        return self._node_tasks_handler.get_smpc_result(
            jobid=jobid,
            context_id=self.context_id,
            command_id=str(command_id),
            command_subid=command_subid,
        )
