from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from mipengine.controller.algorithm_execution_tasks_handler import (
    INodeAlgorithmTasksHandler,
)
from mipengine.controller.algorithm_executor_node_data_objects import NodeData
from mipengine.controller.algorithm_executor_node_data_objects import SMPCTableNames
from mipengine.controller.algorithm_executor_node_data_objects import TableName
from mipengine.node_tasks_DTOs import NodeSMPCDTO
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import NodeUDFDTO
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFKeyArguments
from mipengine.node_tasks_DTOs import UDFPosArguments


class AsyncResult:
    def get(self, timeout=None):
        pass


class _INode(ABC):
    @abstractmethod
    def get_tables(self) -> List[TableName]:
        pass

    @abstractmethod
    def get_table_schema(self, table_name: TableName) -> TableSchema:
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
    def get_merge_tables(self) -> List[TableName]:
        pass

    @abstractmethod
    def create_merge_table(self, command_id: str, table_names: List[str]):
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
        positional_args: UDFPosArguments,
        keyword_args: UDFKeyArguments,
    ) -> AsyncResult:
        pass

    @abstractmethod
    def get_queued_udf_result(self, async_result: AsyncResult) -> List[TableName]:
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
    def get_tables(self) -> List[TableName]:
        tables = [
            TableName(table_name)
            for table_name in self._node_tasks_handler.get_tables(
                request_id=self.request_id,
                context_id=self.context_id,
            )
        ]
        return tables

    def get_table_schema(self, table_name: TableName) -> TableSchema:
        return self._node_tasks_handler.get_table_schema(
            request_id=self.request_id, table_name=table_name.full_table_name
        )

    def get_table_data(self, table_name: TableName) -> TableData:
        return self._node_tasks_handler.get_table_data(
            request_id=self.request_id,
            table_name=table_name.full_table_name,
        )

    def create_table(self, command_id: str, schema: TableSchema) -> TableName:
        return TableName(
            self._node_tasks_handler.create_table(
                request_id=self.request_id,
                context_id=self.context_id,
                command_id=command_id,
                schema=schema,
            )
        )

    # VIEWS functionality
    def get_views(self) -> List[TableName]:
        result = self._node_tasks_handler.get_views(
            request_id=self.request_id, context_id=self.context_id
        )
        return [TableName(table_name) for table_name in result]

    # MERGE TABLES functionality
    def get_merge_tables(self) -> List[TableName]:
        result = self._node_tasks_handler.get_merge_tables(
            request_id=self.request_id, context_id=self.context_id
        )
        return [TableName(table_name) for table_name in result]

    def create_merge_table(self, command_id: str, table_names: List[str]) -> TableName:
        result = self._node_tasks_handler.create_merge_table(
            request_id=self.request_id,
            context_id=self.context_id,
            command_id=command_id,
            table_names=table_names,
        )
        return TableName(result)

    # REMOTE TABLES functionality
    def get_remote_tables(self) -> List[str]:
        return self._node_tasks_handler.get_remote_tables(
            request_id=self.request_id, context_id=self.context_id
        )

    def create_remote_table(
        self,
        table_name: str,
        table_schema: TableSchema,
        native_node: "_Node",
    ):
        monetdb_socket_addr = native_node.node_address
        self._node_tasks_handler.create_remote_table(
            request_id=self.request_id,
            table_name=table_name,
            table_schema=table_schema,
            original_db_url=monetdb_socket_addr,
        )

    # UDFs functionality
    def queue_run_udf(
        self,
        command_id: str,
        func_name: str,
        positional_args: UDFPosArguments,
        keyword_args: UDFKeyArguments,
        use_smpc: bool = False,
    ) -> AsyncResult:
        return self._node_tasks_handler.queue_run_udf(
            request_id=self.request_id,
            context_id=self.context_id,
            command_id=command_id,
            func_name=func_name,
            positional_args=positional_args,
            keyword_args=keyword_args,
            use_smpc=use_smpc,
        )

    def get_udfs(self, algorithm_name) -> List[str]:
        return self._node_tasks_handler.get_udfs(
            request_id=self.request_id, algorithm_name=algorithm_name
        )

    def get_run_udf_query(
        self, command_id: str, func_name: str, positional_args: List[NodeUDFDTO]
    ) -> Tuple[str, str]:
        return self._node_tasks_handler.get_run_udf_query(
            request_id=self.request_id,
            context_id=self.context_id,
            command_id=command_id,
            func_name=func_name,
            positional_args=positional_args,
        )

    def clean_up(self):
        self._node_tasks_handler.clean_up(
            request_id=self.request_id, context_id=self.context_id
        )


class LocalNode(_Node):
    def create_data_model_views(
        self,
        command_id: str,
        data_model: str,
        datasets: List[str],
        columns_per_view: List[List[str]],
        filters: dict = None,
        dropna: bool = True,
        check_min_rows: bool = True,
    ) -> List[TableName]:
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
        List[TableName]
            A list of views(TableName) created, corresponding to the columns_per_view list.
        """
        views = self._node_tasks_handler.create_data_model_views(
            request_id=self.request_id,
            context_id=self.context_id,
            command_id=command_id,
            data_model=data_model,
            datasets=datasets,
            columns_per_view=columns_per_view,
            filters=filters,
            dropna=dropna,
            check_min_rows=check_min_rows,
        )
        return [TableName(view) for view in views]

    def get_queued_udf_result(self, async_result: AsyncResult) -> List[NodeData]:
        node_udf_results = self._node_tasks_handler.get_queued_udf_result(
            async_result=async_result, request_id=self.request_id
        )
        udf_results = []
        for result in node_udf_results.results:
            if isinstance(result, NodeTableDTO):
                udf_results.append(TableName(result.value))
            elif isinstance(result, NodeSMPCDTO):
                udf_results.append(
                    SMPCTableNames(
                        template=TableName(result.value.template.value),
                        sum_op=create_node_table_from_node_table_dto(
                            result.value.sum_op_values
                        ),
                        min_op=create_node_table_from_node_table_dto(
                            result.value.min_op_values
                        ),
                        max_op=create_node_table_from_node_table_dto(
                            result.value.max_op_values
                        ),
                        union_op=create_node_table_from_node_table_dto(
                            result.value.union_op_values
                        ),
                    )
                )
            else:
                raise NotImplementedError
        return udf_results

    def load_data_to_smpc_client(self, table_name: str, jobid: str) -> int:
        return self._node_tasks_handler.load_data_to_smpc_client(
            self.context_id, table_name, jobid
        )


def create_node_table_from_node_table_dto(node_table_dto: NodeTableDTO):
    if not node_table_dto:
        return None

    return TableName(table_name=node_table_dto.value)


class GlobalNode(_Node):
    def get_queued_udf_result(self, async_result: AsyncResult) -> List[TableName]:
        node_udf_results = self._node_tasks_handler.get_queued_udf_result(
            async_result=async_result, request_id=self.request_id
        )
        results = []
        for result in node_udf_results.results:
            if isinstance(result, NodeTableDTO):
                results.append(TableName(result.value))
            elif isinstance(result, NodeSMPCDTO):
                raise TypeError("A global node should not return an SMPC DTO.")
            else:
                raise NotImplementedError
        return results

    def validate_smpc_templates_match(
        self,
        table_name: str,
    ):
        self._node_tasks_handler.validate_smpc_templates_match(
            self.context_id, table_name
        )

    def get_smpc_result(
        self,
        jobid: str,
        command_id: str,
        command_subid: Optional[str] = "0",
    ) -> str:
        return self._node_tasks_handler.get_smpc_result(
            jobid=jobid,
            context_id=self.context_id,
            command_id=str(command_id),
            command_subid=command_subid,
        )
