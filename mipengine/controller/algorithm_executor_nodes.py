from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from mipengine.controller.algorithm_executor_node_data_objects import NodeData
from mipengine.controller.algorithm_executor_node_data_objects import SMPCTableNames
from mipengine.controller.algorithm_executor_node_data_objects import TableName
from mipengine.controller.node_tasks_handler_interface import INodeTasksHandler
from mipengine.controller.node_tasks_handler_interface import IQueuedUDFAsyncResult
from mipengine.node_tasks_DTOs import NodeSMPCDTO
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import NodeUDFDTO
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFKeyArguments
from mipengine.node_tasks_DTOs import UDFPosArguments


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
    def create_data_model_view(
        self,
        command_id: str,
        data_model: str,
        columns: List[str],
        filters: List[str],
    ) -> TableName:
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


class _Node(_INode, ABC):
    def __init__(
        self,
        request_id: str,
        context_id: str,
        node_tasks_handler: INodeTasksHandler,
        initial_view_tables_params: Dict[str, Any] = None,
    ):
        self._node_tasks_handler = node_tasks_handler
        self.node_id = self._node_tasks_handler.node_id
        self.request_id = request_id
        self.context_id = context_id

        self._initial_view_tables = None
        if initial_view_tables_params is not None:
            self._initial_view_tables = self._create_initial_view_tables(
                initial_view_tables_params
            )

    def __repr__(self):
        return f"{self.node_id}"

    @property
    def initial_view_tables(self) -> Dict[str, TableName]:
        return self._initial_view_tables

    def _create_initial_view_tables(
        self, initial_view_tables_params
    ) -> Dict[str, TableName]:
        # will contain the views created from the data model, datasets. Its keys are
        # the variable sets x, y etc
        initial_view_tables = {}

        # initial view for variables in X
        variable = "x"
        if initial_view_tables_params[variable]:
            command_id = str(initial_view_tables_params["commandId"]) + variable
            view_name = self.create_data_model_view(
                command_id=command_id,
                data_model=initial_view_tables_params["data_model"],
                columns=initial_view_tables_params[variable],
                filters=initial_view_tables_params["filters"],
            )
            initial_view_tables["x"] = view_name

        # initial view for variables in Y
        variable = "y"
        if initial_view_tables_params[variable]:
            command_id = str(initial_view_tables_params["commandId"]) + variable
            view_name = self.create_data_model_view(
                command_id=command_id,
                data_model=initial_view_tables_params["data_model"],
                columns=initial_view_tables_params[variable],
                filters=initial_view_tables_params["filters"],
            )

            initial_view_tables["y"] = view_name

        return initial_view_tables

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

    # TODO: this is very specific to mip, very inconsistent with the rest, has to
    # be abstracted somehow
    def create_data_model_view(
        self,
        command_id: str,
        data_model: str,
        columns: List[str],
        filters: List[str],
    ) -> TableName:
        result = self._node_tasks_handler.create_data_model_view(
            request_id=self.request_id,
            context_id=self.context_id,
            command_id=command_id,
            data_model=data_model,
            columns=columns,
            filters=filters,
        )
        return TableName(result)

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
    ) -> IQueuedUDFAsyncResult:
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
    def get_queued_udf_result(
        self, async_result: IQueuedUDFAsyncResult
    ) -> List[NodeData]:
        node_udf_results = self._node_tasks_handler.get_queued_udf_result(async_result)
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
    def get_queued_udf_result(
        self, async_result: IQueuedUDFAsyncResult
    ) -> List[TableName]:
        node_udf_results = self._node_tasks_handler.get_queued_udf_result(async_result)
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
