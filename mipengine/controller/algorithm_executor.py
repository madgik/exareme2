from abc import ABC

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from billiard.exceptions import SoftTimeLimitExceeded
from billiard.exceptions import TimeLimitExceeded
from celery.exceptions import TimeoutError
from pydantic import BaseModel

from mipengine import algorithm_modules
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.controller_common_data_elements import (
    controller_common_data_elements,
)
from mipengine.controller.algorithm_execution_DTOs import AlgorithmExecutionDTO
from mipengine.controller.algorithm_execution_DTOs import NodesTasksHandlersDTO
from mipengine.controller.node_tasks_handler_interface import INodeTasksHandler

from mipengine.controller.node_tasks_handler_interface import IQueuedUDFAsyncResult
from mipengine.controller.node_tasks_handler_interface import UDFPosArguments
from mipengine.controller.node_tasks_handler_interface import UDFKeyArguments
from mipengine.controller.node_tasks_handler_celery import ClosedBrokerConnectionError
from mipengine.controller import controller_logger as ctrl_logger

from mipengine.controller.algorithm_executor_helpers import _INode
from mipengine.controller.algorithm_executor_helpers import TableName

from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFArgument
from mipengine.node_tasks_DTOs import UDFArgumentKind

from mipengine.algorithm_flow_DTOs import Literal


class AlgorithmExecutionException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class NodeDownAlgorithmExecutionException(Exception):
    def __init__(self):
        message = (
            "One of the nodes participating in the algorithm execution "
            "stopped responding"
        )
        super().__init__(message)
        self.message = message


class _Node(_INode):
    def __init__(
        self,
        context_id: str,
        node_tasks_handler: INodeTasksHandler,
        initial_view_tables_params: Dict[str, Any] = None,
    ):
        self._node_tasks_handler = node_tasks_handler
        self.node_id = self._node_tasks_handler.node_id

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
        # will contain the views created from the pathology, datasets. Its keys are
        # the variable sets x, y etc
        initial_view_tables = {}

        # initial view for variables in X
        variable = "x"
        if initial_view_tables_params[variable]:
            command_id = str(initial_view_tables_params["commandId"]) + variable
            view_name = self.create_pathology_view(
                command_id=command_id,
                pathology=initial_view_tables_params["pathology"],
                columns=initial_view_tables_params[variable],
                filters=initial_view_tables_params["filters"],
            )
            initial_view_tables["x"] = view_name

        # initial view for variables in Y
        variable = "y"
        if initial_view_tables_params[variable]:
            command_id = str(initial_view_tables_params["commandId"]) + variable
            view_name = self.create_pathology_view(
                command_id=command_id,
                pathology=initial_view_tables_params["pathology"],
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
        return self._node_tasks_handler.get_tables(context_id=self.context_id)

    def get_table_schema(self, table_name: TableName) -> TableSchema:
        return self._node_tasks_handler.get_table_schema(
            table_name=table_name.full_table_name
        )

    def get_table_data(self, table_name: TableName) -> TableData:
        return self._node_tasks_handler.get_table_data(table_name.full_table_name)

    def create_table(self, command_id: str, schema: TableSchema) -> TableName:
        schema_json = schema.json()
        return self._node_tasks_handler.create_table(
            context_id=self.context_id,
            command_id=command_id,
            schema=schema_json,
        )

    # VIEWS functionality
    def get_views(self) -> List[TableName]:
        result = self._node_tasks_handler.get_views(context_id=self.context_id)
        return [TableName(table_name) for table_name in result]

    # TODO: this is very specific to mip, very inconsistent with the rest, has to
    # be abstracted somehow
    def create_pathology_view(
        self,
        command_id: str,
        pathology: str,
        columns: List[str],
        filters: List[str],
    ) -> TableName:

        result = self._node_tasks_handler.create_pathology_view(
            context_id=self.context_id,
            command_id=command_id,
            pathology=pathology,
            columns=columns,
            filters=filters,
        )
        return TableName(result)

    # MERGE TABLES functionality

    def get_merge_tables(self) -> List[TableName]:
        result = self._node_tasks_handler.get_merge_tables(context_id=self.context_id)
        return [TableName(table_name) for table_name in result]

    def create_merge_table(
        self, command_id: str, table_names: List[TableName]
    ) -> TableName:
        table_names = [table_name.full_table_name for table_name in table_names]
        result = self._node_tasks_handler.create_merge_table(
            context_id=self.context_id,
            command_id=command_id,
            table_names=table_names,
        )
        return TableName(result)

    # REMOTE TABLES functionality

    def get_remote_tables(self) -> List[str]:
        return self._node_tasks_handler.get_remote_tables(context_id=self.context_id)

    def create_remote_table(
        self, table_name: str, table_schema: TableSchema, native_node: "_Node"
    ):

        monetdb_socket_addr = native_node.node_address
        self._node_tasks_handler.create_remote_table(
            table_name=table_name,
            table_schema=table_schema,
            original_db_url=monetdb_socket_addr,
        )

    # UDFs functionality
    def queue_run_udf(
        self,
        command_id: str,
        func_name: str,
        positional_args: Optional[UDFPosArguments] = None,
        keyword_args: Optional[UDFKeyArguments] = None,
    ) -> IQueuedUDFAsyncResult:
        return self._node_tasks_handler.queue_run_udf(
            context_id=self.context_id,
            command_id=command_id,
            func_name=func_name,
            positional_args=positional_args,
            keyword_args=keyword_args,
        )

    def get_queued_udf_result(
        self, async_result: IQueuedUDFAsyncResult
    ) -> List[TableName]:

        result = self._node_tasks_handler.get_queued_udf_result(async_result)
        return [TableName(table) for table in result]

    def get_udfs(self, algorithm_name) -> List[str]:
        return self._node_tasks_handler.get_udfs(algorithm_name)

    def get_run_udf_query(
        self, command_id: str, func_name: str, positional_args: List["_INodeTable"]
    ) -> Tuple[str, str]:
        return self._node_tasks_handler.get_run_udf_query(
            context_id=self.context_id,
            command_id=command_id,
            func_name=func_name,
            positional_args=positional_args,
        )

    def clean_up(self):
        self._node_tasks_handler.clean_up(context_id=self.context_id)


class INodeTable(ABC):
    # TODO: better abstraction here...
    pass


class _LocalNodeTable(INodeTable):
    def __init__(self, nodes_tables: Dict[_Node, TableName]):
        self._nodes_tables = nodes_tables
        self._validate_matching_table_names(list(self._nodes_tables.values()))

    @property
    def nodes_tables(self) -> Dict[_Node, TableName]:
        return self._nodes_tables

    # TODO this is redundant, either remove it or overload all node methods here?
    def get_table_schema(self) -> TableSchema:
        node = list(self.nodes_tables.keys())[0]
        table = self.nodes_tables[node]
        return node.get_table_schema(table)

    def get_table_data(self) -> List[Union[int, float, str]]:
        tables_data = []
        for node, table_name in self.nodes_tables.items():
            tables_data.append(node.get_table_data(table_name))
        tables_data_flat = [table_data.columns for table_data in tables_data]
        tables_data_flat = [
            elem
            for table in tables_data_flat
            for column in table
            for elem in column.data
        ]
        return tables_data_flat

    def __repr__(self):
        r = f"\n\tLocalNodeTable: {self.get_table_schema()}\n"
        for node, table_name in self.nodes_tables.items():
            r += f"\t{node=} {table_name=}\n"
        return r

    def _validate_matching_table_names(self, table_names: List[TableName]) -> bool:
        table_name_without_node_id = table_names[0].without_node_id()
        for table_name in table_names:
            if table_name.without_node_id() != table_name_without_node_id:
                raise self.MismatchingTableNamesException(
                    [table_name.full_table_name for table_name in nodes_tables.values()]
                )

    class MismatchingTableNamesException(Exception):
        def __init__(self, table_names):
            self.message = f"Mismatched table names ->{table_names}"


class _GlobalNodeTable(INodeTable):
    def __init__(self, node: _Node, table_name: TableName):
        self._node = node
        self._table_name = table_name

    @property
    def node(self) -> _INode:
        return self._node

    @property
    def table_name(self) -> TableName:
        return self._table_name

    # TODO this is redundant, either remove it or overload all node methods here?
    def get_table_schema(self) -> TableSchema:
        table_schema: TableSchema = self.node.get_table_schema(self.table_name).columns
        return table_schema

    def get_table_data(self) -> List[Union[int, float, str]]:
        table_data = [
            column.data for column in self.node.get_table_data(self.table_name).columns
        ]
        return table_data

    def __repr__(self):
        r = f"\n\tGlobalNodeTable: \n\tschema={self.get_table_schema()}\n \t{self.table_name=}\n"
        return r


class InconsistentTableSchemasException(Exception):
    def __init__(self, tables_schemas: Dict[INodeTable, TableSchema]):
        message = f"Tables: {tables_schemas} do not have a common schema"
        super().__init__(message)


class InconsistentUDFResultSizeException(Exception):
    def __init__(self, result_tables: Dict[int, List[Tuple["_Node", TableName]]]):
        message = (
            f"The following udf execution results on multiple nodes should have "
            f"the same number of results.\nResults:{result_tables}"
        )
        super().__init__(message)


class InconsistentShareTablesValueException(Exception):
    def __init__(
        self, share_list: Union[bool, List[bool]], number_of_result_tables: int
    ):
        message = f"The size of the {share_list=} does not match the {number_of_result_tables=}"
        super().__init__(message)


class AlgorithmExecutor:
    def __init__(
        self,
        algorithm_execution_dto: AlgorithmExecutionDTO,
        nodes_tasks_handlers_dto: NodesTasksHandlersDTO,
    ):
        self._logger = ctrl_logger.get_request_logger(
            context_id=algorithm_execution_dto.context_id
        )
        self._nodes_tasks_handlers_dto = nodes_tasks_handlers_dto
        self._algorithm_execution_dto = algorithm_execution_dto

        self._context_id = algorithm_execution_dto.context_id
        self._algorithm_name = algorithm_execution_dto.algorithm_name

        self._global_node = None
        self._local_nodes = []
        self._execution_interface = None

    def _instantiate_nodes(self):

        # instantiate the GLOBAL Node object
        self._global_node = _Node(
            context_id=self._context_id,
            node_tasks_handler=self._nodes_tasks_handlers_dto.global_node_tasks_handler,
        )

        # Parameters for the creation of the view tables in the db. Each of the LOCAL
        # nodes will have access only to these view tables and not on the primary data
        # tables
        initial_view_tables_params = {
            "commandId": get_next_command_id(),
            "pathology": self._algorithm_execution_dto.algorithm_request_dto.inputdata.pathology,
            "datasets": self._algorithm_execution_dto.algorithm_request_dto.inputdata.datasets,
            "x": self._algorithm_execution_dto.algorithm_request_dto.inputdata.x,
            "y": self._algorithm_execution_dto.algorithm_request_dto.inputdata.y,
            "filters": self._algorithm_execution_dto.algorithm_request_dto.inputdata.filters,
        }

        # instantiate the LOCAL Node objects
        self._local_nodes = [
            _Node(
                context_id=self._context_id,
                node_tasks_handler=node_tasks_handler,
                initial_view_tables_params=initial_view_tables_params,
            )
            for node_tasks_handler in self._nodes_tasks_handlers_dto.local_nodes_tasks_handlers
        ]

    def _instantiate_algorithm_execution_interface(self):
        algo_execution_interface_dto = _AlgorithmExecutionInterfaceDTO(
            global_node=self._global_node,
            local_nodes=self._local_nodes,
            algorithm_name=self._algorithm_name,
            algorithm_parameters=self._algorithm_execution_dto.algorithm_request_dto.parameters,
            x_variables=self._algorithm_execution_dto.algorithm_request_dto.inputdata.x,
            y_variables=self._algorithm_execution_dto.algorithm_request_dto.inputdata.y,
            pathology=self._algorithm_execution_dto.algorithm_request_dto.inputdata.pathology,
            datasets=self._algorithm_execution_dto.algorithm_request_dto.inputdata.datasets,
        )
        self._execution_interface = _AlgorithmExecutionInterface(
            algo_execution_interface_dto
        )

        # Get algorithm module
        self.algorithm_flow_module = algorithm_modules[self._algorithm_name]

    def run(self):
        try:
            self._instantiate_nodes()
            self._logger.info(
                f"executing algorithm:{self._algorithm_name} on "
                f"local nodes: {self._local_nodes=}"
            )
            self._instantiate_algorithm_execution_interface()
            algorithm_result = self.algorithm_flow_module.run(self._execution_interface)
            self._logger.info(f"finished execution of algorithm:{self._algorithm_name}")
            return algorithm_result
        except (
            SoftTimeLimitExceeded,
            TimeLimitExceeded,
            TimeoutError,
            ClosedBrokerConnectionError,
        ) as err:
            self._logger.error(f"{err=}")

            raise NodeDownAlgorithmExecutionException()
        except Exception as exc:
            import traceback

            self._logger.error(f"{traceback.format_exc()}")
            raise exc
        finally:
            self.clean_up()

    def clean_up(self):
        self._logger.info(f"cleaning up global_node")
        try:
            self._global_node.clean_up()
        except Exception as exc:
            self._logger.error(f"cleaning up global_node FAILED {exc=}")
        self._logger.info(f"cleaning up local nodes:{self._local_nodes}")
        for node in self._local_nodes:
            self._logger.info(f"\tcleaning up {node=}")
            try:
                node.clean_up()
            except Exception as exc:
                self._logger.error(f"cleaning up {node=} FAILED {exc=}")


class _AlgorithmExecutionInterfaceDTO(BaseModel):
    global_node: _Node
    local_nodes: List[_Node]
    algorithm_name: str
    algorithm_parameters: Optional[Dict[str, List[str]]] = None
    x_variables: Optional[List[str]] = None
    y_variables: Optional[List[str]] = None
    pathology: str
    datasets: List[str]

    class Config:
        arbitrary_types_allowed = True


class _AlgorithmExecutionInterface:
    def __init__(self, algo_execution_interface_dto: _AlgorithmExecutionInterfaceDTO):
        self._global_node = algo_execution_interface_dto.global_node
        self._local_nodes = algo_execution_interface_dto.local_nodes
        self._algorithm_name = algo_execution_interface_dto.algorithm_name
        self._algorithm_parameters = algo_execution_interface_dto.algorithm_parameters
        self._x_variables = algo_execution_interface_dto.x_variables
        self._y_variables = algo_execution_interface_dto.y_variables
        pathology = algo_execution_interface_dto.pathology
        self._datasets = algo_execution_interface_dto.datasets
        cdes = controller_common_data_elements.pathologies[pathology]
        varnames = (self._x_variables or []) + (self._y_variables or [])
        self._metadata = {
            varname: cde.__dict__
            for varname, cde in cdes.items()
            if varname in varnames
        }

        if len(self._local_nodes) == 1:
            self._global_node = self._local_nodes[0]

        # TODO: validate all local nodes have created the base_view_table??
        self._initial_view_tables = {}
        tmp_variable_node_table = {}

        # TODO: clean up this mindfuck??
        # https://github.com/madgik/MIP-Engine/pull/132#discussion_r727076138
        for node in self._local_nodes:
            for (variable_name, table_name) in node.initial_view_tables.items():
                if variable_name in tmp_variable_node_table:
                    tmp_variable_node_table[variable_name].update({node: table_name})
                else:
                    tmp_variable_node_table[variable_name] = {node: table_name}

        self._initial_view_tables = {
            variable_name: _LocalNodeTable(node_table)
            for (variable_name, node_table) in tmp_variable_node_table.items()
        }

    def _is_single_node_execution(self) -> bool:
        return self._global_node == self._local_nodes[0]

    @property
    def initial_view_tables(self) -> Dict[str, _LocalNodeTable]:
        return self._initial_view_tables

    @property
    def algorithm_parameters(self) -> Dict[str, Any]:
        return self._algorithm_parameters

    @property
    def x_variables(self) -> List[str]:
        return self._x_variables

    @property
    def y_variables(self) -> List[str]:
        return self._y_variables

    @property
    def metadata(self):
        return self._metadata

    @property
    def datasets(self):
        return self._datasets

    # UDFs functionality
    def run_udf_on_local_nodes(
        self,
        func_name: str,
        positional_args: Optional[List[Union[_LocalNodeTable, Literal]]] = None,
        keyword_args: Optional[Dict[str, Union[_LocalNodeTable, Literal]]] = None,
        share_to_global: Union[bool, List[bool]] = False,
    ) -> List[INodeTable]:
        # 1. check positional_args and keyword_args tables do not contain _GlobalNodeTable(s)
        # 2. queues run_udf task on all local nodes
        # 3. waits for all nodes to complete the tasks execution
        # 4. one(or multiple) new table(s) per local node was generated
        # 5. create remote tables on global for each of the generated tables
        # 6. create merge table on global node to merge the remote tables

        command_id = get_next_command_id()

        self._validate_run_udf_on_local_nodes_args(
            positional_args=positional_args, keyword_args=keyword_args
        )

        # Queue the udf on all local nodes
        tasks = {}
        for node in self._local_nodes:
            positional_udf_args = (
                self._algoexec_udf_posargs_to_node_udf_posargs(positional_args, node)
                if positional_args
                else None
            )
            keyword_udf_args = (
                self._algoexec_udf_kwargs_to_node_udf_kwargs(keyword_args, node)
                if keyword_args
                else None
            )

            task = node.queue_run_udf(
                command_id=command_id,
                func_name=func_name,
                positional_args=positional_udf_args,
                keyword_args=keyword_udf_args,
            )
            tasks[node] = task

        # Get udf results from each local node
        all_nodes_result_tables = self._get_run_udf_results(tasks)

        # validate and transform share_to_global variable
        number_of_results = len(all_nodes_result_tables.keys())
        self._validate_share_to(share_to_global, number_of_results)
        if not isinstance(share_to_global, list):
            share_to_global = [share_to_global]

        # Handle sharing results to global node
        if share_to_global:
            results_after_sharing_step = self._handle_table_sharing_locals_to_global(
                share_list=share_to_global,
                indexed_node_tables=all_nodes_result_tables,
                command_id=command_id,
            )
        else:
            # if nothing to share, package all tables as _LocalNodeTable(s)
            results_after_sharing_step = []
            for index, node_tables in all_nodes_result_tables.items():
                results_after_sharing_step.append(
                    _LocalNodeTable(nodes_tables=dict(node_tables))
                )

        # TODO always return list??
        if len(results_after_sharing_step) == 1:
            results_after_sharing_step = results_after_sharing_step[0]

        return results_after_sharing_step

    def run_udf_on_global_node(
        self,
        func_name: str,
        positional_args: Optional[List[Union[_GlobalNodeTable, Literal]]] = None,
        keyword_args: Optional[Dict[str, Union[_GlobalNodeTable, Literal]]] = None,
        share_to_locals: Union[bool, List[bool]] = False,
    ) -> List[INodeTable]:
        # 1. check positional_args and keyword_args tables do not contain _LocalNodeTable(s)
        # 2. queue run_udf on the global node
        # 3. wait for it to complete
        # 4. a(or multiple) new table(s) was generated on global node
        # 5. queue create_remote_table on each of the local nodes to share the generated table

        command_id = get_next_command_id()

        self._validate_run_udf_on_global_node_args(
            positional_args=positional_args, keyword_args=keyword_args
        )

        positional_udf_args = (
            self._algoexec_udf_posargs_to_node_udf_posargs(positional_args)
            if positional_args
            else None
        )
        keyword_udf_args = (
            self._algoexec_udf_kwargs_to_node_udf_kwargs(keyword_args)
            if keyword_args
            else None
        )

        # Queue the udf on global node
        task = self._global_node.queue_run_udf(
            command_id=command_id,
            func_name=func_name,
            positional_args=positional_udf_args,
            keyword_args=keyword_udf_args,
        )

        # Get udf result from global node
        result_tables = self._global_node.get_queued_udf_result(task)

        # validate and transform share_to_locals variable
        number_of_results = len(result_tables)
        self._validate_share_to(share_to_locals, number_of_results)
        if not isinstance(share_to_locals, list):
            share_to_locals = [share_to_locals]

        # Handle sharing result to local nodes
        if share_to_locals:
            final_results = self._handle_table_sharing_global_to_locals(
                share_list=share_to_locals, tables=result_tables, command_id=command_id
            )

        else:
            final_results = [
                _GlobalNodeTable(node=self._global_node, table_name=result_table)
                for result_table in result_tables
            ]

        # backward compatibility.. TODO always return list??
        if len(final_results) == 1:
            final_results = final_results[0]
        return final_results

    # TABLES functionality
    def get_table_data(self, node_table) -> TableData:
        return node_table.get_table_data()

    def get_table_schema(self, node_table) -> TableSchema:
        return node_table.get_table_schema()

    # -------------helper methods------------
    def _handle_table_sharing_locals_to_global(
        self,
        share_list: List[bool],
        indexed_node_tables: Dict[int, List[Tuple[_Node, TableName]]],
        command_id: int,
    ) -> List[INodeTable]:
        handled_tables = []
        for index, share in enumerate(share_list):
            nodes_tables: List[Tuple[_Node, TableName]] = indexed_node_tables[index]
            nodes = [tupple[0] for tupple in nodes_tables]
            tables = [tupple[1] for tupple in nodes_tables]
            if share:
                if self._is_single_node_execution():
                    merge_table = tables[0]
                else:
                    # check the tables have the same schema
                    check, tables_schemas, common_schema = check_same_schema_tables(
                        nodes_tables
                    )
                    if check == False:
                        raise InconsistentTableSchemasException(tables_schemas)

                    # create remote tabels on global node
                    for index, node in enumerate(nodes):
                        self._global_node.create_remote_table(
                            table_name=tables[index]._full_name,
                            table_schema=common_schema,
                            native_node=node,
                        )
                    # merge remote tables into one merge table on global node
                    merge_table = self._global_node.create_merge_table(
                        command_id, tables
                    )

                handled_tables.append(
                    _GlobalNodeTable(node=self._global_node, table_name=merge_table)
                )
            else:
                handled_tables.append(_LocalNodeTable(nodes_tables=dict(nodes_tables)))
        return handled_tables

    def _get_run_udf_results(
        self, tasks: Dict[_Node, IQueuedUDFAsyncResult]
    ) -> Dict[int, List[Tuple[_Node, TableName]]]:
        raise_inconsistent_udf_result_exc = False
        all_nodes_result_tables = {}
        expected_number_of_result_tables = None
        for node, task in tasks.items():
            node_result_tables: List[TableName] = node.get_queued_udf_result(task)
            number_of_result_tables_current_node = len(node_result_tables)
            if expected_number_of_result_tables:
                if (
                    number_of_result_tables_current_node
                    != expected_number_of_result_tables
                ):
                    raise_inconsistent_udf_result_exc = True
            else:
                expected_number_of_result_tables = number_of_result_tables_current_node
            for index, task_result in enumerate(node_result_tables):
                if index not in all_nodes_result_tables:
                    all_nodes_result_tables[index] = []
                all_nodes_result_tables[index].append((node, task_result))

        if raise_inconsistent_udf_result_exc:
            raise InconsistentUDFResultSizeException(all_nodes_result_tables)
        else:
            return all_nodes_result_tables

    def _handle_table_sharing_global_to_locals(
        self, share_list: List[bool], tables: List[TableName], command_id=int
    ) -> List[INodeTable]:
        handled_tables = []
        for index, share in enumerate(share_list):
            if share:
                if self._is_single_node_execution():
                    local_tables = [(self._global_node, tables[0])]
                else:
                    local_tables = []
                    table_schema = self._global_node.get_table_schema(tables[index])
                    for node in self._local_nodes:
                        node.create_remote_table(
                            table_name=tables[index]._full_name,
                            table_schema=table_schema,
                            native_node=self._global_node,
                        )
                        local_tables.append((node, tables[index]))
                handled_tables.append(_LocalNodeTable(nodes_tables=dict(local_tables)))

            else:
                handled_tables.append(
                    _GlobalNodeTable(node=self._global_node, table_name=tables[index])
                )
        return handled_tables

    def _validate_share_to(self, share_to: Union[bool, List[bool]], number_of_results):
        if isinstance(share_to, list):
            if len(share_to) != number_of_results:
                raise InconsistentShareTablesValueException(
                    share_to_locals, number_of_result_tables
                )
        elif isinstance(share_to, bool):
            if number_of_results != 1:
                raise InconsistentShareTablesValueException(
                    share_to_locals, number_of_result_tables
                )
        else:
            raise Exception(
                f"share_to_locals must be of type bool or List[bool] but "
                f"{type(share_to)=} was passed"
            )

    # check positional_args and keyword_args do not contain _LocalNodeTable(s)
    def _validate_run_udf_on_global_node_args(
        self,
        positional_args: Optional[List[Union[_GlobalNodeTable, Literal]]] = None,
        keyword_args: Optional[Dict[str, Union[_GlobalNodeTable, Literal]]] = None,
    ):
        for arg in positional_args or []:
            if not isinstance(arg, _GlobalNodeTable) and not isinstance(arg, Literal):
                raise Exception(
                    f"positional_args contains {arg=} of "
                    f"type {type(arg)=} which is not acceptable from "
                    f"run_udf_on_global_node. {positional_args=}"
                )
        if keyword_args:
            for arg in keyword_args.values():
                if not isinstance(arg, _GlobalNodeTable) and not isinstance(
                    arg, Literal
                ):
                    raise Exception(
                        f"keyword_args contains {arg=} of "
                        f"type {type(arg)=} which is not acceptable from "
                        f"run_udf_on_global_node. {keyword_args=}"
                    )

    # check positional_args and keyword_args do not contain _GlobalNodeTable(s)
    def _validate_run_udf_on_local_nodes_args(
        self,
        positional_args: Optional[List[Union[_LocalNodeTable, Literal]]] = None,
        keyword_args: Optional[Dict[str, Union[_LocalNodeTable, Literal]]] = None,
    ):
        for arg in positional_args or []:
            if not isinstance(arg, _LocalNodeTable) and not isinstance(arg, Literal):
                raise Exception(
                    f"positional_args contains {arg=} of "
                    f"type {type(arg)=} which is not acceptable from "
                    f"run_udf_on_local_nodes. {positional_args=}"
                )
        if keyword_args:
            for arg in keyword_args.values():
                if not isinstance(arg, _LocalNodeTable) and not isinstance(
                    arg, Literal
                ):
                    raise Exception(
                        f"keyword_args contains {arg=} of "
                        f"type {type(arg)=} which is not acceptable from "
                        f"run_udf_on_local_nodes. {keyword_args=}"
                    )

    def _algoexec_udf_kwargs_to_node_udf_kwargs(
        self,
        algoexec_kwargs: Dict[str, Union[INodeTable, Literal]],
        node: _Node = None,
    ) -> UDFKeyArguments:
        udf_kwargs = UDFKeyArguments(kwargs={})
        for key, val in algoexec_kwargs.items():
            udf_argument = self._algoexec_udf_arg_to_node_udf_arg(val, node)
            udf_kwargs.kwargs[key] = udf_argument.json()
        return udf_kwargs

    def _algoexec_udf_posargs_to_node_udf_posargs(
        self, algoexec_posargs: List[Union[INodeTable, Literal]], node: _Node = None
    ) -> UDFPosArguments:
        udf_posargs = UDFPosArguments(args=[])
        for val in algoexec_posargs:
            udf_argument = self._algoexec_udf_arg_to_node_udf_arg(val, node)
            udf_posargs.args.append(udf_argument.json())
        return udf_posargs

    def _algoexec_udf_arg_to_node_udf_arg(
        self, algoexec_arg: Union[INodeTable, Literal], node: _Node = None
    ) -> UDFArgument:
        if isinstance(algoexec_arg, INodeTable):
            if node:
                table_name = algoexec_arg.nodes_tables[node].full_table_name
            else:
                table_name = algoexec_arg.table_name.full_table_name
            udf_argument = UDFArgument(
                kind=UDFArgumentKind.TABLE,
                value=table_name,
            )
        elif isinstance(algoexec_arg, Literal):
            udf_argument = UDFArgument(kind=algoexec_arg.kind, value=algoexec_arg.value)
        else:
            # TODO specific exception
            raise Exception(f"{type(algoexec_arg)=} is not accepted...")
        return udf_argument


def check_same_schema_tables(
    tables: List[Tuple[_INode, TableName]]
) -> (bool, Dict[TableName, TableSchema], TableSchema):
    """
    Returns :
    bool: True if all tables have the same schema.
    Dict[TableName, TableSchema]: keys:table name and values:the corresponding
    table schema
    TableSchema: the common TableSchema, if all tables have the same schema, else None
    """

    have_common_schema = True
    schemas = {}
    reference_schema = None
    for node, table in tables:
        schemas[table] = node.get_table_schema(table)
        if reference_schema:
            if schemas[table] != reference_schema:
                have_common_schema = False
        else:
            reference_schema = schemas[table]

    if have_common_schema:
        return have_common_schema, schemas, reference_schema
    else:
        return have_common_schema, schemas, None


def get_next_command_id() -> int:
    if hasattr(get_next_command_id, "index"):
        get_next_command_id.index += 1
    else:
        get_next_command_id.index = 0
    return get_next_command_id.index
