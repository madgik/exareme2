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
from mipengine.controller.algorithm_execution_DTOs import AlgorithmExecutionDTO
from mipengine.controller.algorithm_execution_DTOs import NodesTasksHandlersDTO
from mipengine.controller.algorithm_executor_node_data_objects import AlgoExecData
from mipengine.controller.algorithm_executor_node_data_objects import GlobalNodeData
from mipengine.controller.algorithm_executor_node_data_objects import (
    GlobalNodeSMPCTables,
)
from mipengine.controller.algorithm_executor_node_data_objects import GlobalNodeTable
from mipengine.controller.algorithm_executor_node_data_objects import LocalNodesData
from mipengine.controller.algorithm_executor_node_data_objects import (
    LocalNodesSMPCTables,
)
from mipengine.controller.algorithm_executor_node_data_objects import LocalNodesTable
from mipengine.controller.algorithm_executor_node_data_objects import NodeData
from mipengine.controller.algorithm_executor_node_data_objects import NodeSMPCTables
from mipengine.controller.algorithm_executor_node_data_objects import NodeTable
from mipengine.controller.algorithm_executor_node_data_objects import (
    algoexec_udf_kwargs_to_node_udf_kwargs,
)
from mipengine.controller.algorithm_executor_node_data_objects import (
    algoexec_udf_posargs_to_node_udf_posargs,
)
from mipengine.controller.algorithm_executor_nodes import GlobalNode
from mipengine.controller.algorithm_executor_nodes import LocalNode
from mipengine.controller.controller_common_data_elements import (
    controller_common_data_elements,
)
from mipengine.controller.node_tasks_handler_celery import ClosedBrokerConnectionError
from mipengine.controller.node_tasks_handler_interface import IQueuedUDFAsyncResult
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema


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


class InconsistentTableSchemasException(Exception):
    def __init__(self, tables_schemas: Dict[NodeTable, TableSchema]):
        message = f"Tables: {tables_schemas} do not have a common schema"
        super().__init__(message)


class InconsistentUDFResultSizeException(Exception):
    def __init__(self, result_tables: Dict[int, List[Tuple[LocalNode, NodeTable]]]):
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

        self._global_node: GlobalNode = None
        self._local_nodes: List[LocalNode] = []
        self._algorithm_flow_module = None
        self._execution_interface = None

    def _instantiate_nodes(self):

        # instantiate the GLOBAL Node object
        self._global_node = GlobalNode(
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
            LocalNode(
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
        if len(self._local_nodes) > 1:
            self._execution_interface = _AlgorithmExecutionInterface(
                algo_execution_interface_dto
            )
        else:
            self._execution_interface = _SingleLocalNodeAlgorithmExecutionInterface(
                algo_execution_interface_dto
            )

        # Get algorithm module
        self._algorithm_flow_module = algorithm_modules[self._algorithm_name]

    def run(self):
        try:
            self._instantiate_nodes()
            self._logger.info(
                f"executing algorithm:{self._algorithm_name} on "
                f"local nodes: {self._local_nodes=}"
            )
            self._instantiate_algorithm_execution_interface()
            algorithm_result = self._algorithm_flow_module.run(
                self._execution_interface
            )
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
        self._logger.info("cleaning up global_node")
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
    global_node: GlobalNode
    local_nodes: List[LocalNode]
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
        self._global_node: GlobalNode = algo_execution_interface_dto.global_node
        self._local_nodes: List[LocalNode] = algo_execution_interface_dto.local_nodes
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
            variable_name: LocalNodesTable(node_table)
            for (variable_name, node_table) in tmp_variable_node_table.items()
        }

    @property
    def initial_view_tables(self) -> Dict[str, LocalNodesTable]:
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
        positional_args: Optional[List[Any]] = None,
        keyword_args: Optional[Dict[str, Any]] = None,
        share_to_global: Union[None, bool, List[bool]] = None,
    ) -> Union[AlgoExecData, List[AlgoExecData]]:
        # 1. check positional_args and keyword_args tables do not contain _GlobalNodeTable(s)
        # 2. queues run_udf task on all local nodes
        # 3. waits for all nodes to complete the tasks execution
        # 4. one(or multiple) new table(s) per local node was generated
        # 5. create remote tables on global for each of the generated tables
        # 6. create merge table on global node to merge the remote tables

        command_id = get_next_command_id()

        self._validate_local_run_udf_args(
            positional_args=positional_args,
            keyword_args=keyword_args,
        )

        # Queue the udf on all local nodes
        tasks = {}
        for node in self._local_nodes:
            positional_udf_args = algoexec_udf_posargs_to_node_udf_posargs(
                positional_args, node
            )
            keyword_udf_args = algoexec_udf_kwargs_to_node_udf_kwargs(
                keyword_args, node
            )

            task = node.queue_run_udf(
                command_id=str(command_id),
                func_name=func_name,
                positional_args=positional_udf_args,
                keyword_args=keyword_udf_args,
            )
            tasks[node] = task

        all_nodes_results = self._get_local_run_udfs_results(tasks)
        all_local_nodes_data = self._convert_local_udf_results_to_local_nodes_data(
            all_nodes_results
        )

        results_after_sharing_step = all_local_nodes_data
        if share_to_global is not None:
            # validate and transform share_to_global variable
            if not isinstance(share_to_global, list):
                share_to_global = [share_to_global]
            number_of_results = len(all_nodes_results)
            self._validate_share_to(share_to_global, number_of_results)

            # Share result to global node when necessary
            results_after_sharing_step = []
            for share, local_nodes_data in zip(share_to_global, all_local_nodes_data):
                if share:
                    result = self._share_local_node_data(local_nodes_data, command_id)
                else:
                    result = local_nodes_data
                results_after_sharing_step.append(result)

        # SMPC Tables MUST be shared to the global node
        for result in results_after_sharing_step:
            if isinstance(result, LocalNodesSMPCTables):
                raise TypeError("SMPC should only be used when sharing the result.")

        if len(results_after_sharing_step) == 1:
            results_after_sharing_step = results_after_sharing_step[0]

        return results_after_sharing_step

    def _convert_local_udf_results_to_local_nodes_data(
        self, all_nodes_results: List[List[Tuple[LocalNode, NodeData]]]
    ) -> List[LocalNodesData]:
        results = []
        for nodes_result in all_nodes_results:
            # All nodes' results have the same type so only the first_result is needed to define the type
            first_result = nodes_result[0][1]
            if isinstance(first_result, NodeTable):
                results.append(LocalNodesTable(dict(nodes_result)))
            elif isinstance(first_result, NodeSMPCTables):
                # TODO Controller integration with SMPC
                raise NotImplementedError
            else:
                raise NotImplementedError
        return results

    def _share_local_node_data(
        self,
        local_nodes_data: LocalNodesData,
        command_id: int,
    ) -> GlobalNodeData:
        if isinstance(local_nodes_data, LocalNodesTable):
            return self._share_local_table_result_to_global(
                local_node_table=local_nodes_data,
                command_id=command_id,
            )
        elif isinstance(local_nodes_data, LocalNodesSMPCTables):
            return self._share_local_smpc_tables_result_to_global(
                local_nodes_data, command_id
            )

        raise NotImplementedError

    def _share_local_table_result_to_global(
        self,
        local_node_table: LocalNodesTable,
        command_id: int,
    ) -> GlobalNodeTable:
        nodes_tables = local_node_table.nodes_tables

        # check the tables have the same schema
        common_schema = self._validate_same_schema_tables(nodes_tables)

        # create remote tables on global node
        table_names = []
        for node, node_table in nodes_tables.items():
            self._global_node.create_remote_table(
                table_name=node_table.full_table_name,
                table_schema=common_schema,
                native_node=node,
            )
            table_names.append(node_table.full_table_name)

        # merge remote tables into one merge table on global node
        merge_table = self._global_node.create_merge_table(str(command_id), table_names)

        return GlobalNodeTable(node=self._global_node, table=merge_table)

    def _share_local_smpc_tables_result_to_global(
        self,
        nodes_smpc_tables: LocalNodesSMPCTables,
        command_id: int,
    ) -> GlobalNodeSMPCTables:
        # TODO Controller integration with SMPC
        raise NotImplementedError

    def run_udf_on_global_node(
        self,
        func_name: str,
        positional_args: Optional[List[Any]] = None,
        keyword_args: Optional[Dict[str, Any]] = None,
        share_to_locals: Union[None, bool, List[bool]] = None,
    ) -> Union[AlgoExecData, List[AlgoExecData]]:
        # 1. check positional_args and keyword_args tables do not contain _LocalNodeTable(s)
        # 2. queue run_udf on the global node
        # 3. wait for it to complete
        # 4. a(or multiple) new table(s) was generated on global node
        # 5. queue create_remote_table on each of the local nodes to share the generated table

        command_id = get_next_command_id()

        self._validate_global_run_udf_args(
            positional_args=positional_args,
            keyword_args=keyword_args,
        )

        positional_udf_args = algoexec_udf_posargs_to_node_udf_posargs(positional_args)
        keyword_udf_args = algoexec_udf_kwargs_to_node_udf_kwargs(keyword_args)

        # Queue the udf on global node
        task = self._global_node.queue_run_udf(
            command_id=str(command_id),
            func_name=func_name,
            positional_args=positional_udf_args,
            keyword_args=keyword_udf_args,
        )

        node_tables = self._global_node.get_queued_udf_result(task)
        global_node_tables = self._convert_global_udf_results_to_global_node_data(
            node_tables
        )

        results_after_sharing_step = global_node_tables
        if share_to_locals is not None:
            # validate and transform share_to_locals variable
            if not isinstance(share_to_locals, list):
                share_to_locals = [share_to_locals]
            number_of_results = len(global_node_tables)
            self._validate_share_to(share_to_locals, number_of_results)

            # Share result to local nodes when necessary
            results_after_sharing_step = []
            for share, table in zip(share_to_locals, global_node_tables):
                if share:
                    results_after_sharing_step.append(
                        self._share_global_table_to_locals(table)
                    )
                else:
                    results_after_sharing_step.append(table)

        if len(results_after_sharing_step) == 1:
            results_after_sharing_step = results_after_sharing_step[0]

        return results_after_sharing_step

    def _convert_global_udf_results_to_global_node_data(
        self,
        node_tables: List[NodeTable],
    ) -> List[GlobalNodeTable]:
        global_tables = [
            GlobalNodeTable(
                node=self._global_node,
                table=node_table,
            )
            for node_table in node_tables
        ]
        return global_tables

    def _share_global_table_to_locals(
        self, global_table: GlobalNodeTable
    ) -> LocalNodesTable:

        local_tables = []
        table_schema = self._global_node.get_table_schema(global_table.table)
        for node in self._local_nodes:
            node.create_remote_table(
                table_name=global_table.table.full_table_name,
                table_schema=table_schema,
                native_node=self._global_node,
            )
            local_tables.append((node, global_table.table))

        return LocalNodesTable(nodes_tables=dict(local_tables))

    # TABLES functionality
    def get_table_data(self, node_table) -> TableData:
        return node_table.get_table_data()

    def get_table_schema(self, node_table) -> TableSchema:
        return node_table.get_table_schema()

    # -------------helper methods------------
    def _validate_local_run_udf_args(
        self,
        positional_args: Optional[List[Any]] = None,
        keyword_args: Optional[Dict[str, Any]] = None,
    ):
        if self._type_exists_in_udf_args(GlobalNodeTable):
            raise TypeError(
                f"run_udf_on_local_nodes contains a 'GlobalNodeTable' type"
                f"in the arguments which is not acceptable. "
                f"{positional_args=} \n {keyword_args=}"
            )

    def _validate_global_run_udf_args(
        self,
        positional_args: Optional[List[Any]] = None,
        keyword_args: Optional[Dict[str, Any]] = None,
    ):
        if self._type_exists_in_udf_args(LocalNodesTable):
            raise TypeError(
                f"run_udf_on_global_node contains a 'LocalNodesTable' type"
                f"in the arguments which is not acceptable. "
                f"{positional_args=} \n {keyword_args=}"
            )

    def _type_exists_in_udf_args(
        self,
        input_type: type,
        positional_args: Optional[List[Any]] = None,
        keyword_args: Optional[Dict[str, Any]] = None,
    ):
        for arg in positional_args or []:
            if isinstance(arg, input_type):
                return True
        if keyword_args:
            for arg in keyword_args.values():
                if isinstance(arg, input_type):
                    return True

    def _get_local_run_udfs_results(
        self, tasks: Dict[LocalNode, IQueuedUDFAsyncResult]
    ) -> List[List[Tuple[LocalNode, NodeData]]]:
        all_nodes_results = {}
        for node, task in tasks.items():
            node_results = node.get_queued_udf_result(task)
            for index, node_result in enumerate(node_results):
                if index not in all_nodes_results:
                    all_nodes_results[index] = []
                all_nodes_results[index].append((node, node_result))

        # Validate that all nodes should have the same number of results from a udf
        if not all(
            len(nodes_result) == len(all_nodes_results[0])
            for nodes_result in all_nodes_results.values()
        ):
            raise InconsistentUDFResultSizeException(all_nodes_results)

        # Validate that all nodes have the same result type
        for nodes_result in all_nodes_results.values():
            if not all(isinstance(r, type(nodes_result[0])) for r in nodes_result[1:]):
                raise TypeError(
                    f"The NODEs returned results of different type. Results: {nodes_result}"
                )

        all_nodes_results = list(all_nodes_results.values())

        return all_nodes_results

    def _validate_share_to(self, share_to: Union[bool, List[bool]], number_of_results):
        for elem in share_to:
            if not isinstance(elem, bool):
                raise Exception(
                    f"share_to_locals must be of type bool or List[bool] but "
                    f"{type(share_to)=} was passed"
                )
        if len(share_to) != number_of_results:
            raise InconsistentShareTablesValueException(share_to, number_of_results)

    def _validate_same_schema_tables(
        self, tables: Dict[LocalNode, NodeTable]
    ) -> TableSchema:
        """
        Returns : TableSchema the common TableSchema, if all tables have the same schema
        """
        have_common_schema = True
        reference_schema = None
        schemas = {}
        for node, table in tables.items():
            schemas[table] = node.get_table_schema(table)
            if reference_schema:
                if schemas[table] != reference_schema:
                    have_common_schema = False
            else:
                reference_schema = schemas[table]
        if not have_common_schema:
            raise InconsistentTableSchemasException(schemas)
        return reference_schema


class _SingleLocalNodeAlgorithmExecutionInterface(_AlgorithmExecutionInterface):
    def __init__(self, algo_execution_interface_dto: _AlgorithmExecutionInterfaceDTO):
        super().__init__(algo_execution_interface_dto)
        self._global_node = self._local_nodes[0]

    def _share_local_node_data(
        self,
        local_nodes_data: LocalNodesData,
        command_id: int,
    ) -> GlobalNodeData:
        if isinstance(local_nodes_data, LocalNodesTable):
            return GlobalNodeTable(
                node=self._global_node,
                table=local_nodes_data.nodes_tables[self._local_nodes[0]],
            )
        elif isinstance(local_nodes_data, LocalNodesSMPCTables):
            # TODO Controller integration with SMPC
            raise NotImplementedError

        raise NotImplementedError

    def _share_global_table_to_locals(
        self, global_table: GlobalNodeTable
    ) -> LocalNodesTable:
        return LocalNodesTable(
            nodes_tables=dict({self._global_node: global_table.table})
        )


def get_next_command_id() -> int:
    if hasattr(get_next_command_id, "index"):
        get_next_command_id.index += 1
    else:
        get_next_command_id.index = 0
    return get_next_command_id.index
