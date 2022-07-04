import traceback
from logging import Logger
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from pydantic import BaseModel

from mipengine import algorithm_modules
from mipengine.controller import config as ctrl_config
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.algorithm_execution_DTOs import AlgorithmExecutionDTO
from mipengine.controller.algorithm_execution_DTOs import NodesTasksHandlersDTO
from mipengine.controller.algorithm_executor_node_data_objects import NodeData
from mipengine.controller.algorithm_executor_node_data_objects import SMPCTableNames
from mipengine.controller.algorithm_executor_node_data_objects import TableName
from mipengine.controller.algorithm_executor_nodes import GlobalNode
from mipengine.controller.algorithm_executor_nodes import LocalNode
from mipengine.controller.algorithm_executor_smpc_helper import get_smpc_results
from mipengine.controller.algorithm_executor_smpc_helper import (
    load_data_to_smpc_clients,
)
from mipengine.controller.algorithm_executor_smpc_helper import trigger_smpc_operations
from mipengine.controller.algorithm_executor_smpc_helper import (
    wait_for_smpc_results_to_be_ready,
)
from mipengine.controller.algorithm_flow_data_objects import AlgoFlowData
from mipengine.controller.algorithm_flow_data_objects import GlobalNodeData
from mipengine.controller.algorithm_flow_data_objects import GlobalNodeSMPCTables
from mipengine.controller.algorithm_flow_data_objects import GlobalNodeTable
from mipengine.controller.algorithm_flow_data_objects import LocalNodesData
from mipengine.controller.algorithm_flow_data_objects import LocalNodesSMPCTables
from mipengine.controller.algorithm_flow_data_objects import LocalNodesTable
from mipengine.controller.algorithm_flow_data_objects import (
    algoexec_udf_kwargs_to_node_udf_kwargs,
)
from mipengine.controller.algorithm_flow_data_objects import (
    algoexec_udf_posargs_to_node_udf_posargs,
)
from mipengine.controller.api.algorithm_request_dto import USE_SMPC_FLAG
from mipengine.controller.celery_app import CeleryConnectionError
from mipengine.controller.celery_app import CeleryTaskTimeoutException
from mipengine.node_tasks_DTOs import CommonDataElement
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.udfgen import TensorBinaryOp
from mipengine.udfgen import make_unique_func_name


class AsyncResult:
    def get(self, timeout=None):
        pass


class AlgorithmExecutionException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class NodeUnresponsiveAlgorithmExecutionException(Exception):
    def __init__(self):
        message = (
            "One of the nodes participating in the algorithm execution "
            "stopped responding"
        )
        super().__init__(message)
        self.message = message


class InconsistentTableSchemasException(Exception):
    def __init__(self, tables_schemas: Dict[TableName, TableSchema]):
        message = f"Tables: {tables_schemas} do not have a common schema"
        super().__init__(message)


class InconsistentUDFResultSizeException(Exception):
    def __init__(self, result_tables: Dict[int, List[Tuple[LocalNode, TableName]]]):
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
        common_data_elements: Dict[str, CommonDataElement],
    ):
        self._logger = ctrl_logger.get_request_logger(
            request_id=algorithm_execution_dto.request_id
        )
        self._nodes_tasks_handlers_dto = nodes_tasks_handlers_dto
        self._algorithm_execution_dto = algorithm_execution_dto

        self._request_id = algorithm_execution_dto.request_id
        self._context_id = algorithm_execution_dto.context_id
        self._algorithm_name = algorithm_execution_dto.algorithm_name
        self._common_data_elements = common_data_elements
        self._algorithm_flow_module = None
        self._execution_interface = None

        self._global_node: GlobalNode = GlobalNode(
            request_id=self._request_id,
            context_id=self._context_id,
            node_tasks_handler=self._nodes_tasks_handlers_dto.global_node_tasks_handler,
        )
        self._local_nodes: List[LocalNode] = [
            LocalNode(
                request_id=self._request_id,
                context_id=self._context_id,
                node_tasks_handler=node_tasks_handler,
            )
            for node_tasks_handler in self._nodes_tasks_handlers_dto.local_nodes_tasks_handlers
        ]

    def _get_use_smpc_flag(self) -> bool:
        """
        SMPC usage is initially defined from the config file.

        If the smpc flag exists in the request and smpc usage is optional,
        then it's defined from the request.
        """
        flags = self._algorithm_execution_dto.algo_flags

        use_smpc = ctrl_config.smpc.enabled
        if ctrl_config.smpc.optional and flags and USE_SMPC_FLAG in flags.keys():
            use_smpc = flags[USE_SMPC_FLAG]

        return use_smpc

    def _instantiate_algorithm_execution_interface(self):
        algo_execution_interface_dto = _AlgorithmExecutionInterfaceDTO(
            global_node=self._global_node,
            local_nodes=self._local_nodes,
            algorithm_name=self._algorithm_name,
            algorithm_parameters=self._algorithm_execution_dto.algo_parameters,
            x_variables=self._algorithm_execution_dto.x_vars,
            y_variables=self._algorithm_execution_dto.y_vars,
            var_filters=self._algorithm_execution_dto.var_filters,
            data_model=self._algorithm_execution_dto.data_model,
            datasets_per_local_node=self._algorithm_execution_dto.datasets_per_local_node,
            use_smpc=self._get_use_smpc_flag(),
            logger=self._logger,
        )
        if len(self._local_nodes) > 1:
            self._execution_interface = _AlgorithmExecutionInterface(
                algo_execution_interface_dto, self._common_data_elements
            )
        else:
            self._execution_interface = _SingleLocalNodeAlgorithmExecutionInterface(
                algo_execution_interface_dto, self._common_data_elements
            )

        # Get algorithm module
        self._algorithm_flow_module = algorithm_modules[self._algorithm_name]

    def run(self):
        try:
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
        except (CeleryConnectionError, CeleryTaskTimeoutException) as exc:
            self._logger.error(f"ErrorType: '{type(exc)}' and message: '{exc}'")
            raise NodeUnresponsiveAlgorithmExecutionException()
        except Exception as exc:
            self._logger.error(traceback.format_exc())
            raise exc


class _AlgorithmExecutionInterfaceDTO(BaseModel):
    global_node: GlobalNode
    local_nodes: List[LocalNode]
    algorithm_name: str
    algorithm_parameters: Optional[Dict[str, Any]] = None
    x_variables: Optional[List[str]] = None
    y_variables: Optional[List[str]] = None
    var_filters: dict = None
    data_model: str
    datasets_per_local_node: Dict[str, List[str]]
    use_smpc: bool
    logger: Logger

    class Config:
        arbitrary_types_allowed = True


class _AlgorithmExecutionInterface:
    def __init__(
        self,
        algo_execution_interface_dto: _AlgorithmExecutionInterfaceDTO,
        common_data_elements: Dict[str, CommonDataElement],
    ):
        self._global_node: GlobalNode = algo_execution_interface_dto.global_node
        self._local_nodes: List[LocalNode] = algo_execution_interface_dto.local_nodes
        self._algorithm_name = algo_execution_interface_dto.algorithm_name
        self._algorithm_parameters = algo_execution_interface_dto.algorithm_parameters
        self._x_variables = algo_execution_interface_dto.x_variables
        self._y_variables = algo_execution_interface_dto.y_variables
        self._var_filters = algo_execution_interface_dto.var_filters
        self._data_model = algo_execution_interface_dto.data_model
        self._datasets_per_local_node = (
            algo_execution_interface_dto.datasets_per_local_node
        )
        self._use_smpc = algo_execution_interface_dto.use_smpc
        varnames = (self._x_variables or []) + (self._y_variables or [])
        self._metadata = {
            varname: cde.dict()
            for varname, cde in common_data_elements.items()
            if varname in varnames
        }

        self._logger = algo_execution_interface_dto.logger

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
    def datasets_per_local_node(self):
        return self._datasets_per_local_node

    @property
    def use_smpc(self):
        return self._use_smpc

    def create_primary_data_views(
        self,
        variable_groups: List[List[str]],
        dropna: bool = True,
        check_min_rows: bool = True,
    ) -> List[LocalNodesTable]:
        """
        Creates primary data views, for each variable group provided,
        using also the algorithm request arguments (data_model, datasets, filters).

        Parameters
        ----------
        variable_groups : List[List[str]]
            A list of variable_groups. The variable group is a list of columns.
        dropna : bool
            Remove NAs from the view.
        check_min_rows : bool
            Raise an exception if there are not enough rows in the view.

        Returns
        ------
        List[LocalNodesTable]
            A (LocalNodesTable) view for each variable_group provided.
        """

        command_id = str(get_next_command_id())
        views_per_localnode = [
            (
                local_node,
                local_node.create_data_model_views(
                    command_id=command_id,
                    data_model=self._data_model,
                    datasets=self.datasets_per_local_node[local_node.node_id],
                    columns_per_view=variable_groups,
                    filters=self._var_filters,
                    dropna=dropna,
                    check_min_rows=check_min_rows,
                ),
            )
            for local_node in self._local_nodes
        ]

        return _convert_views_per_localnode_to_local_nodes_tables(views_per_localnode)

    # UDFs functionality
    def run_udf_on_local_nodes(
        self,
        func: Optional[Callable] = None,
        tensor_op: Optional[TensorBinaryOp] = None,
        positional_args: Optional[List[Any]] = None,
        keyword_args: Optional[Dict[str, Any]] = None,
        share_to_global: Union[bool, Sequence[bool]] = False,
        output_schema: Optional[TableSchema] = None,
    ) -> Union[AlgoFlowData, List[AlgoFlowData]]:
        # 1. check positional_args and keyword_args tables do not contain _GlobalNodeTable(s)
        # 2. queues run_udf task on all local nodes
        # 3. waits for all nodes to complete the tasks execution
        # 4. one(or multiple) new table(s) per local node was generated
        # 5. create remote tables on global for each of the generated tables
        # 6. create merge table on global node to merge the remote tables

        func_name = get_func_name(func, tensor_op)
        command_id = get_next_command_id()

        self._validate_local_run_udf_args(
            positional_args=positional_args,
            keyword_args=keyword_args,
        )

        if isinstance(share_to_global, bool):
            share_to_global = (share_to_global,)

        if output_schema and len(share_to_global) != 1:
            raise ValueError(
                "output_schema cannot be used with multiple output UDFs for now."
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
                use_smpc=self.use_smpc,
                output_schema=output_schema,
            )
            tasks[node] = task

        all_nodes_results = self._get_local_run_udfs_results(tasks)
        all_local_nodes_data = self._convert_local_udf_results_to_local_nodes_data(
            all_nodes_results
        )

        results_after_sharing_step = all_local_nodes_data

        # validate length of share_to_global
        number_of_results = len(all_nodes_results)
        self._validate_share_to(share_to_global, number_of_results)

        # Share result to global node when necessary
        results_after_sharing_step = [
            self._share_local_node_data(local_nodes_data, get_next_command_id())
            if share
            else local_nodes_data
            for share, local_nodes_data in zip(share_to_global, all_local_nodes_data)
        ]

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
            if isinstance(first_result, TableName):
                results.append(LocalNodesTable(dict(nodes_result)))
            elif isinstance(first_result, SMPCTableNames):
                results.append(LocalNodesSMPCTables(dict(nodes_result)))
            else:
                raise NotImplementedError
        return results

    def _share_local_node_data(
        self,
        local_nodes_data: LocalNodesData,
        command_id: int,
    ) -> GlobalNodeData:
        if isinstance(local_nodes_data, LocalNodesTable):
            return self._share_local_table_to_global(
                local_nodes_table=local_nodes_data,
                command_id=command_id,
            )
        elif isinstance(local_nodes_data, LocalNodesSMPCTables):
            return self._share_local_smpc_tables_to_global(local_nodes_data, command_id)

        raise NotImplementedError

    def _share_local_table_to_global(
        self,
        local_nodes_table: LocalNodesTable,
        command_id: int,
    ) -> GlobalNodeTable:
        nodes_tables = local_nodes_table.nodes_tables

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

    def _share_local_smpc_tables_to_global(
        self,
        local_nodes_smpc_tables: LocalNodesSMPCTables,
        command_id: int,
    ) -> GlobalNodeSMPCTables:
        global_template_table = self._share_local_table_to_global(
            local_nodes_table=local_nodes_smpc_tables.template_local_nodes_table,
            command_id=command_id,
        )
        self._global_node.validate_smpc_templates_match(
            global_template_table.table.full_table_name
        )

        smpc_clients_per_op = load_data_to_smpc_clients(
            command_id, local_nodes_smpc_tables
        )

        (sum_op, min_op, max_op) = trigger_smpc_operations(
            logger=self._logger,
            context_id=self._global_node.context_id,
            command_id=command_id,
            smpc_clients_per_op=smpc_clients_per_op,
        )

        wait_for_smpc_results_to_be_ready(
            logger=self._logger,
            context_id=self._global_node.context_id,
            command_id=command_id,
            sum_op=sum_op,
            min_op=min_op,
            max_op=max_op,
        )

        (
            sum_op_result_table,
            min_op_result_table,
            max_op_result_table,
        ) = get_smpc_results(
            node=self._global_node,
            context_id=self._global_node.context_id,
            command_id=command_id,
            sum_op=sum_op,
            min_op=min_op,
            max_op=max_op,
        )

        return GlobalNodeSMPCTables(
            node=self._global_node,
            smpc_tables=SMPCTableNames(
                template=global_template_table.table,
                sum_op=sum_op_result_table,
                min_op=min_op_result_table,
                max_op=max_op_result_table,
            ),
        )

    def run_udf_on_global_node(
        self,
        func: Optional[Callable] = None,
        tensor_op: Optional[TensorBinaryOp] = None,
        positional_args: Optional[List[Any]] = None,
        keyword_args: Optional[Dict[str, Any]] = None,
        share_to_locals: Union[bool, Sequence[bool]] = False,
        output_schema: Optional[TableSchema] = None,
    ) -> Union[AlgoFlowData, List[AlgoFlowData]]:
        # 1. check positional_args and keyword_args tables do not contain _LocalNodeTable(s)
        # 2. queue run_udf on the global node
        # 3. wait for it to complete
        # 4. a(or multiple) new table(s) was generated on global node
        # 5. queue create_remote_table on each of the local nodes to share the generated table

        func_name = get_func_name(func, tensor_op)
        command_id = get_next_command_id()

        self._validate_global_run_udf_args(
            positional_args=positional_args,
            keyword_args=keyword_args,
        )

        positional_udf_args = algoexec_udf_posargs_to_node_udf_posargs(positional_args)
        keyword_udf_args = algoexec_udf_kwargs_to_node_udf_kwargs(keyword_args)

        if isinstance(share_to_locals, bool):
            share_to_locals = (share_to_locals,)

        if output_schema and len(share_to_locals) != 1:
            raise ValueError(
                "output_schema cannot be used with multiple output UDFs for now."
            )

        # Queue the udf on global node
        task = self._global_node.queue_run_udf(
            command_id=str(command_id),
            func_name=func_name,
            positional_args=positional_udf_args,
            keyword_args=keyword_udf_args,
            use_smpc=self.use_smpc,
            output_schema=output_schema,
        )

        node_tables = self._global_node.get_queued_udf_result(task)
        global_node_tables = self._convert_global_udf_results_to_global_node_data(
            node_tables
        )

        results_after_sharing_step = global_node_tables

        # validate length of share_to_locals
        number_of_results = len(global_node_tables)
        self._validate_share_to(share_to_locals, number_of_results)

        # Share result to local nodes when necessary
        results_after_sharing_step = [
            self._share_global_table_to_locals(table) if share else table
            for share, table in zip(share_to_locals, global_node_tables)
        ]

        if len(results_after_sharing_step) == 1:
            results_after_sharing_step = results_after_sharing_step[0]

        return results_after_sharing_step

    def _convert_global_udf_results_to_global_node_data(
        self,
        node_tables: List[TableName],
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
        self, tasks: Dict[LocalNode, AsyncResult]
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

    @staticmethod
    def _validate_share_to(share_to: Sequence[bool], number_of_results: int):
        if not all(isinstance(elem, bool) for elem in share_to):
            raise Exception(
                f"share_to_locals must be of type bool or List[bool] but "
                f"{type(share_to)=} was passed"
            )
        if len(share_to) != number_of_results:
            raise InconsistentShareTablesValueException(share_to, number_of_results)

    def _validate_same_schema_tables(
        self, tables: Dict[LocalNode, TableName]
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
    def __init__(
        self,
        algo_execution_interface_dto: _AlgorithmExecutionInterfaceDTO,
        common_data_elements: Dict[str, CommonDataElement],
    ):
        super().__init__(algo_execution_interface_dto, common_data_elements)
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
            return GlobalNodeSMPCTables(
                node=self._global_node,
                smpc_tables=SMPCTableNames(
                    template=local_nodes_data.nodes_smpc_tables[
                        self._global_node
                    ].template,
                    sum_op=local_nodes_data.nodes_smpc_tables[self._global_node].sum_op,
                    min_op=local_nodes_data.nodes_smpc_tables[self._global_node].min_op,
                    max_op=local_nodes_data.nodes_smpc_tables[self._global_node].max_op,
                ),
            )

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


def get_func_name(
    func: Optional[Callable] = None,
    tensor_op: Optional[TensorBinaryOp] = None,
) -> str:
    if func and tensor_op:
        raise ValueError("'func' and 'tensor_op' cannot be used at the same time.")

    if tensor_op:
        return tensor_op.name

    if isinstance(func, str):
        return func

    return make_unique_func_name(func)


def _convert_views_per_localnode_to_local_nodes_tables(
    views_per_localnode: List[Tuple[LocalNode, List[TableName]]]
) -> List[LocalNodesTable]:
    """
    In the views_per_localnode the views are stored per the localnode where they exist.
    In order to create LocalNodesTable objects we need to store them according to the similar "LocalNodesTable"
    they belong to. We group together one view from each node, based on the views' order.

    Parameters
    ----------
    views_per_localnode: List[Tuple[LocalNode, List[TableName]]]
        views grouped per the localnode where they exist.

    Returns
    ------
    List[LocalNodesTable]
        One (LocalNodesTable) view for each one existing in the localnodes.
    """
    views_count = _get_amount_of_localnodes_views(views_per_localnode)

    local_nodes_tables_dicts: List[Dict[LocalNode, TableName]] = [
        {} for _ in range(views_count)
    ]
    for localnode, local_node_views in views_per_localnode:
        for view, local_nodes_tables in zip(local_node_views, local_nodes_tables_dicts):
            local_nodes_tables[localnode] = view
    local_nodes_tables = [
        LocalNodesTable(local_nodes_tables_dict)
        for local_nodes_tables_dict in local_nodes_tables_dicts
    ]
    return local_nodes_tables


def _get_amount_of_localnodes_views(
    views_per_localnode: List[Tuple[LocalNode, List[TableName]]]
) -> int:
    """
    Returns the amount of views after validating all localnodes created the same amount of views.
    """
    views_count = len(views_per_localnode[0][1])
    for local_node, local_node_views in views_per_localnode:
        if len(local_node_views) != views_count:
            raise ValueError(
                f"All views from localnodes should have the same length. "
                f"{local_node} has {len(local_node_views)} instead of {views_count}."
            )
    return views_count
