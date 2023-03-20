from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from pydantic import BaseModel

from mipengine.controller import controller_logger as ctrl_logger
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
from mipengine.controller.nodes import GlobalNode
from mipengine.controller.nodes import LocalNode
from mipengine.controller.smpc_helper import get_smpc_results
from mipengine.controller.smpc_helper import load_data_to_smpc_clients
from mipengine.controller.smpc_helper import trigger_smpc_operations
from mipengine.controller.smpc_helper import wait_for_smpc_results_to_be_ready
from mipengine.node_tasks_DTOs import NodeSMPCDTO
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import NodeUDFDTO
from mipengine.node_tasks_DTOs import SMPCTablesInfo
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.udfgen import make_unique_func_name


class Nodes(BaseModel):
    global_node: Optional[GlobalNode]
    local_nodes: List[LocalNode]

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False


class CommandIdGenerator:
    def __init__(self):
        self._index = 0

    def get_next_command_id(self) -> int:
        current = self._index
        self._index += 1
        return str(current)


class AsyncResult:
    def get(self, timeout=None):
        pass


class InconsistentTableSchemasException(Exception):
    def __init__(self, table_infos: List[TableInfo]):
        message = f"Table_infos: {table_infos} do not have a common schema."
        super().__init__(message)


class InconsistentUDFResultSizeException(Exception):
    def __init__(self, result_tables: Dict[int, List[Tuple[LocalNode, TableInfo]]]):
        message = (
            f"The following udf execution results on multiple nodes should have "
            f"the same number of results.\nResults:{result_tables}"
        )
        super().__init__(message)


class InconsistentShareTablesValueException(Exception):
    def __init__(self, share_list: Sequence[bool], number_of_result_tables: int):
        message = f"The size of the {share_list=} does not match the {number_of_result_tables=}"
        super().__init__(message)


class InitializationParams(BaseModel):
    smpc_enabled: bool
    smpc_optional: bool
    request_id: str
    algo_flags: Optional[Dict[str, Any]] = None
    # data_model_views: List[LocalNodesTable]

    class Config:
        arbitrary_types_allowed = True


class AlgorithmExecutionEngine:
    def __init__(
        self,
        initialization_params: InitializationParams,
        command_id_generator: CommandIdGenerator,
        nodes: Nodes,
    ):
        self._logger = ctrl_logger.get_request_logger(
            request_id=initialization_params.request_id
        )
        self._algorithm_execution_flags = initialization_params.algo_flags
        # self._data_model_views = initialization_params.data_model_views
        self._smpc_enabled = initialization_params.smpc_enabled
        self._smpc_optional = initialization_params.smpc_optional

        self._command_id_generator = command_id_generator
        self._local_nodes = nodes.local_nodes
        self._global_node = nodes.global_node

    # @property
    # def data_model_views(self):
    #     return self._data_model_views

    @property
    def use_smpc(self):
        return self._get_use_smpc_flag()

    # UDFs functionality
    def run_udf_on_local_nodes(
        self,
        func: Callable,
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

        func_name = make_unique_func_name(func)
        command_id = self._command_id_generator.get_next_command_id()

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

        # validate length of share_to_global
        number_of_results = len(all_local_nodes_data)
        self._validate_share_to(share_to_global, number_of_results)

        # Share result to global node when necessary
        results_after_sharing_step = [
            self._share_local_node_data(
                local_nodes_data, self._command_id_generator.get_next_command_id()
            )
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

    def run_udf_on_global_node(
        self,
        func: Callable,
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

        func_name = make_unique_func_name(func)
        command_id = self._command_id_generator.get_next_command_id()

        self._validate_global_run_udf_args(
            positional_args=positional_args,
            keyword_args=keyword_args,
        )

        positional_udf_args = algoexec_udf_posargs_to_node_udf_posargs(positional_args)
        keyword_udf_args = algoexec_udf_kwargs_to_node_udf_kwargs(keyword_args)

        if isinstance(share_to_locals, bool):
            share_to_locals = (share_to_locals,)

        if output_schema and len(share_to_locals) != 1:
            raise NotImplementedError(
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

    def _get_use_smpc_flag(self) -> bool:
        """
        SMPC usage is initially defined from the config file.

        If the smpc flag exists in the request and smpc usage is optional,
        then it's defined from the request.
        """
        flags = self._algorithm_execution_flags

        use_smpc = self._smpc_enabled
        if self._smpc_optional and flags and USE_SMPC_FLAG in flags.keys():
            use_smpc = flags[USE_SMPC_FLAG]

        return use_smpc

    def _convert_global_udf_results_to_global_node_data(
        self,
        node_tables: List[NodeTableDTO],
    ) -> List[GlobalNodeTable]:
        global_tables = [
            GlobalNodeTable(
                node=self._global_node,
                table_info=table_dto.value,
            )
            for table_dto in node_tables
        ]
        return global_tables

    def _share_global_table_to_locals(
        self, global_table: GlobalNodeTable
    ) -> LocalNodesTable:
        local_tables = {
            node: node.create_remote_table(
                table_name=global_table.table_info.name,
                table_schema=global_table.table_info.schema_,
                native_node=self._global_node,
            )
            for node in self._local_nodes
        }
        return LocalNodesTable(nodes_tables_info=local_tables)

    # TABLES functionality
    def get_table_data(self, node_table) -> TableData:
        return node_table.get_table_data()

    def get_table_schema(self, node_table) -> TableSchema:
        return node_table.get_table_schema()

    def _convert_local_udf_results_to_local_nodes_data(
        self, all_nodes_results: List[List[Tuple[LocalNode, NodeUDFDTO]]]
    ) -> List[LocalNodesData]:
        results = []
        for nodes_result in all_nodes_results:
            # All nodes' results have the same type so only the first_result is needed
            # to define the type
            first_result = nodes_result[0][1]
            if isinstance(first_result, NodeTableDTO):
                results.append(
                    LocalNodesTable(
                        {node: node_res.value for node, node_res in nodes_result}
                    )
                )
            elif isinstance(first_result, NodeSMPCDTO):
                results.append(
                    LocalNodesSMPCTables(
                        {node: node_res.value for node, node_res in nodes_result}
                    )
                )
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
        nodes_tables = local_nodes_table.nodes_tables_info

        # check the tables have the same schema
        common_schema = self._validate_same_schema_tables(nodes_tables)

        # create remote tables on global node
        table_infos = [
            self._global_node.create_remote_table(
                table_name=node_table.name,
                table_schema=common_schema,
                native_node=node,
            )
            for node, node_table in nodes_tables.items()
        ]

        # merge remote tables into one merge table on global node
        merge_table = self._global_node.create_merge_table(str(command_id), table_infos)

        return GlobalNodeTable(node=self._global_node, table_info=merge_table)

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
            global_template_table.table_info.name
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
            smpc_tables_info=SMPCTablesInfo(
                template=global_template_table.table_info,
                sum_op=sum_op_result_table,
                min_op=min_op_result_table,
                max_op=max_op_result_table,
            ),
        )

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
    ) -> List[List[Tuple[LocalNode, NodeUDFDTO]]]:
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
            raise TypeError(
                f"share_to_locals must be of type Sequence[bool] but "
                f"{type(share_to)=} was passed"
            )
        if len(share_to) != number_of_results:
            raise InconsistentShareTablesValueException(share_to, number_of_results)

    def _validate_same_schema_tables(
        self, table_info_per_node: Dict[LocalNode, TableInfo]
    ) -> TableSchema:
        """
        Returns : TableSchema the common TableSchema, if all tables have the same schema
        """
        reference_schema = next(
            iter(table_info_per_node.values())
        ).schema_  # Use the first table schema as reference
        for _, table_info in table_info_per_node.items():
            if table_info.schema_ != reference_schema:
                raise InconsistentTableSchemasException(
                    list(table_info_per_node.values())
                )

        return reference_schema


class AlgorithmExecutionEngineSingleLocalNode(AlgorithmExecutionEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._global_node = self._local_nodes[0]

    def _share_local_node_data(
        self,
        local_nodes_data: LocalNodesData,
        command_id: int,
    ) -> GlobalNodeData:
        if isinstance(local_nodes_data, LocalNodesTable):
            return GlobalNodeTable(
                node=self._global_node,
                table_info=local_nodes_data.nodes_tables_info[self._local_nodes[0]],
            )
        elif isinstance(local_nodes_data, LocalNodesSMPCTables):
            return GlobalNodeSMPCTables(
                node=self._global_node,
                smpc_tables_info=local_nodes_data.nodes_smpc_tables[self._global_node],
            )
        raise NotImplementedError

    def _share_global_table_to_locals(
        self, global_table: GlobalNodeTable
    ) -> LocalNodesTable:
        return LocalNodesTable(
            nodes_tables_info=dict({self._global_node: global_table.table_info})
        )
