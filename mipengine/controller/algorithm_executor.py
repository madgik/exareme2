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

from mipengine import import_algorithm_modules
from mipengine.controller import controller_logger as ctrl_logger
from mipengine.controller.algorithm_execution_DTOs import AlgorithmExecutionDTO
from mipengine.controller.algorithm_execution_DTOs import NodesTasksHandlersDTO
from mipengine.controller.node_tasks_handler_interface import INodeTasksHandler

from mipengine.controller.node_tasks_handler_interface import IQueuedUDFAsyncResult
from mipengine.controller.node_tasks_handler_celery import ClosedBrokerConnectionError
from mipengine.controller import controller_logger as ctrl_logger

from mipengine.controller.algorithm_executor_helpers import _INode
from mipengine.controller.algorithm_executor_helpers import TableName

from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFArgument
from mipengine.node_tasks_DTOs import UDFArgumentKind


algorithm_modules = import_algorithm_modules()


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
    def __init__(self, tables_schemas: Dict["_INodeTable", TableSchema]):
        message = f"Tables: {tables_schemas} do not have a common schema"
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
    def initial_view_tables(self):
        return self._initial_view_tables

    def _create_initial_view_tables(self, initial_view_tables_params):
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
    def node_address(self):
        return self._node_tasks_handler.node_data_address

    # TABLES functionality

    def get_tables(self) -> List[TableName]:
        return self._node_tasks_handler.get_tables(context_id=self.context_id)

    def get_table_schema(self, table_name: TableName):
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

    def create_merge_table(self, command_id: str, table_names: List[TableName]):
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
        self, command_id: str, func_name: str, positional_args, keyword_args
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

    # CLEANUP functionality

    def clean_up(self):
        self._node_tasks_handler.clean_up(context_id=self.context_id)


class _AlgorithmExecutionInterfaceDTO(BaseModel):
    global_node: _Node
    local_nodes: List[_Node]
    algorithm_name: str
    algorithm_parameters: Optional[Dict[str, List[str]]] = None
    x_variables: Optional[List[str]] = None
    y_variables: Optional[List[str]] = None

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
    def initial_view_tables(self):
        return self._initial_view_tables

    @property
    def algorithm_parameters(self):
        return self._algorithm_parameters

    @property
    def x_variables(self):
        return self._x_variables

    @property
    def y_variables(self):
        return self._y_variables

    # UDFs functionality
    def run_udf_on_local_nodes(
        self,
        func_name: str,
        positional_args: Optional[List["_LocalNodeTable"]] = (),
        keyword_args: Optional[Dict[str, "_LocalNodeTable"]] = {},
        share_to_global: Union[bool, List[bool]] = None,
    ) -> List["_INodeTable"]:
        # 1. queues run_udf task on all local nodes
        # 2. waits for all nodes to complete the tasks execution
        # 3. one(or multiple) new table(s) per local node was generated
        # 4. queues create_remote_table on global for each of the generated tables
        # 4. create merge table on global node to merge the remote tables

        command_id = get_next_command_id()

        # Queue the udf on all local nodes
        tasks = {}
        for node in self._local_nodes:
            positional_udf_args = self._transform_run_udf_posargs(positional_args, node)
            keyword_udf_args = self._transform_run_udf_kwargs(keyword_args, node)

            task = node.queue_run_udf(
                command_id=command_id,
                func_name=func_name,
                positional_args=positional_udf_args,
                keyword_args=keyword_udf_args,
            )
            tasks[node] = task

        # Get udf results from each local node
        all_nodes_result_tables: Dict[int, [(Node, TableName)]] = {}
        for node, task in tasks.items():
            node_result_tables: List[TableName] = node.get_queued_udf_result(task)

            for index, task_result in enumerate(node_result_tables):
                if index not in all_nodes_result_tables:
                    all_nodes_result_tables[index] = []
                all_nodes_result_tables[index].append((node, task_result))

        # Transform share_to_global variable
        if share_to_global != None and not isinstance(share_to_global, list):
            share_to_global = [share_to_global for _ in range(len(node_result_tables))]

        # Handle sharing results to global node
        results_after_sharing_step = []
        if share_to_global:
            for index, share in enumerate(share_to_global):
                nodes_tables: List[Tuple["_Node", TableName]] = all_nodes_result_tables[
                    index
                ]
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
                        # merge remote tables into one merge table on global
                        merge_table = self._global_node.create_merge_table(
                            command_id, tables
                        )

                    results_after_sharing_step.append(
                        _GlobalNodeTable(node=self._global_node, table_name=merge_table)
                    )
                else:
                    # package it to _LocalNodeTable and append it
                    results_after_sharing_step.append(
                        _LocalNodeTable(nodes_tables=dict(nodes_tables))
                    )
        else:
            # package all tables to _LocalNodeTable and return it
            for index, node_tables in all_nodes_result_tables.items():
                results_after_sharing_step.append(
                    _LocalNodeTable(nodes_tables=dict(node_tables))
                )

        # backward compatibility.. TODO always return list??
        if len(results_after_sharing_step) == 1:
            results_after_sharing_step = results_after_sharing_step[0]
        return results_after_sharing_step

    def run_udf_on_global_node(
        self,
        func_name: str,
        positional_args: Optional[List["_GlobalNodeTable"]] = (),
        keyword_args: Optional[Dict[str, "_GlobalNodeTable"]] = {},
        share_to_locals: Union[bool, List[bool]] = None,
    ) -> List["_INodeTable"]:
        # 1. check the input tables are of type _GlobalNodeTable
        # 2. queue run_udf on the global node
        # 3. wait for it to complete
        # 4. a(or multiple) new table(s) was generated on global node
        # 5. queue create_remote_table on each of the local nodes for the generated table

        command_id = get_next_command_id()

        positional_udf_args = self._transform_run_udf_posargs(positional_args)
        keyword_udf_args = self._transform_run_udf_kwargs(keyword_args)

        # Queue the udf on global node
        task = self._global_node.queue_run_udf(
            command_id=command_id,
            func_name=func_name,
            positional_args=positional_udf_args,
            keyword_args=keyword_udf_args,
        )

        # Get udf result from global node
        result_tables = self._global_node.get_queued_udf_result(task)

        # Transform share_to_locals variable
        if share_to_locals != None and not isinstance(share_to_locals, list):
            share_to_locals = [share_to_locals for _ in range(len(result_tables))]

        # Handle sharing result to local nodes
        final_results = []
        if share_to_locals:
            # TODO: check the provided share_to_.. appropriate size
            for index, share in enumerate(share_to_locals):
                if share:
                    if self._is_single_node_execution():
                        # breakpoint()
                        local_tables = [(self._global_node, result_tables[0])]
                    else:
                        local_tables = []
                        table_schema = self._global_node.get_table_schema(
                            result_tables[index]
                        )
                        for node in self._local_nodes:
                            node.create_remote_table(
                                table_name=result_tables[index]._full_name,
                                table_schema=table_schema,
                                native_node=self._global_node,
                            )
                            local_tables.append((node, result_tables[index]))
                    # breakpoint()
                    final_results.append(
                        _LocalNodeTable(nodes_tables=dict(local_tables))
                    )

                else:
                    final_results.append(
                        _GlobalNodeTable(
                            node=self._global_node, table_name=result_tables[index]
                        )
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
        if isinstance(node_table, _LocalNodeTable) or isinstance(
            node_table, _GlobalNodeTable
        ):
            return node_table.get_table_schema()
        else:  # TODO specific exception
            raise Exception(
                f"(AlgorithmExecutionInterface::get_table_schema) node_table type-> {type(node_table)} not acceptable"
            )

    def _transform_run_udf_kwargs(
        self, keyword_args: List["_LocalNodeTable"], node: "_Node" = None
    ):
        keyword_args_transformed = {}
        for var_name, val in keyword_args.items():
            if isinstance(val, _INodeTable):
                if isinstance(val, _LocalNodeTable):
                    table_name = val.nodes_tables[node].full_table_name
                elif isinstance(val, _GlobalNodeTable):
                    table_name = val.table_name.full_table_name
                else:
                    raise Exception(f"_transform_run_udf_posargs received {type(val)=}")
                udf_argument = UDFArgument(kind=UDFArgumentKind.TABLE, value=table_name)
                # elif isinstance(val, _GlobalNodeTable):
                #     raise Exception(
                #         f"(run_udf_on_local_nodes) GlobalNodeTable types are not "
                #         f"accepted from run_udf_on_local_nodes"
                #     )
            else:
                udf_argument = UDFArgument(kind=UDFArgumentKind.LITERAL, value=val)
            keyword_args_transformed[var_name] = udf_argument.json()
        return keyword_args_transformed

    def _transform_run_udf_posargs(
        self, positional_args: List["_INodeTable"], node: "_Node" = None
    ):
        positional_args_transfrormed = []
        for val in positional_args:
            if isinstance(val, _INodeTable):
                if isinstance(val, _LocalNodeTable):
                    table_name = val.nodes_tables[node].full_table_name
                elif isinstance(val, _GlobalNodeTable):
                    # breakpoint()
                    table_name = val.table_name.full_table_name
                else:
                    raise Exception(f"_transform_run_udf_posargs received {type(val)=}")
                udf_argument = UDFArgument(
                    kind=UDFArgumentKind.TABLE,
                    value=table_name,
                )
            # elif isinstance(val, _GlobalNodeTable):
            #     raise Exception(
            #         f"(run_udf_on_local_nodes) GlobalNodeTable types are not "
            #         f"accepted from run_udf_on_local_nodes"
            #     )
            else:
                udf_argument = UDFArgument(kind=UDFArgumentKind.LITERAL, value=val)
            positional_args_transfrormed.append(udf_argument.json())
        return positional_args_transfrormed


class _INodeTable(ABC):
    # TODO: better abstraction here...
    pass


class _LocalNodeTable(_INodeTable):
    def __init__(self, nodes_tables: Dict["_Node", "TableName"]):
        self._nodes_tables = nodes_tables

        if not self._validate_matching_table_names(list(self._nodes_tables.values())):
            raise self.MismatchingTableNamesException(
                [table_name.full_table_name for table_name in nodes_tables.values()]
            )

    @property
    def nodes_tables(self):
        return self._nodes_tables

    # TODO this is redundant, either remove it or overload all node methods here?
    def get_table_schema(self) -> TableSchema:
        node = list(self.nodes_tables.keys())[0]
        table = self.nodes_tables[node]
        return node.get_table_schema(table)

    def get_table_data(self) -> TableData:
        tables_data = []
        for node, table_name in self.nodes_tables.items():
            tables_data.append(node.get_table_data(table_name))
        tables_data_flat = [table_data.data_ for table_data in tables_data]
        tables_data_flat = [
            k for i in tables_data_flat for j in i for k in j
        ]  # TODO bejesus..
        return tables_data_flat

    def __repr__(self):
        r = f"LocalNodeTable: {self.get_table_schema()}\n"
        for node, table_name in self.nodes_tables.items():
            r += f"\t{node=} {table_name=}\n"
        return r

    def _validate_matching_table_names(self, table_names: List[TableName]):
        table_name_without_node_id = table_names[0].without_node_id()
        for table_name in table_names:
            if table_name.without_node_id() != table_name_without_node_id:
                return False
        return True

    class MismatchingTableNamesException(Exception):
        def __init__(self, table_names):
            self.message = f"Mismatched table names ->{table_names}"


class _GlobalNodeTable(_INodeTable):
    def __init__(self, node: "_Node", table_name: "TableName"):
        self._node = node
        self._table_name = table_name

    @property
    def node(self):
        return self._node

    @property
    def table_name(self):
        return self._table_name

    # TODO this is redundant, either remove it or overload all node methods here?
    def get_table_schema(self) -> TableSchema:
        table_schema: TableSchema = self.node.get_table_schema(self.table_name).columns
        return table_schema

    def get_table_data(self) -> TableData:
        table_data = self.node.get_table_data(self.table_name).data_
        return table_data

    def __repr__(self):
        r = f"GlobalNodeTable: \n\tschema={self.get_table_schema()}\n \t{self.table_name=}\n"
        return r


def check_same_schema_tables(
    tables: List[Tuple[_INode, TableName]]
) -> (bool, Dict[TableName, TableSchema], TableSchema):
    """
    Returns :
    First part of the returning tuple is True if all tables have the same schema.
    Second part f the tuple is dictionary with keys:table names and vals:the
    corresponding table schema
    Third is the common TableSchema, if all tables have the same schema, else None
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


# NOTE tried to turn this into a generator, the problem is there are multiple consumers
# so the generator should be singleton in some way, the solutions were more complicated
# than this simple implementation
def get_next_command_id():
    if hasattr(get_next_command_id, "index"):
        get_next_command_id.index += 1
    else:
        get_next_command_id.index = 0
    return get_next_command_id.index
