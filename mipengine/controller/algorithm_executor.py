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

from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFArgument
from mipengine.node_tasks_DTOs import UDFArgumentKind

algorithm_modules = import_algorithm_modules()


class _TableName:
    def __init__(self, table_name):
        self._full_name = table_name
        full_name_split = self._full_name.split("_")
        self._table_type = full_name_split[0]
        self._node_id = full_name_split[1]
        self._context_id = full_name_split[2]
        self._command_id = full_name_split[3]

    @property
    def full_table_name(self):
        return self._full_name

    @property
    def table_type(self):
        return self._table_type

    @property
    def command_id(self):
        return self._command_id

    @property
    def context_id(self):
        return self._context_id

    @property
    def node_id(self):
        return self._node_id

    def without_node_id(self):
        return self._table_type + "_" + self._command_id + "_" + self._context_id

    def __repr__(self):
        return self.full_table_name


class AlgorithmExecutionException(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class AlgorithmExecutor:
    def __init__(
        self,
        algorithm_execution_dto: AlgorithmExecutionDTO,
        nodes_tasks_handlers_dto: NodesTasksHandlersDTO,
    ):
        self._logger=ctrl_logger.get_request_logger(context_id=algorithm_execution_dto.context_id)
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
            self._logger.info(
                f"finished execution of algorithm:{self._algorithm_name}"
            )
            return algorithm_result
        except (
            SoftTimeLimitExceeded,
            TimeLimitExceeded,
            TimeoutError,
            ClosedBrokerConnectionError,
        ) as err:
            error_message = (
                "One of the nodes participating in the algorithm execution "
                "stopped responding"
            )
            self._logger.error(f"{error_message} \n{err=}")
            
            raise AlgorithmExecutionException(error_message)
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
            self._logger.debug(f"cleaning up global_node FAILED {exc=}")
        self._logger.info(f"cleaning up local nodes:{self._local_nodes}")
        for node in self._local_nodes:
            self._logger.info(f"\tcleaning up {node=}")
            try:
                node.clean_up()
            except Exception as exc:
                self._logger.debug(f"cleaning up {node=} FAILED {exc=}")


class _Node:
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
        return f"node_id: {self.node_id}"

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

    def get_tables(self) -> List[_TableName]:
        return self._node_tasks_handler.get_tables(context_id=self.context_id)

    def get_table_schema(self, table_name: _TableName):
        return self._node_tasks_handler.get_table_schema(
            table_name=table_name.full_table_name
        )

    def get_table_data(self, table_name: _TableName) -> TableData:
        return self._node_tasks_handler.get_table_data(table_name.full_table_name)

    def create_table(self, command_id: str, schema: TableSchema) -> _TableName:
        schema_json = schema.json()
        return self._node_tasks_handler.create_table(
            context_id=self.context_id,
            command_id=command_id,
            schema_json=schema_json,
        )

    # VIEWS functionality

    def get_views(self) -> List[_TableName]:
        result = self._node_tasks_handler.get_views(context_id=self.context_id)
        return [_TableName(table_name) for table_name in result]

    # TODO: this is very specific to mip, very inconsistent with the rest, has to
    # be abstracted somehow

    def create_pathology_view(
        self,
        command_id: str,
        pathology: str,
        columns: List[str],
        filters: List[str],
    ) -> _TableName:

        result = self._node_tasks_handler.create_pathology_view(
            context_id=self.context_id,
            command_id=command_id,
            pathology=pathology,
            columns=columns,
            filters=filters,
        )
        return _TableName(result)

    # MERGE TABLES functionality

    def get_merge_tables(self) -> List[_TableName]:
        result = self._node_tasks_handler.get_merge_tables(context_id=self.context_id)
        return [_TableName(table_name) for table_name in result]

    def create_merge_table(self, command_id: str, table_names: List[_TableName]):
        table_names = [table_name.full_table_name for table_name in table_names]
        result = self._node_tasks_handler.create_merge_table(
            context_id=self.context_id,
            command_id=command_id,
            table_names=table_names,
        )
        return _TableName(result)

    # REMOTE TABLES functionality

    def get_remote_tables(self) -> List[str]:
        return self._node_tasks_handler.get_remote_tables(context_id=self.context_id)

    def create_remote_table(
        self, table_name: str, table_schema: TableSchema, native_node: "_Node"
    ) -> _TableName:

        monetdb_socket_addr = native_node.node_address
        return self._node_tasks_handler.create_remote_table(
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

    def get_queued_udf_result(self, async_result: IQueuedUDFAsyncResult) -> List[str]:
        return self._node_tasks_handler.get_queued_udf_result(async_result)

    def get_udfs(self, algorithm_name) -> List[str]:
        return self._node_tasks_handler.get_udfs(algorithm_name)

    # return the generated monetdb pythonudf

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
    ):  # -> List of GlobalNodeTables or LocalNodeTables
        # queue exec_udf task on all local nodes
        # wait for all nodes to complete the tasks execution
        # one new table per local node was generated
        # queue create_remote_table on global for each of the generated tables
        # create merge table on global node to merge the remote tables

        command_id = get_next_command_id()

        tasks = {}
        for node in self._local_nodes:
            # TODO get the nodes from the LocalNodeTables in the positional_args
            positional_udf_args = []
            for val in positional_args:
                if isinstance(val, _LocalNodeTable):
                    udf_argument = UDFArgument(
                        kind=UDFArgumentKind.TABLE,
                        value=val.nodes_tables[node].full_table_name,
                    )
                elif isinstance(val, _GlobalNodeTable):
                    raise Exception(
                        f"(run_udf_on_local_nodes) GlobalNodeTable types are not "
                        f"accepted from run_udf_on_local_nodes"
                    )
                else:
                    udf_argument = UDFArgument(kind=UDFArgumentKind.LITERAL, value=val)
                positional_udf_args.append(udf_argument.json())

            keyword_udf_args = {}
            for var_name, val in keyword_args.items():
                if isinstance(val, _LocalNodeTable):
                    udf_argument = UDFArgument(
                        kind=UDFArgumentKind.TABLE,
                        value=val.nodes_tables[node].full_table_name,
                    )
                elif isinstance(val, _GlobalNodeTable):
                    raise Exception(
                        f"(run_udf_on_local_nodes) GlobalNodeTable types are not "
                        f"accepted from run_udf_on_local_nodes"
                    )
                else:
                    udf_argument = UDFArgument(kind=UDFArgumentKind.LITERAL, value=val)
                keyword_udf_args[var_name] = udf_argument.json()

            task = node.queue_run_udf(
                command_id=command_id,
                func_name=func_name,
                positional_args=positional_udf_args,
                keyword_args=keyword_udf_args,
            )
            tasks[node] = task

        udf_results_tables: List[Dict[_Node, _TableName]] = []
        for node, task in tasks.items():
            task_results = node.get_queued_udf_result(task)
            # Initialize the result tables when we know the number of results
            if len(udf_results_tables) == 0:
                udf_results_tables = []
                for _ in task_results:
                    udf_results_tables.append({})

            for task_result, udf_result_tables in zip(task_results, udf_results_tables):
                udf_result_tables[node] = _TableName(task_result)

            self._share_results_on_global(
                func_name, node, share_to_global, task_results
            )

        final_result_tables = self._create_node_tables_for_global_udf(
            command_id,
            share_to_global,
            udf_results_tables,
        )

        if len(final_result_tables) == 1:
            return final_result_tables[0]
        return final_result_tables

    def _share_results_on_global(
        self,
        func_name: str,
        node: _Node,
        share_results: Union[bool, List[bool]],
        results: List[str],
    ):
        if not share_results:
            return

        if type(share_results) == bool:
            share_results = [share_results]
        if len(share_results) != len(results):
            raise AlgorithmExecutionException(
                f"Method {func_name} has {len(results)} results but 'share_to_global' "
                f"has a length of {len(share_results)}. They should match."
            )

        for share_results, table_name in zip(share_results, results):
            if not share_results:
                continue

            table_schema = node.get_table_schema(_TableName(table_name))
            self._global_node.create_remote_table(
                table_name=table_name,
                table_schema=table_schema,
                native_node=node,
            )

    def _create_node_tables_for_global_udf(
        self,
        command_id: str,
        share_results: Union[bool, List[bool]],
        results_tables: List[Dict[_Node, _TableName]],
    ) -> List["_INodeTable"]:
        """
        Receives a list of results and if they should be shared,
        then creates the proper INodeTable objects.

        If the table should be shared, a merge table will be created in the global node.
        The tables will exist in the globalnode already (as remote tables), so only a "merge" action is needed.
        The response will either be a "GlobalNodeTable" or a "LocalNodeTable".

        Parameters
        ----------
            command_id: str
                The command identifier, common among all nodes for this action.
            share_results: Union[bool, List[bool]]
            results_tables: List[Dict[_Node, _TableName]]
                A udf can have multiple results and runs on multiple nodes.
                For example, there are 3 results, each result is in 2 nodes.
                udf_result_tables = [
                    {"node_1": "table_1",
                     "node_2": "table_1"},
                    {"node_1": "table_2",
                     "node_2": "table_2"},
                    {"node_1": "table_3",
                     "node_2": "table_3"},
                ]
        Returns
        -------
            List[_INodeTable]
        """
        if not share_results:
            return [_LocalNodeTable(nodes_tables=result) for result in results_tables]

        if type(share_results) == bool:
            share_results = [share_results]

        final_results = []
        for share, result_tables in zip(share_results, results_tables):
            if not share:
                final_results.append(_LocalNodeTable(nodes_tables=result_tables))
                continue

            remote_tablenames = list(result_tables.values())
            merge_table = self._global_node.create_merge_table(
                command_id=command_id, table_names=remote_tablenames
            )
            final_results.append(
                _GlobalNodeTable(node_table={self._global_node: merge_table})
            )
        return final_results

    def run_udf_on_global_node(
        self,
        func_name: str,
        positional_args: Optional[List["_GlobalNodeTable"]] = (),
        keyword_args: Optional[Dict[str, "_GlobalNodeTable"]] = {},
        share_to_locals: Union[bool, List[bool]] = None,
    ):  # -> List of GlobalNodeTables or LocalNodeTables
        # check the input tables are GlobalNodeTable(s)
        # queue exec_udf on the global node
        # wait for it to complete
        # a new table was generated on global node
        # queue create_remote_table on each of the local nodes for the ganerated table

        # TODO: try/catches tasks can throw exceptions
        command_id = get_next_command_id()

        if type(share_to_locals) == bool:
            share_to_locals = [share_to_locals]

        positional_udf_args = []
        for val in positional_args:
            if isinstance(val, _GlobalNodeTable):
                udf_argument = UDFArgument(
                    kind=UDFArgumentKind.TABLE,
                    value=list(val.node_table.values())[0].full_table_name,
                )  # TODO: da fuck is dat
            elif isinstance(val, _LocalNodeTable):
                raise Exception(
                    "(run_udf_on_global_node) LocalNodeTable types are not "
                    "accepted from run_udf_on_global_nodes"
                )
            else:
                udf_argument = UDFArgument(kind=UDFArgumentKind.LITERAL, value=str(val))
            positional_udf_args.append(udf_argument.json())

        keyword_udf_args = {}
        for var_name, val in keyword_args.items():
            if isinstance(val, _GlobalNodeTable):
                udf_argument = UDFArgument(
                    kind=UDFArgumentKind.TABLE,
                    value=list(val.node_table.values())[0].full_table_name,
                )  # TODO: da fuck is dat
            elif isinstance(val, _LocalNodeTable):
                raise Exception(
                    "(run_udf_on_global_node) LocalNodeTable types are not "
                    "accepted from run_udf_on_global_nodes"
                )
            else:
                udf_argument = UDFArgument(kind=UDFArgumentKind.LITERAL, value=str(val))
            keyword_udf_args[var_name] = udf_argument.json()

        task = self._global_node.queue_run_udf(
            command_id=command_id,
            func_name=func_name,
            positional_args=positional_udf_args,
            keyword_args=keyword_udf_args,
        )
        udf_result_tables = self._global_node.get_queued_udf_result(task)

        final_results_table = self._share_results_on_locals(
            func_name, share_to_locals, udf_result_tables
        )
        if len(final_results_table) == 1:
            return final_results_table[0]
        return final_results_table

    def _share_results_on_locals(
        self,
        func_name: str,
        share_to_locals: Union[bool, List[bool]],
        results_tables: List[str],
    ) -> List["_INodeTable"]:
        if not share_to_locals:
            return [
                _GlobalNodeTable(
                    node_table={self._global_node: _TableName(result_table)}
                )
                for result_table in results_tables
            ]

        if len(share_to_locals) != len(results_tables):
            raise AlgorithmExecutionException(
                f"Method {func_name} has {len(results_tables)} results but 'share_to_global' "
                f"has a length of {len(share_to_locals)}. They should match."
            )

        final_results_tables = []
        for share, result_table in zip(share_to_locals, results_tables):
            if not share:
                final_results_tables.append(
                    _GlobalNodeTable(
                        node_table={self._global_node: _TableName(result_table)}
                    )
                )
                continue

            table_schema = self._global_node.get_table_schema(_TableName(result_table))
            # TODO do not block here, first send the request to all local nodes and then block for the result
            for node in self._local_nodes:
                node.create_remote_table(
                    table_name=result_table,
                    table_schema=table_schema,
                    native_node=self._global_node,
                )

            local_nodes_tables = {
                node: _TableName(result_table) for node in self._local_nodes
            }
            final_results_tables.append(
                _LocalNodeTable(nodes_tables=local_nodes_tables)
            )

        return final_results_tables

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
                "(AlgorithmExecutionInterface::get_table_schema) node_table type-> {type(node_table)} not acceptable"
            )


class _INodeTable(ABC):
    # TODO: better abstraction here...
    pass


class _LocalNodeTable(_INodeTable):
    def __init__(self, nodes_tables: Dict["_Node", "_TableName"]):
        self._nodes_tables = nodes_tables

        if not self._validate_matching_table_names(list(self._nodes_tables.values())):
            raise self.MismatchingTableNamesException(
                [table_name.full_table_name for table_name in nodes_tables.values()]
            )

    @property
    def nodes_tables(self):
        return self._nodes_tables

    # TODO this is redundant, either remove it or overload all node methods here?
    def get_table_schema(self):
        node = list(self.nodes_tables.keys())[0]
        table = self.nodes_tables[node]
        return node.get_table_schema(table)

    def get_table_data(self):  # -> {Node:TableData}
        tables_data = []
        for node, table_name in self.nodes_tables.items():
            tables_data.append(node.get_table_data(table_name))
        tables_data_flat = [table_data.data_ for table_data in tables_data]
        tables_data_flat = [
            k for i in tables_data_flat for j in i for k in j
        ]  # TODO bejesus..
        return tables_data_flat

    def __repr__(self):
        r = "LocalNodeTable:\n"
        r += f"schema: {self.get_table_schema()}\n"
        for node, table_name in self.nodes_tables.items():
            r += f"{node} - {table_name} \ndata(LIMIT 20):\n"
            tmp = [str(row) for row in node.get_table_data(table_name).data_[0:20]]
            r += "\n".join(tmp)
            r += "\n"
        return r

    def _validate_matching_table_names(self, table_names: List[_TableName]):
        table_name_without_node_id = table_names[0].without_node_id()
        for table_name in table_names:
            if table_name.without_node_id() != table_name_without_node_id:
                return False
        return True

    class MismatchingTableNamesException(Exception):
        def __init__(self, table_names):
            self.message = f"Mismatched table names ->{table_names}"


class _GlobalNodeTable(_INodeTable):
    def __init__(self, node_table: Dict["_Node", "_TableName"]):
        self._node_table = node_table

    @property
    def node_table(self):
        return self._node_table

    # TODO this is redundant, either remove it or overload all node methods here?
    def get_table_schema(self):
        node = list(self.node_table.keys())[0]
        table_name: _TableName = list(self.node_table.values())[0]
        table_schema: TableSchema = node.get_table_schema(table_name).columns
        return table_schema

    def get_table_data(self):  # -> {Node:TableData}
        node = list(self.node_table.keys())[0]
        table_name: _TableName = list(self.node_table.values())[0]
        table_data = node.get_table_data(table_name).data_
        return table_data

    def __repr__(self):
        node = list(self.node_table.keys())[0]
        table_name: _TableName = list(self.node_table.values())[0]
        r = f"GlobalNodeTable: {table_name.full_table_name}"
        r += f"\nschema: {self.get_table_schema()}"
        r += f"\ndata (LIMIT 20): \n"
        tmp = [str(row) for row in self.get_table_data()[0:20]]
        r += "\n".join(tmp)
        return r


# NOTE tried to turn this into a generator, the problem is there are multiple consumers
# so the generator should be singleton in some way, the solutions were more complicated
# than this simple implementation
def get_next_command_id():
    if hasattr(get_next_command_id, "index"):
        get_next_command_id.index += 1
    else:
        get_next_command_id.index = 0
    return get_next_command_id.index
