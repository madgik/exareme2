from typing import Dict, List, Tuple, Any
import importlib
from pydantic import BaseModel

from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import TableType
from mipengine.node_tasks_DTOs import UDFArgument
from mipengine.node_tasks_DTOs import UDFArgumentKind

from mipengine.controller.algorithm_execution_DTOs import (
    AlgorithmExecutionDTO,
    NodesTasksHandlersDTO,
)
from mipengine.controller.node_tasks_handler_interface import INodeTasksHandler

ALGORITHMS_FOLDER = "mipengine.algorithms"


class _TableName:
    def __init__(self, table_name):
        self._full_name = table_name
        full_name_split = self._full_name.split("_")
        self._table_type = full_name_split[0]
        self._command_id = full_name_split[1]
        self._context_id = full_name_split[2]
        self._node_id = full_name_split[3]

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


class AlgorithmExecutor:
    def __init__(
        self,
        algorithm_execution_dto: AlgorithmExecutionDTO,
        nodes_tasks_handlers_dto: NodesTasksHandlersDTO,
    ):
        self._context_id = algorithm_execution_dto.context_id
        self._algorithm_name = algorithm_execution_dto.algorithm_name

        # instantiate the GLOBAL Node object
        self.global_node = _Node(
            context_id=self._context_id,
            node_tasks_handler=nodes_tasks_handlers_dto.global_node_tasks_handler,
        )

        # Parameters for the creation of the view tables in the db. Each of the LOCAL
        # nodes will have access only to these view tables and not on the primary data
        # tables
        initial_view_tables_params = {
            "commandId": get_next_command_id(),
            "pathology": algorithm_execution_dto.algorithm_request_dto.inputdata.pathology,
            "datasets": algorithm_execution_dto.algorithm_request_dto.inputdata.datasets,
            "x": algorithm_execution_dto.algorithm_request_dto.inputdata.x,
            "y": algorithm_execution_dto.algorithm_request_dto.inputdata.y,
            "filters": algorithm_execution_dto.algorithm_request_dto.inputdata.filters,
        }

        # instantiate the LOCAL Node objects
        self.local_nodes = [
            _Node(
                context_id=self._context_id,
                node_tasks_handler=node_tasks_handler,
                initial_view_tables_params=initial_view_tables_params,
            )
            for node_tasks_handler in nodes_tasks_handlers_dto.local_nodes_tasks_handlers
        ]

        algo_execution_interface_dto = _AlgorithmExecutionInterfaceDTO(
            global_node=self.global_node,
            local_nodes=self.local_nodes,
            algorithm_name=self._algorithm_name,
            algorithm_parameters=algorithm_execution_dto.algorithm_request_dto.parameters,
            x_variables=algorithm_execution_dto.algorithm_request_dto.inputdata.x,
            y_variables=algorithm_execution_dto.algorithm_request_dto.inputdata.y,
        )
        self.execution_interface = _AlgorithmExecutionInterface(
            algo_execution_interface_dto
        )

        # import the algorithm flow module
        self.algorithm_flow_module = importlib.import_module(
            f"{ALGORITHMS_FOLDER}.{self._algorithm_name}"
        )

    def run(self):
        algorithm_result = self.algorithm_flow_module.run(self.execution_interface)

        self.clean_up()

        return algorithm_result

    def clean_up(self):
        self.global_node.clean_up()
        for node in self.local_nodes:
            node.clean_up()


class _Node:
    def __init__(
        self,
        context_id: str,
        node_tasks_handler: INodeTasksHandler,
        initial_view_tables_params: Dict[str, Any] = None,
    ):
        self.node_tasks_handler = node_tasks_handler
        self.node_id = self.node_tasks_handler.node_id

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
        return self.node_tasks_handler.node_data_address

    # TABLES functionality
    def get_tables(self) -> List[_TableName]:
        return self.node_tasks_handler.get_tables(context_id=self.context_id)

    def get_table_schema(self, table_name: _TableName):
        return self.node_tasks_handler.get_table_schema(
            table_name=table_name.full_table_name
        )

    def get_table_data(self, table_name: _TableName) -> TableData:
        return self.node_tasks_handler.get_table_data(table_name.full_table_name)

    def create_table(self, command_id: str, schema: TableSchema) -> _TableName:
        schema_json = schema.json()
        return self.node_tasks_handler.create_table(
            context_id=self.context_id,
            command_id=command_id,
            schema_json=schema_json,
        )

    # VIEWS functionality
    def get_views(self) -> List[_TableName]:
        result = self.node_tasks_handler.get_views(context_id=self.context_id)
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
        result = self.node_tasks_handler.create_pathology_view(
            context_id=self.context_id,
            command_id=command_id,
            pathology=pathology,
            columns=columns,
            filters=filters,
        )
        return _TableName(result)

    # MERGE TABLES functionality
    def get_merge_tables(self) -> List[_TableName]:
        result = self.node_tasks_handler.get_merge_tables(context_id=self.context_id)
        return [_TableName(table_name) for table_name in result]

    def create_merge_table(self, command_id: str, table_names: List[_TableName]):
        table_names = [table_name.full_table_name for table_name in table_names]
        result = self.node_tasks_handler.create_merge_table(
            context_id=self.context_id,
            command_id=command_id,
            table_names=table_names,
        )
        return _TableName(result)

    # REMOTE TABLES functionality
    def get_remote_tables(self) -> List["TableInfo"]:
        return self.node_tasks_handler.get_remote_tables(context_id=self.context_id)

    def create_remote_table(
        self, table_info: TableInfo, native_node: "_Node"
    ) -> _TableName:

        monetdb_socket_addr = native_node.node_address
        return self.node_tasks_handler.create_remote_table(
            table_info=table_info, original_db_url=monetdb_socket_addr
        )

    # UDFs functionality
    def queue_run_udf(
        self, command_id: str, func_name: str, positional_args, keyword_args
    ):  # -> "AsyncResult"
        return self.node_tasks_handler.queue_run_udf(
            context_id=self.context_id,
            command_id=command_id,
            func_name=func_name,
            positional_args=positional_args,
            keyword_args=keyword_args,
        )

    def get_udfs(self, algorithm_name) -> List[str]:
        return self.node_tasks_handler.get_udfs(algorithm_name)

    # return the generated monetdb pythonudf
    def get_run_udf_query(
        self, command_id: str, func_name: str, positional_args: List["_NodeTable"]
    ) -> Tuple[str, str]:
        return self.node_tasks_handler.get_run_udf_query(
            context_id=self.context_id,
            command_id=command_id,
            func_name=func_name,
            positional_args=positional_args,
        )

    # CLEANUP functionality
    def clean_up(self):
        self.node_tasks_handler.clean_up(context_id=self.context_id)


class _AlgorithmExecutionInterfaceDTO(BaseModel):
    global_node: _Node
    local_nodes: List[_Node]
    algorithm_name: str
    algorithm_parameters: Dict[str, List[str]]
    x_variables: List[str]
    y_variables: List[str]

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
        positional_args: Dict[str, "_LocalNodeTable"],
        share_to_global: bool = False,
    ):  # -> GlobalNodeTable or LocalNodeTable
        # queue exec_udf task on all local nodes
        # wait for all nodes to complete the tasks execution
        # one new table per local node was generated
        # queue create_remote_table on global for each of the generated tables
        # create merge table on global node to merge the remote tables

        command_id = get_next_command_id()

        tasks = {}
        for node in self._local_nodes:
            # TODO get the nodes from the LocalNodeTables in the positional_args
            positional_args_transfrormed = []
            keyword_args_transformed = {}
            for var_name, val in positional_args.items():
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
                positional_args_transfrormed.append(udf_argument.json())
                # keyword_args_transformed[var_name] = udf_argument.json()

            task = node.queue_run_udf(
                command_id=command_id,
                func_name=func_name,
                positional_args=positional_args_transfrormed,
                keyword_args={},
            )
            tasks[node] = task

        udf_result_tables = {}
        for node, task in tasks.items():
            table_name = _TableName(task.get())
            udf_result_tables[node] = table_name

            # ceate remote table on global node
            if share_to_global:
                # TODO: try block missing
                table_schema = node.get_table_schema(table_name)
                table_info = TableInfo(
                    name=table_name.full_table_name,
                    schema_=table_schema,
                    type_=TableType.REMOTE,
                )
                self._global_node.create_remote_table(
                    table_info=table_info, native_node=node
                )

        # create merge table on global
        if share_to_global:
            remote_tables_info = list(udf_result_tables.values())
            remote_table_names = [
                remote_table_info for remote_table_info in remote_tables_info
            ]
            merge_table_global = self._global_node.create_merge_table(
                command_id=command_id, table_names=remote_table_names
            )
            return _GlobalNodeTable(node_table={self._global_node: merge_table_global})

        else:
            return _LocalNodeTable(nodes_tables=udf_result_tables)

    def run_udf_on_global_node(
        self,
        func_name: str,
        positional_args: List["_GlobalNodeTable"],
        share_to_locals: bool = False,
    ):  # -> GlobalNodeTable or LocalNodeTable
        # check the input tables are GlobalNodeTable(s)
        # queue exec_udf on the global node
        # wait for it to complete
        # a new table was generated on global node
        # queue create_remote_table on each of the local nodes for the ganerated table

        # TODO: try/catches tasks can throw exceptions
        command_id = get_next_command_id()

        positional_args_transfrormed = []
        # keyword_args_transformed = {}
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
            positional_args_transfrormed.append(udf_argument.json())

        udf_result_table: str = self._global_node.queue_run_udf(
            command_id=command_id,
            func_name=func_name,
            positional_args=positional_args_transfrormed,
            keyword_args={},
        ).get()

        if share_to_locals:
            table_schema: TableSchema = self._global_node.get_table_schema(
                _TableName(udf_result_table)
            )
            table_info: TableInfo = TableInfo(
                name=udf_result_table, schema_=table_schema, type_=TableType.REMOTE
            )
            local_nodes_tables = {}
            for node in self._local_nodes:
                # TODO do not block here, first send the request to all local nodes and then block for the result
                node.create_remote_table(
                    table_info=table_info, native_node=self._global_node
                )
                local_nodes_tables[node] = _TableName(udf_result_table)

            return _LocalNodeTable(nodes_tables=local_nodes_tables)

        return _GlobalNodeTable(
            node_table={self._global_node: _TableName(udf_result_table)}
        )

    # TABLES functionality
    def get_table_data(self, node_table) -> TableData:
        return node_table.get_table_data()

    def get_table_schema(self, node_table) -> TableSchema:
        # TODO create super class NodeTable??
        if isinstance(node_table, _LocalNodeTable) or isinstance(
            node_table, _GlobalNodeTable
        ):
            return node_table.get_table_schema()
        else:
            raise Exception(
                "(AlgorithmExecutionInterface::get_table_schema) node_table type-> {type(node_table)} not acceptable"
            )


class _NodeTable:
    # TODO: better abstraction here...
    pass


class _LocalNodeTable:
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


class _GlobalNodeTable:
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
