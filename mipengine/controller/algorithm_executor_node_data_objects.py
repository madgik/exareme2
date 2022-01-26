from abc import ABC
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from mipengine.node_tasks_DTOs import NodeLiteralDTO
from mipengine.node_tasks_DTOs import NodeSMPCDTO
from mipengine.node_tasks_DTOs import NodeSMPCValueDTO
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import NodeUDFDTO
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFKeyArguments
from mipengine.node_tasks_DTOs import UDFPosArguments


class NodeData(ABC):
    """
    NodeData are located into one specific Node.
    """

    pass


class NodeTable(NodeData):
    def __init__(self, table_name):
        self._full_name = table_name
        full_name_split = self._full_name.split("_")
        self._table_type = full_name_split[0]
        self._node_id = full_name_split[1]
        self._context_id = full_name_split[2]
        self._command_id = full_name_split[3]
        self._command_subid = full_name_split[4]

    @property
    def full_table_name(self) -> str:
        return self._full_name

    @property
    def table_type(self) -> str:
        return self._table_type

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def context_id(self) -> str:
        return self._context_id

    @property
    def command_id(self) -> str:
        return self._command_id

    @property
    def command_subid(self) -> str:
        return self._command_subid

    def without_node_id(self) -> str:
        return (
            self._table_type
            + "_"
            + self._context_id
            + "_"
            + self._command_id
            + "_"
            + self._command_subid
        )

    def __repr__(self):
        return self.full_table_name


class NodeSMPCTables(NodeData):
    template: NodeTable
    add_op: NodeTable
    min_op: NodeTable
    max_op: NodeTable
    union_op: NodeTable

    def __init__(self, template, add_op, min_op, max_op, union_op):
        self.template = template
        self.add_op = add_op
        self.min_op = min_op
        self.max_op = max_op
        self.union_op = union_op


class AlgoExecData(ABC):
    """
    AlgoExecData are representing one data object but could be located into
    more than one node. For example the LocalNodesTable is treated as one table
    from the algorithm executor but is located in multiple local nodes.
    """

    pass


class LocalNodesData(AlgoExecData, ABC):
    pass


class GlobalNodeData(AlgoExecData, ABC):
    pass


# When _AlgorithmExecutionInterface::run_udf_on_local_nodes(..) is called, depending on
# how many local nodes are participating in the current algorithm execution, several
# database tables are created on all participating local nodes. Irrespectevely of the
# number of local nodes participating, the number of tables created on each of these local
# nodes will be the same.
# Class _LocalNodeTable is the structure that represents the concept of these database
# tables, created during the execution of a udf, in the algorithm execution layer. A key
# concept is that a _LocalNodeTable stores 'pointers' to 'relevant' tables existing in
# different local nodes accross the federation. By 'relevant' I mean tables that are
# generated when triggering a udf execution accross several local nodes. By 'pointers'
# I mean mapping between local nodes and table names and the aim is to hide the underline
# complexity from the algorithm flow and exposing a single 'local node table' object that
# stores in the background pointers to several tables in several local nodes.
class LocalNodesTable(LocalNodesData):
    def __init__(self, nodes_tables: Dict["LocalNode", NodeTable]):
        self._nodes_tables = nodes_tables
        self._validate_matching_table_names(list(self._nodes_tables.values()))

    @property
    def nodes_tables(self) -> Dict["LocalNode", NodeTable]:
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

    def _validate_matching_table_names(self, table_names: List[NodeTable]):
        table_name_without_node_id = table_names[0].without_node_id()
        for table_name in table_names:
            if table_name.without_node_id() != table_name_without_node_id:
                raise self.MismatchingTableNamesException(
                    [table_name.full_table_name for table_name in table_names]
                )

    class MismatchingTableNamesException(Exception):
        def __init__(self, table_names: List[str]):
            message = f"Mismatched table names ->{table_names}"
            super().__init__(message)
            self.message = message


class GlobalNodeTable(GlobalNodeData):
    def __init__(self, node: "GlobalNode", table: NodeTable):
        self._node = node
        self._table = table

    @property
    def node(self) -> "GlobalNode":
        return self._node

    @property
    def table(self) -> NodeTable:
        return self._table

    # TODO this is redundant, either remove it or overload all node methods here?
    def get_table_schema(self) -> TableSchema:
        return self._node.get_table_schema(self.table)

    def get_table_data(self) -> List[List[Any]]:
        table_data = [
            column.data for column in self.node.get_table_data(self.table).columns
        ]
        return table_data

    def __repr__(self):
        r = f"\n\tGlobalNodeTable: \n\tschema={self.get_table_schema()}\n \t{self.table=}\n"
        return r


class LocalNodesSMPCTables(LocalNodesData):
    template: LocalNodesTable
    add_op: LocalNodesTable
    min_op: LocalNodesTable
    max_op: LocalNodesTable
    union_op: LocalNodesTable

    def __init__(self, nodes_smpc_tables: Dict["LocalNode", NodeSMPCTables]):
        template_nodes_tables = {}
        add_op_nodes_tables = {}
        min_op_nodes_tables = {}
        max_op_nodes_tables = {}
        union_op_nodes_tables = {}
        for node, node_smpc_tables in nodes_smpc_tables.items():
            template_nodes_tables[node] = node_smpc_tables.template
            if node_smpc_tables.add_op:
                add_op_nodes_tables[node] = node_smpc_tables.add_op
            if node_smpc_tables.min_op:
                min_op_nodes_tables[node] = node_smpc_tables.min_op
            if node_smpc_tables.max_op:
                max_op_nodes_tables[node] = node_smpc_tables.max_op
            if node_smpc_tables.union_op:
                union_op_nodes_tables[node] = node_smpc_tables.union_op
        self.template = LocalNodesTable(template_nodes_tables)
        self.add_op = (
            LocalNodesTable(add_op_nodes_tables) if add_op_nodes_tables else None
        )
        self.min_op = (
            LocalNodesTable(min_op_nodes_tables) if min_op_nodes_tables else None
        )
        self.max_op = (
            LocalNodesTable(max_op_nodes_tables) if max_op_nodes_tables else None
        )
        self.union_op = (
            LocalNodesTable(union_op_nodes_tables) if union_op_nodes_tables else None
        )


class GlobalNodeSMPCTables(GlobalNodeData):
    template: GlobalNodeTable
    add_op: GlobalNodeTable
    min_op: GlobalNodeTable
    max_op: GlobalNodeTable
    union_op: GlobalNodeTable

    def __init__(self, template, add_op, min_op, max_op, union_op):
        self.template = template
        self.add_op = add_op
        self.min_op = min_op
        self.max_op = max_op
        self.union_op = union_op


def algoexec_udf_kwargs_to_node_udf_kwargs(
    algoexec_kwargs: Dict[str, Any],
    local_node: "LocalNode" = None,
) -> UDFKeyArguments:
    if not algoexec_kwargs:
        return UDFKeyArguments(args={})

    args = {}
    for key, arg in algoexec_kwargs.items():
        udf_argument = _algoexec_udf_arg_to_node_udf_arg(arg, local_node)
        args[key] = udf_argument
    return UDFKeyArguments(args=args)


def algoexec_udf_posargs_to_node_udf_posargs(
    algoexec_posargs: List[Any],
    local_node: "LocalNode" = None,
) -> UDFPosArguments:
    if not algoexec_posargs:
        return UDFPosArguments(args=[])

    args = []
    for arg in algoexec_posargs:
        args.append(_algoexec_udf_arg_to_node_udf_arg(arg, local_node))
    return UDFPosArguments(args=args)


def _algoexec_udf_arg_to_node_udf_arg(
    algoexec_arg: AlgoExecData, local_node: "LocalNode" = None
) -> NodeUDFDTO:
    """
    Converts the algorithm executor run_udf input arguments, coming from the algorithm flow
    to node udf pos/key arguments to be send to the NODE.

    Parameters
    ----------
    algoexec_arg is the argument to be converted.
    local_node is need only when the algoexec_arg is of LocalNodesTable, to know
                which local table should be selected.

    Returns
    -------
    a NodeUDFDTO
    """
    if isinstance(algoexec_arg, LocalNodesTable):
        if not local_node:
            raise ValueError(
                "local_node parameter is required on LocalNodesTable convertion."
            )
        return NodeTableDTO(value=algoexec_arg.nodes_tables[local_node].full_table_name)
    elif isinstance(algoexec_arg, GlobalNodeTable):
        return NodeTableDTO(value=algoexec_arg.table.full_table_name)
    elif isinstance(algoexec_arg, LocalNodesSMPCTables):
        return NodeSMPCDTO(
            value=NodeSMPCValueDTO(
                template=algoexec_arg.template.nodes_tables[local_node].full_table_name,
                add_op_values=algoexec_arg.add_op.nodes_tables[
                    local_node
                ].full_table_name,
                min_op_values=algoexec_arg.min_op.nodes_tables[
                    local_node
                ].full_table_name,
                max_op_values=algoexec_arg.max_op.nodes_tables[
                    local_node
                ].full_table_name,
                union_op_values=algoexec_arg.union_op.nodes_tables[
                    local_node
                ].full_table_name,
            )
        )
    elif isinstance(algoexec_arg, GlobalNodeSMPCTables):
        return NodeSMPCDTO(
            value=NodeSMPCValueDTO(
                template=NodeTableDTO(
                    value=algoexec_arg.template.table.full_table_name
                ),
                add_op_values=NodeTableDTO(
                    value=algoexec_arg.add_op.table.full_table_name
                )
                if algoexec_arg.add_op
                else None,
                min_op_values=NodeTableDTO(
                    value=algoexec_arg.min_op.table.full_table_name
                )
                if algoexec_arg.min_op
                else None,
                max_op_values=NodeTableDTO(
                    value=algoexec_arg.max_op.table.full_table_name
                )
                if algoexec_arg.max_op
                else None,
                union_op_values=NodeTableDTO(
                    value=algoexec_arg.union_op.table.full_table_name
                )
                if algoexec_arg.union_op
                else None,
            )
        )
    else:
        return NodeLiteralDTO(value=algoexec_arg)
