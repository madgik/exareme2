import warnings
from abc import ABC
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from mipengine.controller.algorithm_executor_node_data_objects import SMPCTableNames
from mipengine.controller.algorithm_executor_node_data_objects import TableName
from mipengine.controller.algorithm_executor_nodes import GlobalNode
from mipengine.controller.algorithm_executor_nodes import LocalNode
from mipengine.node_tasks_DTOs import NodeLiteralDTO
from mipengine.node_tasks_DTOs import NodeSMPCDTO
from mipengine.node_tasks_DTOs import NodeSMPCValueDTO
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import NodeUDFDTO
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFKeyArguments
from mipengine.node_tasks_DTOs import UDFPosArguments


class AlgoFlowData(ABC):
    """
    AlgoFlowData are representing data objects in the algorithm flow.
    These objects are the result of running udfs and are used as input
    as well in the udfs.
    """

    pass


class LocalNodesData(AlgoFlowData, ABC):
    """
    LocalNodesData are representing data objects in the algorithm flow
    that are located in many (or one) local nodes.
    """

    pass


class GlobalNodeData(AlgoFlowData, ABC):
    """
    GlobalNodeData are representing data objects in the algorithm flow
    that are located in the global node.
    """

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
    _nodes_tables: Dict[LocalNode, TableName]

    def __init__(self, nodes_tables: Dict[LocalNode, TableName]):
        self._nodes_tables = nodes_tables
        self._validate_matching_table_names(list(self._nodes_tables.values()))

    @property
    def nodes_tables(self) -> Dict[LocalNode, TableName]:
        return self._nodes_tables

    # TODO this is redundant, either remove it or overload all node methods here?
    def get_table_schema(self) -> TableSchema:
        node = list(self.nodes_tables.keys())[0]
        table = self.nodes_tables[node]
        return node.get_table_schema(table)

    def get_table_data(self) -> List[Union[int, float, str]]:
        """
        Should be used ONLY for debugging.
        """
        warnings.warn(
            "'get_table_data' of 'LocalNodesTable' should not be used in production."
        )

        tables_data = []
        for node, table_name in self.nodes_tables.items():
            tables_data.append(node.get_table_data(table_name).columns)

        merged_table_data = []
        for table in tables_data:
            for index, column in enumerate(table):
                if len(merged_table_data) <= index:
                    merged_table_data.append(column.data)
                else:
                    merged_table_data[index].extend(column.data)
        return merged_table_data

    def __repr__(self):
        r = f"\n\tLocalNodeTable: {self.get_table_schema()}\n"
        for node, table_name in self.nodes_tables.items():
            r += f"\t{node=} {table_name=}\n"
        return r

    def _validate_matching_table_names(self, table_names: List[TableName]):
        table_name_without_node_id = table_names[0].without_node_id()
        for table_name in table_names:
            if table_name.without_node_id() != table_name_without_node_id:
                raise MismatchingTableNamesException(
                    [table_name.full_table_name for table_name in table_names]
                )


class GlobalNodeTable(GlobalNodeData):
    _node: GlobalNode
    _table: TableName

    def __init__(self, node: GlobalNode, table: TableName):
        self._node = node
        self._table = table

    @property
    def node(self) -> GlobalNode:
        return self._node

    @property
    def table(self) -> TableName:
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
    _nodes_smpc_tables: Dict[LocalNode, SMPCTableNames]

    def __init__(self, nodes_smpc_tables: Dict[LocalNode, SMPCTableNames]):
        self._nodes_smpc_tables = nodes_smpc_tables

    @property
    def nodes_smpc_tables(self) -> Dict[LocalNode, SMPCTableNames]:
        return self._nodes_smpc_tables

    @property
    def template_local_nodes_table(self) -> LocalNodesTable:
        return LocalNodesTable(
            {node: tables.template for node, tables in self.nodes_smpc_tables.items()}
        )

    @property
    def sum_op_local_nodes_table(self) -> Optional[LocalNodesTable]:
        nodes_tables = {}
        for node, tables in self.nodes_smpc_tables.items():
            if not tables.sum_op:
                return None
            nodes_tables[node] = tables.sum_op
        return LocalNodesTable(nodes_tables)

    @property
    def min_op_local_nodes_table(self) -> Optional[LocalNodesTable]:
        nodes_tables = {}
        for node, tables in self.nodes_smpc_tables.items():
            if not tables.min_op:
                return None
            nodes_tables[node] = tables.min_op
        return LocalNodesTable(nodes_tables)

    @property
    def max_op_local_nodes_table(self) -> Optional[LocalNodesTable]:
        nodes_tables = {}
        for node, tables in self.nodes_smpc_tables.items():
            if not tables.max_op:
                return None
            nodes_tables[node] = tables.max_op
        return LocalNodesTable(nodes_tables)


class GlobalNodeSMPCTables(GlobalNodeData):
    _node: GlobalNode
    _smpc_tables: SMPCTableNames

    def __init__(self, node: GlobalNode, smpc_tables: SMPCTableNames):
        self._node = node
        self._smpc_tables = smpc_tables

    @property
    def node(self) -> GlobalNode:
        return self._node

    @property
    def smpc_tables(self) -> SMPCTableNames:
        return self._smpc_tables


def algoexec_udf_kwargs_to_node_udf_kwargs(
    algoexec_kwargs: Dict[str, Any],
    local_node: LocalNode = None,
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
    local_node: LocalNode = None,
) -> UDFPosArguments:
    if not algoexec_posargs:
        return UDFPosArguments(args=[])

    args = []
    for arg in algoexec_posargs:
        args.append(_algoexec_udf_arg_to_node_udf_arg(arg, local_node))
    return UDFPosArguments(args=args)


def _algoexec_udf_arg_to_node_udf_arg(
    algoexec_arg: AlgoFlowData, local_node: LocalNode = None
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
                "local_node parameter is required on LocalNodesTable conversion."
            )
        return NodeTableDTO(value=algoexec_arg.nodes_tables[local_node].full_table_name)
    elif isinstance(algoexec_arg, GlobalNodeTable):
        return NodeTableDTO(value=algoexec_arg.table.full_table_name)
    elif isinstance(algoexec_arg, LocalNodesSMPCTables):
        raise ValueError(
            "'LocalNodesSMPCTables' cannot be used as argument. It must be shared."
        )
    elif isinstance(algoexec_arg, GlobalNodeSMPCTables):
        return NodeSMPCDTO(
            value=NodeSMPCValueDTO(
                template=NodeTableDTO(
                    value=algoexec_arg.smpc_tables.template.full_table_name
                ),
                sum_op_values=create_node_table_dto_from_global_node_table(
                    algoexec_arg.smpc_tables.sum_op
                ),
                min_op_values=create_node_table_dto_from_global_node_table(
                    algoexec_arg.smpc_tables.min_op
                ),
                max_op_values=create_node_table_dto_from_global_node_table(
                    algoexec_arg.smpc_tables.max_op
                ),
            )
        )
    else:
        return NodeLiteralDTO(value=algoexec_arg)


def create_node_table_dto_from_global_node_table(table: TableName):
    if not table:
        return None

    return NodeTableDTO(value=table.full_table_name)


def create_local_nodes_table_from_nodes_tables(
    nodes_tables: Dict[LocalNode, Union[TableName, None]]
):
    for table in nodes_tables.values():
        if not table:
            return None

    return LocalNodesTable(nodes_tables)


class MismatchingTableNamesException(Exception):
    def __init__(self, table_names: List[str]):
        message = f"Mismatched table names ->{table_names}"
        super().__init__(message)
        self.message = message
