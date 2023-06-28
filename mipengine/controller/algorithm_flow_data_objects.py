import warnings
from abc import ABC
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd

from mipengine import DATA_TABLE_PRIMARY_KEY
from mipengine.controller.nodes import GlobalNode
from mipengine.controller.nodes import LocalNode
from mipengine.node_tasks_DTOs import NodeLiteralDTO
from mipengine.node_tasks_DTOs import NodeSMPCDTO
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import NodeUDFDTO
from mipengine.node_tasks_DTOs import NodeUDFKeyArguments
from mipengine.node_tasks_DTOs import NodeUDFPosArguments
from mipengine.node_tasks_DTOs import SMPCTablesInfo
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableSchema


class AlgoFlowData(ABC):
    """
    AlgoFlowData are representing data objects in the algorithm flow.
    These objects are the result of running udfs and are used as input
    as well in the udfs.
    """

    _schema: TableSchema

    @property
    def full_schema(self):
        """
        Returns the full schema of the table, index + column names in a `TableSchema` format.
        """
        return self._schema

    @property
    def index(self):
        """
        Returns the index of the table schema if one exists.
        """
        if DATA_TABLE_PRIMARY_KEY in self._schema.column_names:
            return DATA_TABLE_PRIMARY_KEY
        else:
            return None

    @property
    def columns(self):
        """
        Returns the columns of the table without the index.
        """
        return [
            column.name
            for column in self._schema.columns
            if column.name != DATA_TABLE_PRIMARY_KEY
        ]


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


class LocalNodesTable(LocalNodesData):
    """
    A LocalNodesTable is a representation of a table across multiple nodes. To this end,
    it holds refferences to the actual nodes and tables through a dictionary with its keys
    being nodes and values being a table on that node

    example:
      When AlgorithmExecutionEngine::run_udf_on_local_nodes(..) is called, depending on
    how many local nodes are participating in the current algorithm execution, several
    database tables are created on all participating local nodes. Irrespective of the
    number of local nodes participating, the number of tables created on each of these local
    nodes will be the same.
    Class LocalNodeTable is the structure that represents the concept of these database
    tables, created during the execution of a udf, in the algorithm execution layer. A key
    concept is that a LocalNodeTable stores 'pointers' to 'relevant' tables existing in
    different local nodes across the federation. What 'relevant' means is that the tables
    are generated when triggering a udf execution across several local nodes. What
    'pointers' means is that there is a mapping between local nodes and table names and
    the aim is to hide the underline complexity from the algorithm flow and exposing a
    single 'local node table' object that stores in the background pointers to several
    tables in several local nodes.
    """

    _nodes_tables_info: Dict[LocalNode, TableInfo]

    def __init__(self, nodes_tables_info: Dict[LocalNode, TableInfo]):
        self._nodes_tables_info = nodes_tables_info
        self._validate_matching_table_names()
        self._schema = next(iter(nodes_tables_info.values())).schema_

    @property
    def nodes_tables_info(self) -> Dict[LocalNode, TableInfo]:
        return self._nodes_tables_info

    def get_table_data(self) -> List[Union[List[int], List[float], List[str]]]:
        """Gets merged data from corresponding table in all nodes

        A LocalNodesTable represents a collection of tables spread across
        multiple nodes. This method gets data from all tables and merges it in
        a common tables.

        The `row_id` column is treated differently. In each node `row_id` is an
        incrementing integer which is unique within each node, but not across
        nodes. Therefor, concatenating `row_id` columns would result in non
        unique values, hence the column would not be a primary key anymore. To
        remedy this, the current method prepends the node_id to each row_id
        using the format "node_id:row_id". This guarantees uniqueness of
        values, making `row_id` a primary key of the merged table.
        """

        msg = "LocalNodesTable.get_table_data should not be used in production."
        warnings.warn(msg)

        dataframe_per_node = {
            node.node_id: node.get_table_data(table_info.name).to_pandas()
            for node, table_info in self.nodes_tables_info.items()
        }

        # Sort according to node_id to have a common order for all tables
        sorted_node_ids = sorted(dataframe_per_node.keys())
        dataframe_per_node = {
            node_id: dataframe_per_node[node_id] for node_id in sorted_node_ids
        }

        # Make multi-index dataframe using both node_id and row_id
        for node_id, df in dataframe_per_node.items():
            df["node_id"] = node_id
            df.set_index(["node_id", "row_id"], inplace=True)

        # Merge dataframes from all nodes
        dataframes = list(dataframe_per_node.values())
        merged_df = pd.concat(dataframes)

        # Convert to List[Union[List[int], List[float], List[str]]] where the
        # row_id column is now composed of strings of the form node_id:row_id
        index = merged_df.index.tolist()
        index = [f"{node_id}:{row_id}" for node_id, row_id in index]
        data = merged_df.T.values.tolist()
        return [index] + data

    def __repr__(self):
        r = "LocalNodeTable:\n"
        for node, table_info in self.nodes_tables_info.items():
            r += f"\t{node=} {table_info=}\n"
        return r

    def _validate_matching_table_names(self):
        table_infos = list(self._nodes_tables_info.values())
        table_name_without_node_id = table_infos[0].name_without_node_id
        for table_name in table_infos:
            if table_name.name_without_node_id != table_name_without_node_id:
                raise MismatchingTableNamesException(
                    [table_info.name for table_info in table_infos]
                )


class GlobalNodeTable(GlobalNodeData):
    _node: GlobalNode
    _table_info: TableInfo

    def __init__(self, node: GlobalNode, table_info: TableInfo):
        self._node = node
        self._table_info = table_info
        self._schema = table_info.schema_

    @property
    def node(self) -> GlobalNode:
        return self._node

    @property
    def table_info(self) -> TableInfo:
        return self._table_info

    def get_table_data(self) -> List[List[Any]]:
        table_data = [
            column.data
            for column in self.node.get_table_data(self.table_info.name).columns
        ]
        return table_data

    def __repr__(self):
        r = f"\n\tGlobalNodeTable: \n\t{self._schema=}\n \t{self.table_info=}\n"
        return r


class LocalNodesSMPCTables(LocalNodesData):
    _smpc_tables_info_per_node: Dict[LocalNode, SMPCTablesInfo]

    def __init__(self, smpc_tables_info_per_node: Dict[LocalNode, SMPCTablesInfo]):
        self._smpc_tables_info_per_node = smpc_tables_info_per_node

    @property
    def nodes_smpc_tables(self) -> Dict[LocalNode, SMPCTablesInfo]:
        return self._smpc_tables_info_per_node

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
    _smpc_tables_info: SMPCTablesInfo

    def __init__(self, node: GlobalNode, smpc_tables_info: SMPCTablesInfo):
        self._node = node
        self._smpc_tables_info = smpc_tables_info

    @property
    def node(self) -> GlobalNode:
        return self._node

    @property
    def smpc_tables_info(self) -> SMPCTablesInfo:
        return self._smpc_tables_info


def algoexec_udf_kwargs_to_node_udf_kwargs(
    algoexec_kwargs: Dict[str, Any],
    local_node: LocalNode = None,
) -> NodeUDFKeyArguments:
    if not algoexec_kwargs:
        return NodeUDFKeyArguments(args={})

    args = {}
    for key, arg in algoexec_kwargs.items():
        udf_argument = _algoexec_udf_arg_to_node_udf_arg(arg, local_node)
        args[key] = udf_argument
    return NodeUDFKeyArguments(args=args)


def algoexec_udf_posargs_to_node_udf_posargs(
    algoexec_posargs: List[Any],
    local_node: LocalNode = None,
) -> NodeUDFPosArguments:
    if not algoexec_posargs:
        return NodeUDFPosArguments(args=[])

    args = []
    for arg in algoexec_posargs:
        args.append(_algoexec_udf_arg_to_node_udf_arg(arg, local_node))
    return NodeUDFPosArguments(args=args)


def _algoexec_udf_arg_to_node_udf_arg(
    algoexec_arg: AlgoFlowData, local_node: LocalNode = None
) -> NodeUDFDTO:
    """
    Converts the algorithm executor run_udf input arguments, coming from the algorithm flow
    to node udf pos/key arguments to be sent to the NODE.

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
        return NodeTableDTO(value=algoexec_arg.nodes_tables_info[local_node])
    elif isinstance(algoexec_arg, GlobalNodeTable):
        return NodeTableDTO(value=algoexec_arg.table_info)
    elif isinstance(algoexec_arg, LocalNodesSMPCTables):
        raise ValueError(
            "'LocalNodesSMPCTables' cannot be used as argument. It must be shared."
        )
    elif isinstance(algoexec_arg, GlobalNodeSMPCTables):
        return NodeSMPCDTO(value=algoexec_arg.smpc_tables_info)
    else:
        return NodeLiteralDTO(value=algoexec_arg)


def create_node_table_dto_from_global_node_table(table_info: TableInfo):
    if not table_info:
        return None

    return NodeTableDTO(value=table_info)


class MismatchingTableNamesException(Exception):
    def __init__(self, table_names: List[str]):
        message = f"Mismatched table names ->{table_names}"
        super().__init__(message)
        self.message = message
