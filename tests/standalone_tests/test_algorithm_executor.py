from random import randint
from typing import Dict
from typing import List

import pytest

from mipengine.controller.algorithm_executor_node_data_objects import NodeTable
from mipengine.controller.algorithm_executor_nodes import _INode
from mipengine.controller.node_tasks_handler_interface import IQueuedUDFAsyncResult
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import DType
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema


# TODO does not contain any test, just a placeholder..


class NodeMock(_INode):
    def __init__(self):
        self.tables: Dict[str, TableSchema] = {}

    def get_tables(self) -> List[NodeTable]:
        pass

    def get_table_schema(self, table_name: NodeTable):
        return self.tables[table_name]

    def get_table_data(self, table_name: NodeTable) -> TableData:
        pass

    def create_table(self, command_id: str, schema: TableSchema) -> NodeTable:
        table_name = f"normal_testnode_cntxtid1_cmdid{randint(0,999)}_cmdsubid1"
        table_name = NodeTable(table_name)
        self.tables[table_name] = schema
        return table_name

    def get_views(self) -> List[NodeTable]:
        pass

    def create_pathology_view(
        self,
        command_id: str,
        pathology: str,
        version: str,
        columns: List[str],
        filters: List[str],
    ) -> NodeTable:
        pass

    def get_merge_tables(self) -> List[NodeTable]:
        pass

    def create_merge_table(self, command_id: str, table_names: List[NodeTable]):
        pass

    def get_remote_tables(self) -> List[str]:
        pass

    def create_remote_table(
        self, table_name: str, table_schema: TableSchema, native_node: _INode
    ):
        pass

    def queue_run_udf(
        self, command_id: str, func_name: str, positional_args, keyword_args
    ) -> IQueuedUDFAsyncResult:
        pass

    def get_queued_udf_result(
        self, async_result: IQueuedUDFAsyncResult
    ) -> List[NodeTable]:
        pass

    def get_udfs(self, algorithm_name) -> List[str]:
        pass


@pytest.fixture
def test_table_schema_a():
    command_id = "cmndid1"
    schema = TableSchema(
        columns=[
            ColumnInfo(name="var1", dtype=DType.INT),
            ColumnInfo(name="var2", dtype=DType.STR),
        ]
    )
    return {"command_id": command_id, "schema": schema}


@pytest.fixture
def test_table_schema_b():
    command_id = "cmndid1"
    schema = TableSchema(
        columns=[
            ColumnInfo(name="var1", dtype=DType.FLOAT),
            ColumnInfo(name="var2", dtype=DType.INT),
        ]
    )
    return {"command_id": command_id, "schema": schema}
