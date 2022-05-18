from random import randint
from typing import Dict
from typing import List

import pytest

from mipengine.controller.algorithm_executor_node_data_objects import TableName
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

    def get_tables(self) -> List[TableName]:
        pass

    def get_table_schema(self, table_name: TableName):
        return self.tables[table_name]

    def get_table_data(self, table_name: TableName) -> TableData:
        pass

    def create_table(self, command_id: str, schema: TableSchema) -> TableName:
        table_name = f"normal_testnode_cntxtid1_cmdid{randint(0,999)}_cmdsubid1"
        table_name = TableName(table_name)
        self.tables[table_name] = schema
        return table_name

    def get_views(self) -> List[TableName]:
        pass

    def create_data_model_views(
        self,
        command_id: str,
        data_model: str,
        datasets: List[str],
        columns_per_view: List[List[str]],
        filters: dict = None,
        dropna: bool = True,
        check_min_rows: bool = True,
    ) -> List[TableName]:
        pass

    def get_merge_tables(self) -> List[TableName]:
        pass

    def create_merge_table(self, command_id: str, table_names: List[TableName]):
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
    ) -> List[TableName]:
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
