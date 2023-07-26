from typing import Dict
from typing import List

import pytest

from exareme2.controller.nodes import _INode
from exareme2.node_tasks_DTOs import ColumnInfo
from exareme2.node_tasks_DTOs import DType
from exareme2.node_tasks_DTOs import TableData
from exareme2.node_tasks_DTOs import TableInfo
from exareme2.node_tasks_DTOs import TableSchema

# TODO does not contain any test, just a placeholder..


class AsyncResult:
    def get(self, timeout=None):
        pass


class NodeMock(_INode):
    def __init__(self):
        self.tables: Dict[str, TableSchema] = {}

    def get_tables(self) -> List[str]:
        pass

    def get_table_data(self, table_name: str) -> TableData:
        pass

    def create_table(self, command_id: str, schema: TableSchema) -> TableInfo:
        # table_name = f"normal_testnode_cntxtid1_cmdid{randint(0,999)}_cmdsubid1"
        # table_name = TableName(table_name)
        # self.tables[table_name] = schema
        # return table_name
        pass  # TODO

    def get_views(self) -> List[str]:
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
    ) -> List[str]:
        pass

    def get_merge_tables(self) -> List[str]:
        pass

    def create_merge_table(self, command_id: str, table_names: List[str]):
        pass

    def get_remote_tables(self) -> List[str]:
        pass

    def create_remote_table(
        self, table_name: str, table_schema: TableSchema, native_node: _INode
    ):
        pass

    def queue_run_udf(
        self, command_id: str, func_name: str, positional_args, keyword_args
    ) -> AsyncResult:
        pass

    def get_queued_udf_result(self, async_result: AsyncResult) -> List[str]:
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
