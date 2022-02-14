import uuid as uuid

import pytest

from mipengine.datatypes import DType
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.table_data_DTOs import ColumnDataFloat
from mipengine.table_data_DTOs import ColumnDataInt
from mipengine.table_data_DTOs import ColumnDataStr
from tests.dev_env_tests.nodes_communication import get_celery_app
from tests.dev_env_tests.nodes_communication import get_celery_task_signature

local_node = get_celery_app("localnode1")
local_node_create_table = get_celery_task_signature(local_node, "create_table")
local_node_get_tables = get_celery_task_signature(local_node, "get_tables")
local_node_insert_data_to_table = get_celery_task_signature(
    local_node, "insert_data_to_table"
)

local_node_get_table_schema = get_celery_task_signature(local_node, "get_table_schema")
local_node_get_table_data = get_celery_task_signature(local_node, "get_table_data")
local_node_cleanup = get_celery_task_signature(local_node, "clean_up")


@pytest.fixture(autouse=True)
def request_id():
    return "test_tables_" + uuid.uuid4().hex + "_request"


@pytest.fixture(autouse=True)
def context_id(request_id):
    context_id = "test_tables_" + uuid.uuid4().hex

    yield context_id

    local_node_cleanup.delay(request_id=request_id, context_id=context_id).get()


def test_create_and_find_tables(request_id, context_id):
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )

    table_1_name = local_node_create_table.delay(
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    ).get()
    tables = local_node_get_tables.delay(
        request_id=request_id, context_id=context_id
    ).get()
    assert table_1_name in tables

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    local_node_insert_data_to_table.delay(
        request_id=request_id, table_name=table_1_name, values=values
    ).get()

    table_data_json = local_node_get_table_data.delay(
        request_id=request_id, table_name=table_1_name
    ).get()
    table_data = TableData.parse_raw(table_data_json)
    expected_columns = [
        ColumnDataInt(name="col1", data=[1, 2, 3]),
        ColumnDataFloat(name="col2", data=[0.1, 0.2, 0.3]),
        ColumnDataStr(name="col3", data=["test1", None, "test3"]),
    ]
    assert table_data.name == table_1_name
    assert table_data.columns == expected_columns

    table_2_name = local_node_create_table.delay(
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    ).get()
    tables = local_node_get_tables.delay(
        request_id=request_id, context_id=context_id
    ).get()
    assert table_2_name in tables

    values = [[1, 0.1, "test1"], [2, None, "None"], [3, 0.3, None]]
    local_node_insert_data_to_table.delay(
        request_id=request_id, table_name=table_2_name, values=values
    ).get()

    table_data_json = local_node_get_table_data.delay(
        request_id=request_id, table_name=table_2_name
    ).get()
    table_data = TableData.parse_raw(table_data_json)
    expected_columns = [
        ColumnDataInt(name="col1", data=[1, 2, 3]),
        ColumnDataFloat(name="col2", data=[0.1, None, 0.3]),
        ColumnDataStr(name="col3", data=["test1", "None", None]),
    ]
    assert table_data.name == table_2_name
    assert table_data.columns == expected_columns

    table_schema_json = local_node_get_table_schema.delay(
        request_id=request_id, table_name=table_2_name
    ).get()
    table_schema_1 = TableSchema.parse_raw(table_schema_json)
    assert table_schema_1 == table_schema
