import pytest
import uuid as uuid

from mipengine.common.node_tasks_DTOs import ColumnInfo
from mipengine.common.node_tasks_DTOs import TableData
from mipengine.common.node_tasks_DTOs import TableSchema
from tests.integration_tests.nodes_communication import get_celery_app
from tests.integration_tests.nodes_communication import get_celery_task_signature

local_node = get_celery_app("localnode1")
local_node_create_table = get_celery_task_signature(local_node, "create_table")
local_node_get_tables = get_celery_task_signature(local_node, "get_tables")
local_node_insert_data_to_table = get_celery_task_signature(
    local_node, "insert_data_to_table"
)

local_node_get_table_schema = get_celery_task_signature(local_node, "get_table_schema")
local_node_get_table_data = get_celery_task_signature(local_node, "get_table_data")
local_node_cleanup = get_celery_task_signature(local_node, "clean_up")

context_id_1 = "regrEssion"
context_id_2 = "HISTOGRAMS"


@pytest.fixture(autouse=True)
def context_id():
    context_id = "test_tables_" + str(uuid.uuid4()).replace("-", "")

    yield context_id

    local_node_cleanup.delay(context_id=context_id).get()


def test_create_and_find_tables(context_id):
    table_schema = TableSchema(
        [
            ColumnInfo("col1", "int"),
            ColumnInfo("col2", "real"),
            ColumnInfo("col3", "text"),
        ]
    )

    table_1_name = local_node_create_table.delay(
        context_id=context_id,
        command_id=str(uuid.uuid4()).replace("-", ""),
        schema_json=table_schema.to_json(),
    ).get()
    tables = local_node_get_tables.delay(context_id=context_id).get()
    assert table_1_name in tables

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    local_node_insert_data_to_table.delay(table_name=table_1_name, values=values).get()

    table_data_json = local_node_get_table_data.delay(table_name=table_1_name).get()
    table_data = TableData.from_json(table_data_json)
    assert table_data.data == values
    assert table_data.schema == table_schema

    table_2_name = local_node_create_table.delay(
        context_id=context_id,
        command_id=str(uuid.uuid4()).replace("-", ""),
        schema_json=table_schema.to_json(),
    ).get()
    tables = local_node_get_tables.delay(context_id=context_id).get()
    assert table_2_name in tables

    values = [[1, 0.1, "test1"], [2, None, "None"], [3, 0.3, None]]
    local_node_insert_data_to_table.delay(table_name=table_2_name, values=values).get()

    table_data_json = local_node_get_table_data.delay(table_name=table_2_name).get()
    table_data = TableData.from_json(table_data_json)
    assert table_data.data == values
    assert table_data.schema == table_schema

    table_schema_json = local_node_get_table_schema.delay(table_name=table_2_name).get()
    table_schema_1 = TableSchema.from_json(table_schema_json)
    assert table_schema_1 == table_schema
