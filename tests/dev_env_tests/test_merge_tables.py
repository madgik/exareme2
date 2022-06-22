import uuid

import pytest

from mipengine.datatypes import DType
from mipengine.node_exceptions import IncompatibleSchemasMergeException
from mipengine.node_exceptions import TablesNotFound
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableSchema
from tests.dev_env_tests.nodes_communication import get_celery_app
from tests.dev_env_tests.nodes_communication import get_celery_task_signature

local_node_id = "localnode1"
local_node = get_celery_app(local_node_id)
local_node_create_table = get_celery_task_signature(local_node, "create_table")
local_node_create_merge_table = get_celery_task_signature(
    local_node, "create_merge_table"
)
local_node_insert_data_to_table = get_celery_task_signature(
    local_node, "insert_data_to_table"
)
local_node_get_merge_tables = get_celery_task_signature(local_node, "get_merge_tables")
local_node_cleanup = get_celery_task_signature(local_node, "clean_up")


@pytest.fixture(autouse=True)
def request_id():
    return "testmergetables" + uuid.uuid4().hex + "request"


@pytest.fixture(autouse=True)
def context_id(request_id):
    context_id = "testmergetables" + uuid.uuid4().hex

    yield context_id

    local_node_cleanup.delay(request_id=request_id, context_id=context_id.lower()).get()


def create_two_column_table(request_id, context_id, table_id: int):
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
        ]
    )
    table_name = local_node_create_table.delay(
        request_id=request_id,
        context_id=f"{context_id}table{table_id}",
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    ).get()
    return table_name


def create_three_column_table_with_data(request_id, context_id, table_id: int):
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    table_name = local_node_create_table.delay(
        request_id=request_id,
        context_id=f"{context_id}table{table_id}",
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    ).get()

    values = [[1, 0.1, "test1"], [2, 0.2, "test2"], [3, 0.3, "test3"]]
    local_node_insert_data_to_table.delay(
        request_id=request_id, table_name=table_name, values=values
    ).get()

    return table_name


def test_create_and_get_merge_table(request_id, context_id):
    tables_to_be_merged = [
        create_three_column_table_with_data(request_id, context_id, 1),
        create_three_column_table_with_data(request_id, context_id, 2),
        create_three_column_table_with_data(request_id, context_id, 3),
        create_three_column_table_with_data(request_id, context_id, 4),
    ]
    merge_table_1_name = local_node_create_merge_table.delay(
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        table_names=tables_to_be_merged,
    ).get()
    merge_tables = local_node_get_merge_tables.delay(
        request_id=request_id, context_id=context_id
    ).get()
    assert merge_table_1_name in merge_tables


def test_incompatible_schemas_merge(request_id, context_id):
    with pytest.raises(IncompatibleSchemasMergeException):
        incompatible_partition_tables = [
            create_three_column_table_with_data(request_id, context_id, 1),
            create_two_column_table(request_id, context_id, 2),
            create_two_column_table(request_id, context_id, 3),
            create_three_column_table_with_data(request_id, context_id, 4),
        ]
        local_node_create_merge_table.delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            table_names=incompatible_partition_tables,
        ).get()


def test_table_cannot_be_found(request_id, context_id):
    with pytest.raises(TablesNotFound):
        not_found_tables = [
            create_three_column_table_with_data(request_id, context_id, 1),
            create_three_column_table_with_data(request_id, context_id, 2),
            "non_existing_table",
        ]

        local_node_create_merge_table.delay(
            request_id=request_id,
            context_id=context_id,
            command_id=uuid.uuid4().hex,
            table_names=not_found_tables,
        ).get()
