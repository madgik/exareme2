import uuid

import pytest

from mipengine.common.node_exceptions import IncompatibleSchemasMergeException
from mipengine.common.node_exceptions import TablesNotFound
from mipengine.common.node_tasks_DTOs import ColumnInfo
from mipengine.common.node_tasks_DTOs import TableSchema
from tests.integration_tests import nodes_communication
from tests.integration_tests.node_db_connections import get_node_db_connection

local_node_id = "localnode1"
local_node = nodes_communication.get_celery_app(local_node_id)
local_node_create_table = nodes_communication.get_celery_create_table_signature(
    local_node
)
local_node_create_merge_table = (
    nodes_communication.get_celery_create_merge_table_signature(local_node)
)
local_node_get_merge_tables = nodes_communication.get_celery_get_merge_tables_signature(
    local_node
)
local_node_cleanup = nodes_communication.get_celery_cleanup_signature(local_node)


@pytest.fixture(autouse=True)
def context_id():
    context_id = "test_merge_tables_" + str(uuid.uuid4()).replace("-", "")

    yield context_id

    local_node_cleanup.delay(context_id=context_id.lower()).get()


def create_two_column_table(context_id, table_id: int):
    table_schema = TableSchema([ColumnInfo("col1", "int"), ColumnInfo("col2", "real")])
    table_name = local_node_create_table.delay(
        context_id=f"{context_id}_table_{table_id}",
        command_id=str(uuid.uuid1()).replace("-", ""),
        schema_json=table_schema.to_json(),
    ).get()
    return table_name


def create_three_column_table_with_data(context_id, table_id: int):
    table_schema = TableSchema(
        [
            ColumnInfo("col1", "int"),
            ColumnInfo("col2", "real"),
            ColumnInfo("col3", "text"),
        ]
    )
    table_name = local_node_create_table.delay(
        context_id=f"{context_id}_table_{table_id}",
        command_id=str(uuid.uuid1()).replace("-", ""),
        schema_json=table_schema.to_json(),
    ).get()

    connection = get_node_db_connection(local_node_id)
    connection.cursor().execute(f"INSERT INTO {table_name} VALUES ( 1, 2.0, '3')")
    connection.commit()
    connection.close()
    return table_name


def test_create_and_get_merge_table(context_id):
    tables_to_be_merged = [
        create_three_column_table_with_data(context_id, 1),
        create_three_column_table_with_data(context_id, 2),
        create_three_column_table_with_data(context_id, 3),
        create_three_column_table_with_data(context_id, 4),
    ]
    merge_table_1_name = local_node_create_merge_table.delay(
        context_id=context_id,
        command_id=str(uuid.uuid1()).replace("-", ""),
        table_names=tables_to_be_merged,
    ).get()
    merge_tables = local_node_get_merge_tables.delay(context_id=context_id).get()
    assert merge_table_1_name in merge_tables


def test_incompatible_schemas_merge(context_id):
    with pytest.raises(IncompatibleSchemasMergeException):
        incompatible_partition_tables = [
            create_three_column_table_with_data(context_id, 1),
            create_two_column_table(context_id, 2),
            create_two_column_table(context_id, 3),
            create_three_column_table_with_data(context_id, 4),
        ]
        local_node_create_merge_table.delay(
            context_id=context_id,
            command_id=str(uuid.uuid1()).replace("-", ""),
            table_names=incompatible_partition_tables,
        ).get()


def test_table_cannot_be_found(context_id):
    with pytest.raises(TablesNotFound):
        not_found_tables = [
            create_three_column_table_with_data(context_id, 1),
            create_three_column_table_with_data(context_id, 2),
            "non_existing_table",
        ]

        local_node_create_merge_table.delay(
            context_id=context_id,
            command_id=str(uuid.uuid1()).replace("-", ""),
            table_names=not_found_tables,
        ).get()
