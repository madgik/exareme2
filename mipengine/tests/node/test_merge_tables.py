import uuid

import pytest

from mipengine.common.node_exceptions import IncompatibleSchemasMergeException
from mipengine.common.node_exceptions import TableCannotBeFound
from mipengine.common.node_tasks_DTOs import ColumnInfo
from mipengine.common.node_tasks_DTOs import TableSchema
from mipengine.tests.node import nodes_communication
from mipengine.tests.node.node_db_connections import get_node_db_connection

local_node_id = "localnode1"
local_node = nodes_communication.get_celery_app(local_node_id)
local_node_get_tables = nodes_communication.get_celery_get_tables_signature(local_node)
local_node_create_table = nodes_communication.get_celery_create_table_signature(local_node)
local_node_create_merge_table = nodes_communication.get_celery_create_merge_table_signature(local_node)
local_node_get_merge_tables = nodes_communication.get_celery_get_merge_tables_signature(local_node)
local_node_cleanup = nodes_communication.get_celery_cleanup_signature(local_node)


def create_two_column_table(context_id: str, table_id: int):
    table_schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT")])
    table_name = local_node_create_table.delay(context_id=f"{context_id}_table_{table_id}",
                                               command_id=str(uuid.uuid4()).replace("-", ""),
                                               schema_json=table_schema.to_json()).get()
    tables = local_node_get_tables.delay(context_id=f"{context_id}_table_{table_id}").get()
    assert table_name in tables
    return table_name


def create_three_column_table_with_data(context_id: str, table_id: int):
    table_schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])
    table_name = local_node_create_table.delay(context_id=f"{context_id}_table_{table_id}",
                                               command_id=str(uuid.uuid4()).replace("-", ""),
                                               schema_json=table_schema.to_json()).get()

    connection = get_node_db_connection(local_node_id)
    cursor = connection.cursor()
    cursor.execute(f"INSERT INTO {table_name} VALUES ( 1, 2.0, '3')")
    cursor.close()
    connection.close()
    tables = local_node_get_tables.delay(context_id=f"{context_id}_table_{table_id}").get()
    assert table_name in tables

    return table_name


@pytest.fixture(autouse=True)
def cleanup_context_id():
    context_id = "test_merge_tables_" + str(uuid.uuid4()).replace("-", "")

    yield context_id

    local_node_cleanup.delay(context_id=context_id).get()


def test_create_and_get_merge_table(cleanup_context_id):
    tables_to_be_merged = [create_three_column_table_with_data(cleanup_context_id, 1),
                           create_three_column_table_with_data(cleanup_context_id, 2),
                           create_three_column_table_with_data(cleanup_context_id, 3),
                           create_three_column_table_with_data(cleanup_context_id, 4)]
    merge_table_1_name = local_node_create_merge_table.delay(context_id=cleanup_context_id,
                                                             command_id=str(uuid.uuid4()).replace("-", ""),
                                                             table_names=tables_to_be_merged).get()
    merge_tables = local_node_get_merge_tables.delay(context_id=cleanup_context_id).get()
    assert merge_table_1_name in merge_tables


def test_incompatible_schemas_merge(cleanup_context_id):
    with pytest.raises(IncompatibleSchemasMergeException):
        incompatible_partition_tables = [create_three_column_table_with_data(cleanup_context_id, 1),
                                         create_two_column_table(cleanup_context_id, 2),
                                         create_two_column_table(cleanup_context_id, 3),
                                         create_three_column_table_with_data(cleanup_context_id, 4)]
        local_node_create_merge_table.delay(context_id=cleanup_context_id,
                                            command_id=str(uuid.uuid4()).replace("-", ""),
                                            table_names=incompatible_partition_tables).get()


def test_table_cannot_be_found(cleanup_context_id):
    with pytest.raises(TableCannotBeFound):
        not_found_tables = [create_three_column_table_with_data(cleanup_context_id, 1),
                            create_three_column_table_with_data(cleanup_context_id, 2),
                            "non_existing_table"]

        local_node_create_merge_table.delay(context_id=cleanup_context_id,
                                            command_id=str(uuid.uuid4()).replace("-", ""),
                                            table_names=not_found_tables).get()
