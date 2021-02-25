import json

import pymonetdb
import pytest

from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableSchema
from mipengine.tests.node import nodes_communication
from mipengine.tests.node.node_db_connections import get_node_db_connection
from mipengine.utils.custom_exception import IncompatibleSchemasMergeException, TableCannotBeFound

local_node_id = "local_node_1"
local_node = nodes_communication.get_celery_app(local_node_id)
local_node_create_table = nodes_communication.get_celery_create_table_signature(local_node)
local_node_create_merge_table = nodes_communication.get_celery_create_merge_table_signature(local_node)
local_node_get_merge_tables = nodes_communication.get_celery_get_merge_tables_signature(local_node)
local_node_cleanup = nodes_communication.get_celery_cleanup_signature(local_node)

context_id = "regrEssion"


@pytest.fixture(autouse=True)
def cleanup_tables():
    yield

    local_node_cleanup.delay(context_id.lower()).get()


def create_two_column_table(table_id: int):
    table_schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT")])
    table_name = local_node_create_table.delay(f"{context_id}_table_{table_id}",
                                               str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                               table_schema.to_json()
                                               ).get()
    return table_name


def create_three_column_table_with_data(table_id: int):
    table_schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])
    table_name = local_node_create_table.delay(f"{context_id}_table_{table_id}",
                                               str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                               table_schema.to_json()
                                               ).get()

    connection = get_node_db_connection(local_node_id)
    connection.cursor().execute(f"INSERT INTO {table_name} VALUES ( 1, 2.0, '3')")
    connection.commit()
    connection.close()
    return table_name


def test_create_and_get_merge_table():
    tables_to_be_merged = [create_three_column_table_with_data(1),
                           create_three_column_table_with_data(2),
                           create_three_column_table_with_data(3),
                           create_three_column_table_with_data(4)]
    merge_table_1_name = local_node_create_merge_table.delay(context_id,
                                                             str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                                             json.dumps(tables_to_be_merged)).get()
    merge_tables = local_node_get_merge_tables.delay(context_id).get()
    assert merge_table_1_name in merge_tables


def test_incompatible_schemas_merge():
    with pytest.raises(IncompatibleSchemasMergeException):
        incompatible_partition_tables = [create_three_column_table_with_data(1),
                                         create_two_column_table(2),
                                         create_two_column_table(3),
                                         create_three_column_table_with_data(4)]
        local_node_create_merge_table.delay(context_id,
                                            str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                            json.dumps(incompatible_partition_tables)
                                            ).get()


def test_table_cannot_be_found():
    with pytest.raises(TableCannotBeFound):
        not_found_tables = [create_three_column_table_with_data(1),
                            create_three_column_table_with_data(2),
                            "non_existing_table"]

        local_node_create_merge_table.delay(context_id,
                                            str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                            json.dumps(not_found_tables)
                                            ).get()