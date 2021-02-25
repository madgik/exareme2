import pymonetdb
import pytest

from mipengine.node.tasks.data_classes import ColumnInfo
from mipengine.node.tasks.data_classes import TableData
from mipengine.node.tasks.data_classes import TableSchema
from mipengine.tests.node import nodes_communication

local_node = nodes_communication.get_celery_app("local_node_1")
local_node_create_table = nodes_communication.get_celery_create_table_signature(local_node)
local_node_get_tables = nodes_communication.get_celery_get_tables_signature(local_node)
local_node_get_table_schema = nodes_communication.get_celery_get_table_schema_signature(local_node)
local_node_get_table_data = nodes_communication.get_celery_get_table_data_signature(local_node)
local_node_cleanup = nodes_communication.get_celery_cleanup_signature(local_node)

context_id_1 = "regrEssion"
context_id_2 = "HISTOGRAMS"


@pytest.fixture(autouse=True)
def cleanup_tables():
    yield

    local_node_cleanup.delay(context_id_1.lower()).get()
    local_node_cleanup.delay(context_id_2.lower()).get()


def test_create_and_find_tables():
    table_schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])

    # TODO All delay calls should be done with keyword arguments
    table_1_name = local_node_create_table.delay(context_id=context_id_1,
                                                 command_id=str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                                 schema_json=table_schema.to_json()
                                                 ).get()

    tables = local_node_get_tables.delay(context_id_1).get()
    assert table_1_name in tables

    table_2_name = local_node_create_table.delay(context_id_2,
                                                 str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                                 table_schema.to_json()
                                                 ).get()

    tables = local_node_get_tables.delay(context_id_2).get()
    assert table_2_name in tables

    table_data_json = local_node_get_table_data.delay(table_1_name).get()
    table_data = TableData.from_json(table_data_json)
    assert table_data.data == []
    assert table_data.schema == table_schema

    table_schema_json = local_node_get_table_schema.delay(table_2_name).get()
    table_schema_1 = TableSchema.from_json(table_schema_json)
    assert table_schema_1 == table_schema
