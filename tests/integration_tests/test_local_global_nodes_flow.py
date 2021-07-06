import uuid
import pytest
from mipengine.common.node_tasks_DTOs import ColumnInfo, TableData
from mipengine.common.node_tasks_DTOs import TableInfo
from mipengine.common.node_tasks_DTOs import TableSchema
from tests.integration_tests.nodes_communication import get_celery_task_signature
from tests.integration_tests.nodes_communication import get_celery_app
from mipengine.node_registry import (
    NodeRegistryClient,
    Pathologies,
    Pathology,
    NodeRole,
    NodeParams,
    DBParams,
)

local_node_1_id = "localnode1"
local_node_2_id = "localnode2"
global_node_id = "globalnode"
local_node_1 = get_celery_app(local_node_1_id)
local_node_2 = get_celery_app(local_node_2_id)
global_node = get_celery_app(global_node_id)

local_node_1_create_table = get_celery_task_signature(local_node_1, "create_table")
local_node_2_create_table = get_celery_task_signature(local_node_2, "create_table")
local_node_1_insert_data_to_table = get_celery_task_signature(
    local_node_1, "insert_data_to_table"
)
local_node_2_insert_data_to_table = get_celery_task_signature(
    local_node_2, "insert_data_to_table"
)
global_node_create_remote_table = get_celery_task_signature(
    global_node, "create_remote_table"
)
global_node_get_remote_tables = get_celery_task_signature(
    global_node, "get_remote_tables"
)
global_node_create_merge_table = get_celery_task_signature(
    global_node, "create_merge_table"
)
global_node_get_merge_tables = get_celery_task_signature(
    global_node, "get_merge_tables"
)
global_node_get_merge_table_data = get_celery_task_signature(
    global_node, "get_table_data"
)

clean_up_node1 = get_celery_task_signature(local_node_1, "clean_up")

clean_up_node2 = get_celery_task_signature(local_node_2, "clean_up")
clean_up_global = get_celery_task_signature(global_node, "clean_up")


@pytest.fixture(autouse=True)
def context_id():
    context_id = "test_flow_" + str(uuid.uuid4()).replace("-", "")

    yield context_id

    clean_up_node1.delay(context_id=context_id.lower()).get()
    clean_up_node2.delay(context_id=context_id.lower()).get()
    clean_up_global.delay(context_id=context_id.lower()).get()


def test_create_merge_table_with_remote_tables(context_id):
    nrclient = NodeRegistryClient()
    local_node_1_db = nrclient.get_db_by_node_id(local_node_1_id)
    local_node_2_db = nrclient.get_db_by_node_id(local_node_2_id)

    schema = TableSchema(
        [
            ColumnInfo("col1", "int"),
            ColumnInfo("col2", "real"),
            ColumnInfo("col3", "text"),
        ]
    )

    # Create local tables
    local_node_1_table_name = local_node_1_create_table.delay(
        context_id=context_id,
        command_id=str(uuid.uuid1()).replace("-", ""),
        schema_json=schema.to_json(),
    ).get()
    local_node_2_table_name = local_node_2_create_table.delay(
        context_id=context_id,
        command_id=str(uuid.uuid1()).replace("-", ""),
        schema_json=schema.to_json(),
    ).get()
    # Insert data into local tables
    values = [[1, 0.1, "test1"], [2, 0.2, "test2"], [3, 0.3, "test3"]]
    local_node_1_insert_data_to_table.delay(
        table_name=local_node_1_table_name, values=values
    ).get()
    local_node_2_insert_data_to_table.delay(
        table_name=local_node_2_table_name, values=values
    ).get()

    # Create remote tables
    table_info_local_1 = TableInfo(local_node_1_table_name, schema)
    table_info_local_2 = TableInfo(local_node_2_table_name, schema)
    local_node_1_monetdb_sock_address = (
        f"{str(local_node_1_db.ip)}:{local_node_1_db.port}"
    )
    local_node_2_monetdb_sock_address = (
        f"{str(local_node_2_db.ip)}:{local_node_2_db.port}"
    )
    global_node_create_remote_table.delay(
        table_info_json=table_info_local_1.to_json(),
        monetdb_socket_address=local_node_1_monetdb_sock_address,
    ).get()
    global_node_create_remote_table.delay(
        table_info_json=table_info_local_2.to_json(),
        monetdb_socket_address=local_node_2_monetdb_sock_address,
    ).get()
    remote_tables = global_node_get_remote_tables.delay(context_id=context_id).get()
    assert local_node_1_table_name in remote_tables
    assert local_node_2_table_name in remote_tables

    # Create merge table
    merge_table_name = global_node_create_merge_table.delay(
        context_id=context_id,
        command_id=str(uuid.uuid1()).replace("-", ""),
        table_names=remote_tables,
    ).get()

    # Validate merge table exists
    merge_tables = global_node_get_merge_tables.delay(context_id=context_id).get()
    assert merge_table_name in merge_tables

    # Validate merge table row count
    table_data_json = global_node_get_merge_table_data.delay(
        table_name=merge_table_name
    ).get()
    table_data = TableData.from_json(table_data_json)
    row_count = len(table_data.data)
    assert row_count == 6
