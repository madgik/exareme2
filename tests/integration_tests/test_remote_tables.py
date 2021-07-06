import uuid

import pytest

from mipengine.common.node_tasks_DTOs import ColumnInfo
from mipengine.common.node_tasks_DTOs import TableInfo
from mipengine.common.node_tasks_DTOs import TableSchema
from mipengine.common.node_catalog import node_catalog
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

global_node_id = "globalnode"
local_node_id = "localnode1"
global_node = get_celery_app(global_node_id)
local_node = get_celery_app(local_node_id)
local_node_create_table = get_celery_task_signature(local_node, "create_table")
global_node_create_remote_table = get_celery_task_signature(
    global_node, "create_remote_table"
)
global_node_get_remote_tables = get_celery_task_signature(
    global_node, "get_remote_tables"
)
local_node_cleanup = get_celery_task_signature(local_node, "clean_up")
global_node_cleanup = get_celery_task_signature(global_node, "clean_up")


@pytest.fixture(autouse=True)
def context_id():
    context_id = "test_remote_tables_" + str(uuid.uuid4()).replace("-", "")

    yield context_id

    local_node_cleanup.delay(context_id=context_id.lower()).get()
    global_node_cleanup.delay(context_id=context_id.lower()).get()


def test_create_and_get_remote_table(context_id):
    nrclient = NodeRegistryClient()
    db = nrclient.get_db_by_node_id(local_node_id)

    local_node_monetdb_sock_address = f"{str(db.ip)}:{db.port}"

    table_schema = TableSchema(
        [
            ColumnInfo("col1", "int"),
            ColumnInfo("col2", "real"),
            ColumnInfo("col3", "text"),
        ]
    )

    table_name = local_node_create_table.delay(
        context_id=context_id,
        command_id=str(uuid.uuid1()).replace("-", ""),
        schema_json=table_schema.to_json(),
    ).get()

    table_info = TableInfo(table_name, table_schema)

    global_node_create_remote_table.delay(
        table_info_json=table_info.to_json(),
        monetdb_socket_address=local_node_monetdb_sock_address,
    ).get()
    remote_tables = global_node_get_remote_tables.delay(context_id=context_id).get()
    assert table_name in remote_tables
