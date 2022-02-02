import uuid

import pytest

from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.datatypes import DType
from mipengine.node_tasks_DTOs import TableSchema
from tests.dev_env_tests.nodes_communication import get_celery_task_signature
from tests.dev_env_tests.nodes_communication import get_celery_app
from tests.dev_env_tests.nodes_communication import get_node_config_by_id

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
    context_id = "test_remote_tables_" + uuid.uuid4().hex

    yield context_id

    local_node_cleanup.delay(context_id=context_id.lower()).get()
    global_node_cleanup.delay(context_id=context_id.lower()).get()


def test_create_and_get_remote_table(context_id):
    node_config = get_node_config_by_id(local_node_id)

    local_node_monetdb_sock_address = (
        f"{str(node_config.monetdb.ip)}:{node_config.monetdb.port}"
    )

    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )

    table_name = local_node_create_table.delay(
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    ).get()

    global_node_create_remote_table.delay(
        context_id=context_id,
        table_name=table_name,
        table_schema_json=table_schema.json(),
        monetdb_socket_address=local_node_monetdb_sock_address,
    ).get()
    remote_tables = global_node_get_remote_tables.delay(context_id=context_id).get()
    assert table_name in remote_tables
