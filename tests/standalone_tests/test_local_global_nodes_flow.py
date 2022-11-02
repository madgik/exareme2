import uuid

import pytest

from mipengine.datatypes import DType
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableSchema
from tests.standalone_tests.conftest import COMMON_IP
from tests.standalone_tests.conftest import MONETDB_LOCALNODE1_PORT
from tests.standalone_tests.conftest import MONETDB_LOCALNODE2_PORT
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger

create_table_task_signature = get_celery_task_signature("create_table")
insert_task_signature = get_celery_task_signature("insert_data_to_table")
create_remote_task_signature = get_celery_task_signature("create_remote_table")
get_remote_tables_task_signature = get_celery_task_signature("get_remote_tables")
create_merge_task_signature = get_celery_task_signature("create_merge_table")
get_merge_tables_task_signature = get_celery_task_signature("get_merge_tables")
get_merge_table_data_task_signature = get_celery_task_signature("get_table_data")


@pytest.fixture(autouse=True)
def request_id():
    request_id = "testflow" + uuid.uuid4().hex + "request"
    return request_id


@pytest.fixture(autouse=True)
def context_id(request_id):
    context_id = "testflow" + uuid.uuid4().hex

    yield context_id


@pytest.mark.slow
def test_create_merge_table_with_remote_tables(
    request_id,
    context_id,
    localnode1_node_service,
    localnode1_celery_app,
    localnode2_node_service,
    localnode2_celery_app,
    globalnode_node_service,
    globalnode_celery_app,
):
    schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )

    # Create local tables
    async_result = localnode1_celery_app.queue_task(
        task_signature=create_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=schema.json(),
    )

    local_node_1_table_info = TableInfo.parse_raw(
        localnode1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
    )

    async_result = localnode2_celery_app.queue_task(
        task_signature=create_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=schema.json(),
    )

    local_node_2_table_info = TableInfo.parse_raw(
        localnode2_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
    )

    # Insert data into local tables
    values = [[1, 0.1, "test1"], [2, 0.2, "test2"], [3, 0.3, "test3"]]
    async_result = localnode1_celery_app.queue_task(
        task_signature=insert_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=local_node_1_table_info.name,
        values=values,
    )
    localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    async_result = localnode2_celery_app.queue_task(
        task_signature=insert_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=local_node_2_table_info.name,
        values=values,
    )

    localnode2_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    # Create remote tables
    local_node_1_monetdb_sock_address = f"{str(COMMON_IP)}:{MONETDB_LOCALNODE1_PORT}"
    local_node_2_monetdb_sock_address = f"{str(COMMON_IP)}:{MONETDB_LOCALNODE2_PORT}"

    async_result = globalnode_celery_app.queue_task(
        task_signature=create_remote_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=local_node_1_table_info.name,
        table_schema_json=schema.json(),
        monetdb_socket_address=local_node_1_monetdb_sock_address,
    )

    globalnode_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    async_result = globalnode_celery_app.queue_task(
        task_signature=create_remote_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=local_node_2_table_info.name,
        table_schema_json=schema.json(),
        monetdb_socket_address=local_node_2_monetdb_sock_address,
    )

    globalnode_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    async_result = globalnode_celery_app.queue_task(
        task_signature=get_remote_tables_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
    )

    remote_tables = globalnode_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )
    assert local_node_1_table_info.name in remote_tables
    assert local_node_2_table_info.name in remote_tables

    # Create merge table
    async_result = globalnode_celery_app.queue_task(
        task_signature=create_merge_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        table_infos_json=[
            local_node_1_table_info.json(),
            local_node_2_table_info.json(),
        ],
    )

    merge_table_info = TableInfo.parse_raw(
        globalnode_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
    )

    # Validate merge table exists
    async_result = globalnode_celery_app.queue_task(
        task_signature=get_merge_tables_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
    )

    merge_tables = globalnode_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )
    assert merge_table_info.name in merge_tables

    # Validate merge table row count
    async_result = globalnode_celery_app.queue_task(
        task_signature=get_merge_table_data_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=merge_table_info.name,
    )

    table_data_json = globalnode_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )
    table_data = TableData.parse_raw(table_data_json)
    column_count = len(table_data.columns)
    assert column_count == 3
    row_count = len(table_data.columns[0].data)
    assert row_count == 6
