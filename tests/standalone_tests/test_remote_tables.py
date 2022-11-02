import uuid

import pytest

from mipengine.datatypes import DType
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableInfo
from mipengine.node_tasks_DTOs import TableSchema
from tests.standalone_tests.conftest import COMMON_IP
from tests.standalone_tests.conftest import MONETDB_LOCALNODE1_PORT
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger

create_table_task_signature = get_celery_task_signature("create_table")
create_remote_table_task_signature = get_celery_task_signature("create_remote_table")
get_remote_tables_task_signature = get_celery_task_signature("get_remote_tables")


@pytest.fixture(autouse=True)
def request_id():
    return "testremotetables" + uuid.uuid4().hex + "request"


@pytest.fixture(autouse=True)
def context_id(request_id):
    context_id = "testremotetables" + uuid.uuid4().hex

    yield context_id


@pytest.mark.slow
def test_create_and_get_remote_table(
    request_id,
    context_id,
    localnode1_node_service,
    localnode1_celery_app,
    globalnode_node_service,
    globalnode_celery_app,
):

    local_node_monetdb_sock_address = f"{str(COMMON_IP)}:{MONETDB_LOCALNODE1_PORT}"

    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )

    async_result = localnode1_celery_app.queue_task(
        task_signature=create_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    )
    table_info = TableInfo.parse_raw(
        localnode1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
    )

    async_result = globalnode_celery_app.queue_task(
        task_signature=create_remote_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_info.name,
        table_schema_json=table_schema.json(),
        monetdb_socket_address=local_node_monetdb_sock_address,
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

    assert table_info.name in remote_tables
