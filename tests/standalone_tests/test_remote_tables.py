import uuid

import pytest
from sqlalchemy.exc import OperationalError

from exareme2.datatypes import DType
from exareme2.worker_communication import ColumnInfo
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from exareme2.worker_communication import TableType
from tests.standalone_tests.conftest import COMMON_IP
from tests.standalone_tests.conftest import MONETDB_LOCALWORKER1_PORT
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.conftest import create_table_in_db
from tests.standalone_tests.conftest import get_table_data_from_db
from tests.standalone_tests.conftest import insert_data_to_db
from tests.standalone_tests.std_output_logger import StdOutputLogger
from tests.standalone_tests.workers_communication_helper import (
    get_celery_task_signature,
)

create_remote_table_task_signature = get_celery_task_signature("create_remote_table")


@pytest.fixture(autouse=True)
def request_id():
    return "testremotetables" + uuid.uuid4().hex + "request"


@pytest.fixture(autouse=True)
def context_id(request_id):
    context_id = "testremotetables" + uuid.uuid4().hex

    yield context_id


@pytest.mark.slow
def test_remote_table_properly_mirrors_data(
    request_id,
    context_id,
    localworker1_db_cursor,
    localworker1_worker_service,
    localworker1_celery_app,
    globalworker_db_cursor,
    globalworker_worker_service,
    globalworker_celery_app,
):
    local_worker_monetdb_sock_address = f"{str(COMMON_IP)}:{MONETDB_LOCALWORKER1_PORT}"

    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    initial_table_values = [[1, 0.1, "test1"], [2, 0.2, "test2"], [3, 0.3, "test3"]]
    table_info = TableInfo(
        name=f"normal_testlocalworker1_{context_id}",
        schema_=table_schema,
        type_=TableType.NORMAL,
    )
    create_table_in_db(
        localworker1_db_cursor,
        table_info.name,
        table_schema,
        True,
    )
    insert_data_to_db(table_info.name, initial_table_values, localworker1_db_cursor)

    async_result = globalworker_celery_app.queue_task(
        task_signature=create_remote_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_info.name,
        table_schema_json=table_schema.json(),
        monetdb_socket_address=local_worker_monetdb_sock_address,
    )
    globalworker_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    table_values = get_table_data_from_db(globalworker_db_cursor, table_info.name)

    # Validate same size table result
    assert len(table_values[0]) == len(initial_table_values[0])
    assert len(table_values) == len(initial_table_values)


@pytest.mark.slow
def test_remote_table_error_on_non_published_table(
    request_id,
    context_id,
    localworker1_db_cursor,
    localworker1_worker_service,
    localworker1_celery_app,
    globalworker_db_cursor,
    globalworker_worker_service,
    globalworker_celery_app,
):
    """
    In this test we check if the "public" user is used when creating a remote table.
    If the initial table, on which the remote table is created, is not published in the "public" user then
    the remote table does not work and returns an error.

    The error returned is an OperationalError with message
    "Exception occurred in the remote server, please check the log there".
    """
    local_worker_monetdb_sock_address = f"{str(COMMON_IP)}:{MONETDB_LOCALWORKER1_PORT}"

    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    table_info = TableInfo(
        name=f"normal_testlocalworker1_{context_id}",
        schema_=table_schema,
        type_=TableType.NORMAL,
    )
    create_table_in_db(
        localworker1_db_cursor,
        table_info.name,
        table_schema,
        False,
    )

    async_result = globalworker_celery_app.queue_task(
        task_signature=create_remote_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_info.name,
        table_schema_json=table_schema.json(),
        monetdb_socket_address=local_worker_monetdb_sock_address,
    )
    globalworker_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    with pytest.raises(OperationalError):
        get_table_data_from_db(globalworker_db_cursor, table_info.name)
