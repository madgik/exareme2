import uuid as uuid

import pytest
from pymonetdb import OperationalError

from exareme2.datatypes import DType
from exareme2.worker_communication import ColumnInfo
from exareme2.worker_communication import TableInfo
from exareme2.worker_communication import TableSchema
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.conftest import create_table_in_db
from tests.standalone_tests.conftest import get_table_data_from_db
from tests.standalone_tests.std_output_logger import StdOutputLogger
from tests.standalone_tests.workers_communication_helper import (
    get_celery_task_signature,
)

create_table_task_signature = get_celery_task_signature("create_table")
get_tables_task_signature = get_celery_task_signature("get_tables")
get_table_data_task_signature = get_celery_task_signature("get_table_data")


@pytest.fixture(autouse=True)
def request_id():
    return "testtables" + uuid.uuid4().hex + "request"


@pytest.fixture(autouse=True)
def context_id(request_id):
    context_id = "testtables" + uuid.uuid4().hex

    yield context_id


@pytest.mark.slow
def test_create_table(
    request_id,
    context_id,
    localworker1_worker_service,
    localworker1_celery_app,
    localworker1_db_cursor,
):
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )

    async_result = localworker1_celery_app.queue_task(
        task_signature=create_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    )
    table_1_info = TableInfo.parse_raw(
        localworker1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
    )

    table_values = get_table_data_from_db(localworker1_db_cursor, table_1_info.name)
    assert len(table_values) == 0


@pytest.mark.slow
def test_get_tables(
    request_id,
    context_id,
    localworker1_worker_service,
    localworker1_celery_app,
    localworker1_db_cursor,
):
    table_name = f"normal_testlocalworker1_{context_id}"
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    create_table_in_db(localworker1_db_cursor, table_name, table_schema)

    async_result = localworker1_celery_app.queue_task(
        task_signature=get_tables_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
    )
    tables = localworker1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )
    assert table_name in tables


@pytest.mark.slow
def test_get_table_data_not_working_from_unpublished_table(
    request_id,
    context_id,
    localworker1_worker_service,
    localworker1_celery_app,
    localworker1_db_cursor,
):
    table_name = f"normal_testlocalworker1_{context_id}"
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    create_table_in_db(localworker1_db_cursor, table_name, table_schema)

    async_result = localworker1_celery_app.queue_task(
        task_signature=get_table_data_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_name,
    )
    with pytest.raises(OperationalError):
        localworker1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )


@pytest.mark.slow
def test_get_table_data_works_on_published_table(
    request_id,
    context_id,
    localworker1_worker_service,
    localworker1_celery_app,
    localworker1_db_cursor,
):
    table_name = f"normal_testlocalworker1_{context_id}"
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )
    create_table_in_db(
        localworker1_db_cursor, table_name, table_schema, publish_table=True
    )

    async_result = localworker1_celery_app.queue_task(
        task_signature=get_table_data_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_name,
    )

    try:
        localworker1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
    except OperationalError:
        pytest.fail(
            "The table data should be fetched without error since the table is published."
        )
