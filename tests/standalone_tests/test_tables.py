import uuid as uuid

import pytest

from exareme2.datatypes import DType
from exareme2.node_tasks_DTOs import ColumnInfo
from exareme2.node_tasks_DTOs import TableData
from exareme2.node_tasks_DTOs import TableInfo
from exareme2.node_tasks_DTOs import TableSchema
from exareme2.table_data_DTOs import ColumnDataFloat
from exareme2.table_data_DTOs import ColumnDataInt
from exareme2.table_data_DTOs import ColumnDataStr
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger

create_table_task_signature = get_celery_task_signature("create_table")
get_tables_task_signature = get_celery_task_signature("get_tables")
insert_data_to_table_task_signature = get_celery_task_signature("insert_data_to_table")
get_table_data_task_signature = get_celery_task_signature("get_table_data")


@pytest.fixture(autouse=True)
def request_id():
    return "testtables" + uuid.uuid4().hex + "request"


@pytest.fixture(autouse=True)
def context_id(request_id):
    context_id = "testtables" + uuid.uuid4().hex

    yield context_id


@pytest.mark.slow
def test_create_and_find_tables(
    request_id,
    context_id,
    localnode1_node_service,
    localnode1_celery_app,
):
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
    table_1_info = TableInfo.parse_raw(
        localnode1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
    )

    async_result = localnode1_celery_app.queue_task(
        task_signature=get_tables_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
    )
    tables = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    assert table_1_info.name in tables

    values = [[1, 0.1, "test1"], [2, 0.2, None], [3, 0.3, "test3"]]
    async_result = localnode1_celery_app.queue_task(
        task_signature=insert_data_to_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_1_info.name,
        values=values,
    )
    localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    async_result = localnode1_celery_app.queue_task(
        task_signature=get_table_data_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_1_info.name,
    )
    table_data_json = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    table_data = TableData.parse_raw(table_data_json)
    expected_columns = [
        ColumnDataInt(name="col1", data=[1, 2, 3]),
        ColumnDataFloat(name="col2", data=[0.1, 0.2, 0.3]),
        ColumnDataStr(name="col3", data=["test1", None, "test3"]),
    ]
    assert table_data.name == table_1_info.name
    assert table_data.columns == expected_columns

    async_result = localnode1_celery_app.queue_task(
        task_signature=create_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    )
    table_2_info = TableInfo.parse_raw(
        localnode1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )
    )

    async_result = localnode1_celery_app.queue_task(
        task_signature=get_tables_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
    )
    tables = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )
    assert table_2_info.name in tables

    values = [[1, 0.1, "test1"], [2, None, "None"], [3, 0.3, None]]

    async_result = localnode1_celery_app.queue_task(
        task_signature=insert_data_to_table_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_2_info.name,
        values=values,
    )
    localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    async_result = localnode1_celery_app.queue_task(
        task_signature=get_table_data_task_signature,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_2_info.name,
    )
    table_data_json = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )
    table_data = TableData.parse_raw(table_data_json)
    expected_columns = [
        ColumnDataInt(name="col1", data=[1, 2, 3]),
        ColumnDataFloat(name="col2", data=[0.1, None, 0.3]),
        ColumnDataStr(name="col3", data=["test1", "None", None]),
    ]
    assert table_data.name == table_2_info.name
    assert table_data.columns == expected_columns
    assert table_schema == table_2_info.schema_
