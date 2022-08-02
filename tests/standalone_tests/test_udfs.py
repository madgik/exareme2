import json
import pickle
import uuid
from typing import Tuple

import pytest
from billiard.exceptions import TimeLimitExceeded

from mipengine import DType
from mipengine.node.tasks.udfs import _parse_output_schema
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFKeyArguments
from mipengine.node_tasks_DTOs import UDFPosArguments
from mipengine.node_tasks_DTOs import UDFResults
from mipengine.udfgen import make_unique_func_name
from tests.algorithms.orphan_udfs import get_column_rows
from tests.algorithms.orphan_udfs import local_step
from tests.algorithms.orphan_udfs import very_slow_udf
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger

command_id = "command123"
request_id = "testsmpcudfs" + str(uuid.uuid4().hex)[:10] + "request"
context_id = "testsmpcudfs" + str(uuid.uuid4().hex)[:10]


def create_table_with_one_column_and_ten_rows(celery_app) -> Tuple[str, int]:
    create_table_task = get_celery_task_signature("create_table")
    insert_data_to_table_task = get_celery_task_signature("insert_data_to_table")

    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
        ]
    )
    async_result = celery_app.queue_task(
        task_signature=create_table_task,
        logger=StdOutputLogger(),
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    )
    table_name = celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    values = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    async_result = celery_app.queue_task(
        task_signature=insert_data_to_table_task,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_name,
        values=values,
    )
    celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    return table_name, 55


@pytest.mark.slow
def test_get_udf(localnode1_node_service, localnode1_celery_app):
    get_udf_task = get_celery_task_signature("get_udf")

    async_result = localnode1_celery_app.queue_task(
        task_signature=get_udf_task,
        logger=StdOutputLogger(),
        request_id=request_id,
        func_name=make_unique_func_name(get_column_rows),
    )
    fetched_udf = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    assert get_column_rows.__name__ in fetched_udf


@pytest.mark.slow
def test_run_udf_relation_to_scalar(
    localnode1_node_service, use_localnode1_database, localnode1_celery_app
):
    run_udf_task = get_celery_task_signature("run_udf")

    local_node_get_table_data = get_celery_task_signature("get_table_data")

    input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        localnode1_celery_app
    )
    kw_args_str = UDFKeyArguments(
        args={"table": NodeTableDTO(value=input_table_name)}
    ).json()

    async_result = localnode1_celery_app.queue_task(
        task_signature=run_udf_task,
        logger=StdOutputLogger(),
        command_id="1",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(get_column_rows),
        positional_args_json=UDFPosArguments(args=[]).json(),
        keyword_args_json=kw_args_str,
    )
    udf_results_str = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    results = UDFResults.parse_raw(udf_results_str).results
    assert len(results) == 1

    result = results[0]
    assert isinstance(result, NodeTableDTO)

    async_result = localnode1_celery_app.queue_task(
        task_signature=local_node_get_table_data,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=result.value,
    )
    table_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    table_data = TableData.parse_raw(table_data_json)

    assert table_data.columns[0].data[0] == 10


@pytest.mark.slow
def test_run_udf_state_and_transfer_output(
    localnode1_node_service,
    use_localnode1_database,
    localnode1_db_cursor,
    localnode1_celery_app,
):
    run_udf_task = get_celery_task_signature("run_udf")

    local_node_get_table_data = get_celery_task_signature("get_table_data")

    input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        localnode1_celery_app
    )

    kw_args_str = UDFKeyArguments(
        args={"table": NodeTableDTO(value=input_table_name)}
    ).json()

    async_result = localnode1_celery_app.queue_task(
        task_signature=run_udf_task,
        logger=StdOutputLogger(),
        command_id="1",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(local_step),
        positional_args_json=UDFPosArguments(args=[]).json(),
        keyword_args_json=kw_args_str,
    )
    udf_results_str = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    results = UDFResults.parse_raw(udf_results_str).results
    assert len(results) == 2

    state_result = results[0]
    assert isinstance(state_result, NodeTableDTO)

    transfer_result = results[1]
    assert isinstance(transfer_result, NodeTableDTO)

    async_result = localnode1_celery_app.queue_task(
        task_signature=local_node_get_table_data,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=transfer_result.value,
    )
    transfer_table_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    table_data = TableData.parse_raw(transfer_table_data_json)
    transfer_result_str, *_ = table_data.columns[1].data
    transfer_result = json.loads(transfer_result_str)
    assert "count" in transfer_result.keys()
    assert transfer_result["count"] == 10
    assert "sum" in transfer_result.keys()
    assert transfer_result["sum"] == input_table_name_sum

    _, state_result_str = localnode1_db_cursor.execute(
        f"SELECT * FROM {state_result.value};"
    ).fetchone()
    state_result = pickle.loads(state_result_str)
    assert "count" in state_result.keys()
    assert state_result["count"] == 10
    assert "sum" in state_result.keys()
    assert state_result["sum"] == input_table_name_sum


@pytest.mark.skip(reason="https://team-1617704806227.atlassian.net/browse/MIP-473")
@pytest.mark.slow
def test_slow_udf_exception(
    localnode1_node_service, use_localnode1_database, localnode1_celery_app
):
    run_udf_task = get_celery_task_signature("run_udf")

    input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        localnode1_celery_app
    )

    kw_args_str = UDFKeyArguments(
        args={"table": NodeTableDTO(value=input_table_name)}
    ).json()

    with pytest.raises(TimeLimitExceeded):
        async_result = localnode1_celery_app.queue_task(
            task_signature=run_udf_task,
            logger=StdOutputLogger(),
            command_id="1",
            context_id=context_id,
            func_name=make_unique_func_name(very_slow_udf),
            positional_args_json=UDFPosArguments(args=[]).json(),
            keyword_args_json=kw_args_str,
        )
        localnode1_celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )


def test_parse_output_schema():
    output_schema = TableSchema(
        columns=[
            ColumnInfo(name="a", dtype=DType.INT),
            ColumnInfo(name="b", dtype=DType.FLOAT),
        ]
    ).json()
    result = _parse_output_schema(output_schema)
    assert result == [("a", DType.INT), ("b", DType.FLOAT)]
