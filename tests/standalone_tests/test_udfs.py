import json
import pickle
import uuid
from typing import Tuple

import pytest
from billiard.exceptions import TimeLimitExceeded

from exareme2 import DType
from exareme2.node.monetdb_interface.common_actions import create_table_name
from exareme2.node.tasks.udfs import _convert_output_schema
from exareme2.node.tasks.udfs import _convert_result
from exareme2.node.tasks.udfs import _make_output_table_names
from exareme2.node_tasks_DTOs import ColumnInfo
from exareme2.node_tasks_DTOs import NodeTableDTO
from exareme2.node_tasks_DTOs import NodeUDFKeyArguments
from exareme2.node_tasks_DTOs import NodeUDFPosArguments
from exareme2.node_tasks_DTOs import NodeUDFResults
from exareme2.node_tasks_DTOs import TableData
from exareme2.node_tasks_DTOs import TableInfo
from exareme2.node_tasks_DTOs import TableSchema
from exareme2.node_tasks_DTOs import TableType
from exareme2.udfgen import make_unique_func_name
from exareme2.udfgen.udfgen_DTOs import UDFGenTableResult
from tests.algorithms.orphan_udfs import get_column_rows
from tests.algorithms.orphan_udfs import local_step
from tests.algorithms.orphan_udfs import one_hundred_seconds_udf
from tests.standalone_tests.conftest import TASKS_TIMEOUT
from tests.standalone_tests.nodes_communication_helper import get_celery_task_signature
from tests.standalone_tests.std_output_logger import StdOutputLogger

command_id = "command123"
request_id = "testsmpcudfs" + str(uuid.uuid4().hex)[:10] + "request"
context_id = "testsmpcudfs" + str(uuid.uuid4().hex)[:10]


def create_table_with_one_column_and_ten_rows(
    celery_app, request_id
) -> Tuple[TableInfo, int]:
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
        context_id=request_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    )
    table_info = TableInfo.parse_raw(
        celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
        )
    )

    values = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    async_result = celery_app.queue_task(
        task_signature=insert_data_to_table_task,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_info.name,
        values=values,
    )
    celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    return table_info, 55


def create_non_existing_remote_table(celery_app, request_id) -> TableInfo:
    create_remote_table_task = get_celery_task_signature("create_remote_table")
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
        ]
    )
    table_name = "non_existing_remote_table"
    async_result = celery_app.queue_task(
        task_signature=create_remote_table_task,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=table_name,
        table_schema_json=table_schema.json(),
        monetdb_socket_address="127.0.0.1:50000",
    )
    celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )
    return TableInfo(name=table_name, schema_=table_schema, type_=TableType.REMOTE)


@pytest.mark.slow
def test_run_udf_state_and_transfer_output(
    localnode1_node_service,
    use_localnode1_database,
    localnode1_db_cursor,
    localnode1_celery_app,
):
    run_udf_task = get_celery_task_signature("run_udf")

    local_node_get_table_data = get_celery_task_signature("get_table_data")

    input_table_info, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        localnode1_celery_app, request_id
    )

    kw_args_str = NodeUDFKeyArguments(
        args={"table": NodeTableDTO(value=input_table_info)}
    ).json()

    async_result = localnode1_celery_app.queue_task(
        task_signature=run_udf_task,
        logger=StdOutputLogger(),
        command_id="1",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(local_step),
        positional_args_json=NodeUDFPosArguments(args=[]).json(),
        keyword_args_json=kw_args_str,
    )
    udf_results_str = localnode1_celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=TASKS_TIMEOUT,
    )

    results = NodeUDFResults.parse_raw(udf_results_str).results
    assert len(results) == 2

    state_result = results[0]
    assert isinstance(state_result, NodeTableDTO)

    transfer_result = results[1]
    assert isinstance(transfer_result, NodeTableDTO)

    async_result = localnode1_celery_app.queue_task(
        task_signature=local_node_get_table_data,
        logger=StdOutputLogger(),
        request_id=request_id,
        table_name=transfer_result.value.name,
    )
    transfer_table_data_json = localnode1_celery_app.get_result(
        async_result=async_result, logger=StdOutputLogger(), timeout=TASKS_TIMEOUT
    )

    table_data = TableData.parse_raw(transfer_table_data_json)
    transfer_result_str, *_ = table_data.columns[0].data
    transfer_result = json.loads(transfer_result_str)
    assert "count" in transfer_result.keys()
    assert transfer_result["count"] == 10
    assert "sum" in transfer_result.keys()
    assert transfer_result["sum"] == input_table_name_sum

    [state_result_str] = localnode1_db_cursor.execute(
        f"SELECT * FROM {state_result.value.name};"
    ).fetchone()
    state_result = pickle.loads(state_result_str)
    assert "count" in state_result.keys()
    assert state_result["count"] == 10
    assert "sum" in state_result.keys()
    assert state_result["sum"] == input_table_name_sum


@pytest.mark.slow
def test_run_udf_with_remote_state_table_passed_as_normal_table(
    localnode1_node_service,
    use_localnode1_database,
    localnode1_db_cursor,
    localnode1_celery_app,
):
    run_udf_task = get_celery_task_signature("run_udf")

    table_info = create_non_existing_remote_table(
        celery_app=localnode1_celery_app, request_id=request_id
    )
    invalid_table_info = TableInfo(
        name=table_info.name, schema_=table_info.schema_, type_=TableType.NORMAL
    )

    kw_args_str = NodeUDFKeyArguments(
        args={"remote_state_table": NodeTableDTO(value=invalid_table_info)}
    ).json()

    async_result = localnode1_celery_app.queue_task(
        task_signature=run_udf_task,
        logger=StdOutputLogger(),
        command_id="1",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(local_step),
        positional_args_json=NodeUDFPosArguments(args=[]).json(),
        keyword_args_json=kw_args_str,
    )
    with pytest.raises(ValueError) as exc_info:
        localnode1_celery_app.get_result(
            async_result=async_result,
            logger=StdOutputLogger(),
            timeout=TASKS_TIMEOUT,
        )

    assert (
        str(exc_info.value)
        == "Table: 'non_existing_remote_table' is not of type: 'NORMAL'."
    )


@pytest.mark.skip(reason="https://team-1617704806227.atlassian.net/browse/MIP-473")
@pytest.mark.slow
def test_slow_udf_exception(
    localnode1_node_service, use_localnode1_database, localnode1_celery_app
):
    run_udf_task = get_celery_task_signature("run_udf")

    input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        localnode1_celery_app, request_id
    )

    kw_args_str = NodeUDFKeyArguments(
        args={"table": NodeTableDTO(value=input_table_name)}
    ).json()

    with pytest.raises(TimeLimitExceeded):
        async_result = localnode1_celery_app.queue_task(
            task_signature=run_udf_task,
            logger=StdOutputLogger(),
            command_id="1",
            context_id=context_id,
            func_name=make_unique_func_name(one_hundred_seconds_udf),
            positional_args_json=NodeUDFPosArguments(args=[]).json(),
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
    result = _convert_output_schema(output_schema)
    assert result == [("a", DType.INT), ("b", DType.FLOAT)]


def test_create_table_name():
    table_name = create_table_name(
        table_type=TableType.NORMAL,
        node_id="node1",
        context_id="context2",
        command_id="command3",
        result_id="output4",
    )
    assert table_name == "normal_node1_context2_command3_output4"


def test_convert_table_result():
    udfgen_result = UDFGenTableResult(
        table_schema=[("a", DType.INT)], create_query="", table_name="table_name"
    )
    expected = NodeTableDTO(
        value=TableInfo(
            name="table_name",
            schema_=TableSchema(columns=[ColumnInfo(name="a", dtype=DType.INT)]),
            type_=TableType.NORMAL,
        )
    )
    result = _convert_result(udfgen_result)
    assert result == expected


def test_create_output_table_names():
    names = _make_output_table_names(
        outputlen=2, node_id="node1", context_id="context2", command_id="command3"
    )
    assert names == [
        "normal_node1_context2_command3_0",
        "normal_node1_context2_command3_1",
    ]
