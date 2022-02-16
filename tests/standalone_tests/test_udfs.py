import json
import pickle
import uuid
from typing import Tuple

import pytest
from billiard.exceptions import TimeLimitExceeded

from mipengine import DType
from mipengine.node_tasks_DTOs import (
    ColumnInfo,
    NodeTableDTO,
    TableData,
    TableSchema,
    UDFKeyArguments,
    UDFPosArguments,
    UDFResults,
)
from mipengine.udfgen import make_unique_func_name
from tests.algorithms.orphan_udfs import get_column_rows, local_step, very_slow_udf
from tests.standalone_tests.conftest import LOCALNODE1_CONFIG_FILE
from tests.standalone_tests.nodes_communication_helper import (
    get_celery_app,
    get_celery_task_signature,
    get_node_config_by_id,
)

command_id = "command123"
request_id = "test_smpc_udfs_" + str(uuid.uuid4().hex)[:10] + "_request"
context_id = "test_smpc_udfs_" + str(uuid.uuid4().hex)[:10]


@pytest.fixture(scope="session")
def localnode_1_celery_app():
    localnode1_config = get_node_config_by_id(LOCALNODE1_CONFIG_FILE)
    yield get_celery_app(localnode1_config)


def create_table_with_one_column_and_ten_rows(celery_app) -> Tuple[str, int]:
    create_table_task = get_celery_task_signature(celery_app, "create_table")
    insert_data_to_table_task = get_celery_task_signature(
        celery_app, "insert_data_to_table"
    )

    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
        ]
    )
    table_name = create_table_task.delay(
        request_id=request_id,
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    ).get()
    values = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    insert_data_to_table_task.delay(
        request_id=request_id, table_name=table_name, values=values
    ).get()

    return table_name, 55


def test_get_udf(localnode1_node_service, localnode_1_celery_app):
    get_udf_task = get_celery_task_signature(localnode_1_celery_app, "get_udf")

    fetched_udf = get_udf_task.delay(
        request_id=request_id, func_name=make_unique_func_name(get_column_rows)
    ).get()

    assert get_column_rows.__name__ in fetched_udf


def test_run_udf_relation_to_scalar(
    localnode1_node_service, use_localnode1_database, localnode_1_celery_app
):
    run_udf_task = get_celery_task_signature(localnode_1_celery_app, "run_udf")
    local_node_get_table_data = get_celery_task_signature(
        localnode_1_celery_app, "get_table_data"
    )
    input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        localnode_1_celery_app
    )
    kw_args_str = UDFKeyArguments(
        args={"table": NodeTableDTO(value=input_table_name)}
    ).json()

    udf_results_str = run_udf_task.delay(
        command_id="1",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(get_column_rows),
        positional_args_json=UDFPosArguments(args=[]).json(),
        keyword_args_json=kw_args_str,
    ).get()

    results = UDFResults.parse_raw(udf_results_str).results
    assert len(results) == 1

    result = results[0]
    assert isinstance(result, NodeTableDTO)

    table_data_json = local_node_get_table_data.delay(
        request_id=request_id, table_name=result.value
    ).get()

    table_data = TableData.parse_raw(table_data_json)

    assert table_data.columns[0].data[0] == 10


def test_run_udf_state_and_transfer_output(
    localnode_1_node_service,
    use_localnode_1_database,
    localnode_1_db_cursor,
    localnode_1_celery_app,
):
    run_udf_task = get_celery_task_signature(localnode_1_celery_app, "run_udf")
    local_node_get_table_data = get_celery_task_signature(
        localnode_1_celery_app, "get_table_data"
    )
    input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        localnode_1_celery_app
    )

    kw_args_str = UDFKeyArguments(
        args={"table": NodeTableDTO(value=input_table_name)}
    ).json()

    udf_results_str = run_udf_task.delay(
        command_id="1",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(local_step),
        positional_args_json=UDFPosArguments(args=[]).json(),
        keyword_args_json=kw_args_str,
    ).get()

    results = UDFResults.parse_raw(udf_results_str).results
    assert len(results) == 2

    state_result = results[0]
    assert isinstance(state_result, NodeTableDTO)

    transfer_result = results[1]
    assert isinstance(transfer_result, NodeTableDTO)

    transfer_table_data_json = local_node_get_table_data.delay(
        request_id=request_id, table_name=transfer_result.value
    ).get()
    table_data = TableData.parse_raw(transfer_table_data_json)
    transfer_result_str, *_ = table_data.columns[1].data
    transfer_result = json.loads(transfer_result_str)
    assert "count" in transfer_result.keys()
    assert transfer_result["count"] == 10
    assert "sum" in transfer_result.keys()
    assert transfer_result["sum"] == input_table_name_sum

    _, state_result_str = localnode_1_db_cursor.execute(
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
    localnode_1_node_service, use_localnode_1_database, localnode_1_celery_app
):
    run_udf_task = get_celery_task_signature(localnode_1_celery_app, "run_udf")
    input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        localnode_1_celery_app
    )

    kw_args_str = UDFKeyArguments(
        args={"table": NodeTableDTO(value=input_table_name)}
    ).json()

    with pytest.raises(TimeLimitExceeded):
        run_udf_task.delay(
            command_id="1",
            context_id=context_id,
            func_name=make_unique_func_name(very_slow_udf),
            positional_args_json=UDFPosArguments(args=[]).json(),
            keyword_args_json=kw_args_str,
        ).get()
