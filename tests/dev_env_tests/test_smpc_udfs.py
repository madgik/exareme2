import json
import uuid

import pytest
from typing import Tuple

from mipengine import DType
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFArgument
from mipengine.node_tasks_DTOs import UDFArgumentKind
from mipengine.udfgen import make_unique_func_name
from tests.dev_env_tests import nodes_communication


local_node_id = "localnode1"
command_id = "command123"
localnode_app = nodes_communication.get_celery_app(local_node_id)
local_node_run_udf = nodes_communication.get_celery_task_signature(
    localnode_app, "run_udf"
)
local_node_create_table = nodes_communication.get_celery_task_signature(
    localnode_app, "create_table"
)
local_node_insert_data_to_table = nodes_communication.get_celery_task_signature(
    localnode_app, "insert_data_to_table"
)
local_node_get_table_data = nodes_communication.get_celery_task_signature(
    localnode_app, "get_table_data"
)
local_node_cleanup = nodes_communication.get_celery_task_signature(
    localnode_app, "clean_up"
)


@pytest.fixture()
def context_id():
    context_id = "test_smpc_udfs_" + str(uuid.uuid4().hex)[:10]

    yield context_id

    local_node_cleanup.delay(context_id=context_id).get()


@pytest.fixture()
def table_with_one_column_and_ten_rows(context_id) -> Tuple[str, int]:
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
        ]
    )
    table_name = local_node_create_table.delay(
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    ).get()
    values = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    local_node_insert_data_to_table.delay(table_name=table_name, values=values).get()

    return table_name, 55


@pytest.fixture()
def table_with_secure_transfer_results(context_id) -> Tuple[str, int]:
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="secure_transfer", dtype=DType.JSON),
        ]
    )
    table_name = local_node_create_table.delay(
        context_id=context_id,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    ).get()

    secure_transfer_1_value = 100
    secure_transfer_2_value = 11

    secure_transfer_1 = {
        "sum": {"data": secure_transfer_1_value, "type": "int", "operation": "addition"}
    }
    secure_transfer_2 = {
        "sum": {"data": secure_transfer_2_value, "type": "int", "operation": "addition"}
    }
    values = [[json.dumps(secure_transfer_1)], [json.dumps(secure_transfer_2)]]
    local_node_insert_data_to_table.delay(table_name=table_name, values=values).get()

    return table_name, secure_transfer_1_value + secure_transfer_2_value


def test_secure_transfer_without_smpc(context_id, table_with_one_column_and_ten_rows):
    from tests.algorithms.orphan_udfs import smpc_local_step

    input_table_name, input_table_name_sum = table_with_one_column_and_ten_rows

    args = [UDFArgument(kind=UDFArgumentKind.TABLE, value=input_table_name).json()]

    secure_transfer_result_table, *_ = local_node_run_udf.delay(
        command_id="1",
        context_id=context_id,
        func_name=make_unique_func_name(smpc_local_step),
        positional_args_json=args,
        keyword_args_json={},
    ).get()

    table_data_json = local_node_get_table_data.delay(
        table_name=secure_transfer_result_table
    ).get()
    table_data: TableData = TableData.parse_raw(table_data_json)
    _, result_str = table_data.data_[0]
    result = json.loads(result_str)

    expected_result = {
        "sum": {"data": input_table_name_sum, "type": "int", "operation": "addition"}
    }
    assert result == expected_result


def test_merge_secure_transfer_without_smpc(
    context_id, table_with_secure_transfer_results
):
    from tests.algorithms.orphan_udfs import smpc_global_step

    (
        secure_transfer_results_tablename,
        secure_transfer_results_values_sum,
    ) = table_with_secure_transfer_results

    args = [
        UDFArgument(
            kind=UDFArgumentKind.TABLE, value=secure_transfer_results_tablename
        ).json()
    ]

    result_table, *_ = local_node_run_udf.delay(
        command_id="1",
        context_id=context_id,
        func_name=make_unique_func_name(smpc_global_step),
        positional_args_json=args,
        keyword_args_json={},
    ).get()

    table_data_json = local_node_get_table_data.delay(table_name=result_table).get()
    table_data: TableData = TableData.parse_raw(table_data_json)
    _, result_str = table_data.data_[0]
    result = json.loads(result_str)

    expected_result = {"total_sum": secure_transfer_results_values_sum}
    assert result == expected_result
