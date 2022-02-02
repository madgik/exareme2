import json
import pickle
import uuid

import pytest
from billiard.exceptions import TimeLimitExceeded

from mipengine import DType
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFArgument
from mipengine.node_tasks_DTOs import UDFArgumentKind
from mipengine.udfgen import make_unique_func_name
from tests.dev_env_tests import nodes_communication
from tests.dev_env_tests.nodes_communication import execute_in_db

local_node_id = "localnode1"
command_id = "command123"
localnode_app = nodes_communication.get_celery_app(local_node_id)
local_node_get_udf = nodes_communication.get_celery_task_signature(
    localnode_app, "get_udf"
)
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
    context_id = "test_udfs_" + str(uuid.uuid4().hex)[:10]

    yield context_id

    local_node_cleanup.delay(context_id=context_id).get()


@pytest.fixture()
def table_with_one_column_and_ten_rows(context_id):
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
    local_node_insert_data_to_table.delay(
        context_id=context_id, table_name=table_name, values=values
    ).get()

    return table_name


def test_get_udf(context_id):
    from tests.algorithms.orphan_udfs import get_column_rows

    fetched_udf = local_node_get_udf.delay(
        context_id=context_id, func_name=make_unique_func_name(get_column_rows)
    ).get()

    assert get_column_rows.__name__ in fetched_udf


def test_run_udf_relation_to_scalar(context_id, table_with_one_column_and_ten_rows):
    from tests.algorithms.orphan_udfs import get_column_rows

    args = {
        "table": UDFArgument(
            kind=UDFArgumentKind.TABLE, value=table_with_one_column_and_ten_rows
        ).json()
    }

    (result_table_name,) = local_node_run_udf.delay(
        command_id="1",
        context_id=context_id,
        func_name=make_unique_func_name(get_column_rows),
        positional_args_json=[],
        keyword_args_json=args,
    ).get()

    table_data_json = local_node_get_table_data.delay(
        context_id=context_id, table_name=result_table_name
    ).get()

    table_data: TableData = TableData.parse_raw(table_data_json)

    assert table_data.columns[0].data[0] == 10


def test_run_udf_state_and_transfer_output(
    context_id, table_with_one_column_and_ten_rows
):
    from tests.algorithms.orphan_udfs import local_step

    args = {
        "table": UDFArgument(
            kind=UDFArgumentKind.TABLE, value=table_with_one_column_and_ten_rows
        ).json()
    }

    state_result_table, transfer_result_table = local_node_run_udf.delay(
        command_id="1",
        context_id=context_id,
        func_name=make_unique_func_name(local_step),
        positional_args_json=[],
        keyword_args_json=args,
    ).get()

    transfer_table_data_json = local_node_get_table_data.delay(
        context_id=context_id, table_name=transfer_result_table
    ).get()
    table_data: TableData = TableData.parse_raw(transfer_table_data_json)
    transfer_result_str = table_data.columns[1].data[0]
    transfer_result = json.loads(transfer_result_str)
    assert "count" in transfer_result.keys()
    assert transfer_result["count"] == 10
    assert "sum" in transfer_result.keys()
    assert transfer_result["sum"] == 55

    _, state_result_str = execute_in_db(
        local_node_id, f"SELECT * FROM {state_result_table};"
    )
    state_result = pickle.loads(state_result_str)
    assert "count" in state_result.keys()
    assert state_result["count"] == 10
    assert "sum" in state_result.keys()
    assert state_result["sum"] == 55


@pytest.mark.skip(reason="https://team-1617704806227.atlassian.net/browse/MIP-473")
@pytest.mark.slow
def test_slow_udf_exception(context_id, table_with_one_column_and_ten_rows):
    from tests.algorithms.orphan_udfs import very_slow_udf

    args = {
        "table": UDFArgument(
            kind=UDFArgumentKind.TABLE, value=table_with_one_column_and_ten_rows
        ).json()
    }

    with pytest.raises(TimeLimitExceeded):
        local_node_run_udf.delay(
            command_id="1",
            context_id=context_id,
            func_name=make_unique_func_name(very_slow_udf),
            positional_args_json=[],
            keyword_args_json=args,
        ).get()
