import uuid

import pytest

from mipengine import DType
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableData
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFArgument
from mipengine.node_tasks_DTOs import UDFArgumentKind
from mipengine.udfgen import make_unique_func_name
from tests.integration_tests import nodes_communication

local_node_id = "localnode1"
command_id = "command123"
localnode_app = nodes_communication.get_celery_app(local_node_id)
local_node_get_udf = nodes_communication.get_celery_task_signature(
    localnode_app, "get_udf"
)
# local_node_get_run_udf_query = nodes_communication.get_celery_task_signature(
#     localnode_app, "get_run_udf_query"
# )
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
def table_with_one_column_and_three_rows(context_id):
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
    values = [[1], [2], [3]]
    local_node_insert_data_to_table.delay(table_name=table_name, values=values).get()

    return table_name


def test_get_udf():
    from tests.algorithms.count_rows import get_column_rows

    fetched_udf = local_node_get_udf.delay(
        func_name=make_unique_func_name(get_column_rows)
    ).get()

    assert get_column_rows.__name__ in fetched_udf


def test_run_udf_get_column_rows(context_id, table_with_one_column_and_three_rows):
    from tests.algorithms.count_rows import get_column_rows

    args = {
        "table": UDFArgument(
            kind=UDFArgumentKind.TABLE, value=table_with_one_column_and_three_rows
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
        table_name=result_table_name
    ).get()

    table_data: TableData = TableData.parse_raw(table_data_json)

    found_data = False
    for column, data in zip(table_data.schema_.columns, table_data.data_[0]):
        if column.name == "val":
            assert data == 3
            found_data = True

    assert found_data is True


# TODOOO Integration test for run_udf with multiple state/transfer outputs
