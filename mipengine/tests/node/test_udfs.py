import inspect

import pymonetdb
import pytest

from mipengine.algorithms.demo import table_and_literal_arguments
from mipengine.algorithms.demo import func
from mipengine.algorithms.demo import tensor1
from mipengine.algorithms.demo import tensor2
from mipengine.common.node_tasks_DTOs import ColumnInfo
from mipengine.common.node_tasks_DTOs import TableSchema
from mipengine.common.node_tasks_DTOs import UDFArgument
from mipengine.tests.node import nodes_communication

local_node_id = "localnode1"
context_id = "udfs_test"
command_id = "command123"
local_node = nodes_communication.get_celery_app(local_node_id)
local_node_get_udfs = nodes_communication.get_celery_get_udfs_signature(local_node)
local_node_get_run_udf = nodes_communication.get_celery_get_run_udf_query_signature(local_node)
local_node_create_table = nodes_communication.get_celery_create_table_signature(local_node)
local_node_cleanup = nodes_communication.get_celery_cleanup_signature(local_node)


@pytest.fixture(autouse=True)
def cleanup_tables():
    yield

    local_node_cleanup.delay(context_id=context_id.lower()).get()


# def test_get_udfs():
#     udfs = [inspect.getsource(func),
#             inspect.getsource(tensor2),
#             inspect.getsource(tensor1),
#             inspect.getsource(table_and_literal_arguments)]
#
#     fetched_udfs = local_node_get_udfs.delay(algorithm_name="demo").get()
#
#     assert udfs == fetched_udfs


def test_get_run_udf_query():
    table_schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])

    table_1_name = local_node_create_table.delay(context_id=context_id,
                                                 command_id=str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                                 schema_json=table_schema.to_json()).get()

    positional_args = [UDFArgument(type="table", value=table_1_name).to_json(),
                       UDFArgument(type="literal", value="15").to_json()]

    udf_statements = local_node_get_run_udf.delay(command_id=command_id,
                                                  context_id=context_id,
                                                  func_name="demo.table_and_literal_arguments",
                                                  positional_args_json=positional_args,
                                                  keyword_args_json={}
                                                  ).get()

    assert udf_statements == ""
