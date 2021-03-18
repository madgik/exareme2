import inspect
from string import Template

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
from mipengine.tests.node.node_db_connections import get_node_db_connection

local_node_id = "localnode1"
context_id = "udfs_test"
command_id = "command123"
local_node = nodes_communication.get_celery_app(local_node_id)
local_node_get_udfs = nodes_communication.get_celery_get_udfs_signature(local_node)
local_node_get_run_udf_query = nodes_communication.get_celery_get_run_udf_query_signature(local_node)
local_node_run_udf = nodes_communication.get_celery_run_udf_signature(local_node)
local_node_create_table = nodes_communication.get_celery_create_table_signature(local_node)
local_node_cleanup = nodes_communication.get_celery_cleanup_signature(local_node)

#TODO:Fix this one to make proper cleanup.
#
#
# @pytest.fixture(autouse=True)
# def cleanup_tables():
#     yield
#
#     local_node_cleanup.delay(context_id=context_id.lower()).get()


def test_get_udfs():
    udfs = [inspect.getsource(func),
            inspect.getsource(tensor2),
            inspect.getsource(tensor1),
            inspect.getsource(table_and_literal_arguments)]

    fetched_udfs = local_node_get_udfs.delay(algorithm_name="demo").get()

    assert udfs == fetched_udfs


def test_run_udf():
    table_schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "INT"), ColumnInfo("col3", "INT")])

    table_1_name = local_node_create_table.delay(context_id=context_id,
                                                 command_id=str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                                 schema_json=table_schema.to_json()).get()

    # Add data to table_1
    connection = get_node_db_connection(local_node_id)
    cursor = connection.cursor()
    cursor.execute(f"INSERT INTO {table_1_name} VALUES (1, 12,3)")
    cursor.execute(f"INSERT INTO {table_1_name} VALUES (2, 5,5)")
    cursor.execute(f"INSERT INTO {table_1_name} VALUES (4, 6,7)")
    connection.commit()
    connection.close()

    positional_args = [UDFArgument(type="table", value=table_1_name).to_json(),
                       UDFArgument(type="literal", value="15").to_json()]

    udf_table_name = local_node_run_udf.delay(command_id=command_id,
                                              context_id=context_id,
                                              func_name="demo.table_and_literal_arguments",
                                              positional_args_json=positional_args,
                                              keyword_args_json={}
                                              ).get()


def test_get_run_udf_query():
    table_schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "FLOAT"), ColumnInfo("col3", "TEXT")])

    table_1_name = local_node_create_table.delay(context_id=context_id,
                                                 command_id=str(pymonetdb.uuid.uuid1()).replace("-", ""),
                                                 schema_json=table_schema.to_json()).get()

    positional_args = [UDFArgument(type="table", value=table_1_name).to_json(),
                       UDFArgument(type="literal", value="15").to_json()]

    udf_creation_statement, execution_statement = \
        local_node_get_run_udf_query.delay(command_id=command_id,
                                           context_id=context_id,
                                           func_name="demo.table_and_literal_arguments",
                                           positional_args_json=positional_args,
                                           keyword_args_json={}
                                           ).get()

    assert udf_creation_statement == proper_udf_creation_statement
    assert execution_statement == proper_execution_statement.substitute(input_table_name=table_1_name)


proper_udf_creation_statement = """CREATE OR REPLACE
FUNCTION
demo_table_and_literal_arguments_command123_udfs_test(x_col1 int, x_col2 float, x_col3 text)
RETURNS
TABLE(col1 int, col2 float, col3 text)
LANGUAGE PYTHON
{
    import pandas as pd
    import udfio
    x = pd.DataFrame({n: _columns[n] for n in ['x_col1', 'x_col2', 'x_col3']})
    n = 15
    result = x + n
    return result
};"""

proper_execution_statement = Template("""DROP TABLE IF EXISTS table_command123_udfs_test_localnode1;
CREATE TABLE table_command123_udfs_test_localnode1 AS (
    SELECT CAST('localnode1' AS varchar(50)) AS node_id, *
    FROM
        demo_table_and_literal_arguments_command123_udfs_test(
            (
                SELECT

                        $input_table_name.col1, 
                        $input_table_name.col2, 
                        $input_table_name.col3

                FROM
                    $input_table_name
            )
        )
);""")
