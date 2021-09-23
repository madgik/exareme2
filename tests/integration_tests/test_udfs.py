# import inspect
# import logging
# import uuid
# from string import Template
#
# import pytest
#
# from mipengine.algorithms.demo import func
# from mipengine.algorithms.demo import table_and_literal_arguments
# from mipengine.algorithms.demo import tensor1
# from mipengine.algorithms.demo import tensor2
# from mipengine.common.node_tasks_DTOs import ColumnInfo
# from mipengine.common.node_tasks_DTOs import TableSchema
# from mipengine.common.node_tasks_DTOs import UDFArgument
# from integration_tests import nodes_communication
# from integration_tests.node_db_connections import get_node_db_connection
#
# local_node_id = "localnode1"
# command_id = "command123"
# local_node = nodes_communication.get_celery_app(local_node_id)
# local_node_get_udfs = nodes_communication.get_celery_get_udfs_signature(local_node)
# local_node_get_run_udf_query = nodes_communication.get_celery_get_run_udf_query_signature(local_node)
# local_node_run_udf = nodes_communication.get_celery_run_udf_signature(local_node)
# local_node_create_table = nodes_communication.get_celery_create_table_signature(local_node)
# local_node_cleanup = nodes_communication.get_celery_cleanup_signature(local_node)
#
# @pytest.fixture()
# def context_id():
#     context_id = "test_udfs_" + uuid.uuid4().hex
#
#     yield context_id
#
#     local_node_cleanup.delay(context_id=context_id).get()
#
#
# def test_get_udfs():
#     udfs = [inspect.getsource(func),
#             inspect.getsource(tensor2),
#             inspect.getsource(tensor1),
#             inspect.getsource(table_and_literal_arguments)]
#
#     fetched_udfs = local_node_get_udfs.delay(algorithm_name="demo").get()
#
#     assert udfs == fetched_udfs
#
#
# def test_run_udf(context_id):
#     table_schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "INT"), ColumnInfo("col3", "INT")])
#
#     table_1_name = local_node_create_table.delay(context_id=context_id,
#                                                  command_id=uuid.uuid4().hex,
#                                                  schema_json=table_schema.to_json()).get()
#
#     # Add data to table_1
#     connection = get_node_db_connection(local_node_id)
#     cursor = connection.cursor()
#     cursor.execute(f"INSERT INTO {table_1_name} VALUES (1, 12,3)")
#     cursor.execute(f"INSERT INTO {table_1_name} VALUES (2, 5,5)")
#     cursor.execute(f"INSERT INTO {table_1_name} VALUES (4, 6,7)")
#     cursor.close()
#     connection.close()
#
#     positional_args = [UDFArgument(type="table", value=table_1_name).to_json(),
#                        UDFArgument(type="literal", value="15").to_json()]
#
#     local_node_run_udf.delay(command_id=command_id,
#                              context_id=context_id,
#                              func_name="demo.table_and_literal_arguments",
#                              positional_args_json=positional_args,
#                              keyword_args_json={}
#                              ).get()
#
#
# def test_get_run_udf_query(context_id):
#     table_schema = TableSchema([ColumnInfo("col1", "INT"), ColumnInfo("col2", "real"), ColumnInfo("col3", "TEXT")])
#
#     table_1_name = local_node_create_table.delay(context_id=context_id,
#                                                  command_id=uuid.uuid4().hex,
#                                                  schema_json=table_schema.to_json()).get()
#
#     positional_args = [UDFArgument(type="table", value=table_1_name).to_json(),
#                        UDFArgument(type="literal", value="15").to_json()]
#
#     func_name = "demo.table_and_literal_arguments"
#     udf_creation_statement, execution_statement = \
#         local_node_get_run_udf_query.delay(command_id=command_id,
#                                            context_id=context_id,
#                                            func_name=func_name,
#                                            positional_args_json=positional_args,
#                                            keyword_args_json={}
#                                            ).get()
#
#     udf_name = func_name.replace('.', '_') + "_" + command_id + "_" + context_id
#     assert udf_creation_statement == proper_udf_creation_statement.substitute(udf_name=udf_name)
#     output_table_name = "table_" + command_id + "_" + context_id + "_" + local_node_id
#     assert execution_statement == proper_execution_statement.substitute(output_table_name=output_table_name,
#                                                                         udf_name=udf_name,
#                                                                         input_table_name=table_1_name)
#
#
# proper_udf_creation_statement = Template("""CREATE OR REPLACE
# FUNCTION
# $udf_name(x_col1 int, x_col2 float, x_col3 text)
# RETURNS
# TABLE(col1 int, col2 float, col3 text)
# LANGUAGE PYTHON
# {
#     import pandas as pd
#     import udfio
#     x = pd.DataFrame({n: _columns[n] for n in ['x_col1', 'x_col2', 'x_col3']})
#     n = 15
#     result = x + n
#     return result
# };""")
#
# proper_execution_statement = Template("""DROP TABLE IF EXISTS $output_table_name;
# CREATE TABLE $output_table_name AS (
#     SELECT CAST('localnode1' AS varchar(50)) AS node_id, *
#     FROM
#         $udf_name(
#             (
#                 SELECT
#
#                         $input_table_name.col1,
#                         $input_table_name.col2,
#                         $input_table_name.col3
#
#                 FROM
#
#                         $input_table_name
#
#             )
#         )
# );""")
