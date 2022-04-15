import time
import uuid

from mipengine import DType
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFKeyArguments
from mipengine.node_tasks_DTOs import UDFPosArguments
from mipengine.udfgen import make_unique_func_name
from tests.algorithms.orphan_udfs import high_proccess_need_step
from tests.dev_env_tests.nodes_communication import get_celery_app
from tests.dev_env_tests.nodes_communication import get_celery_task_signature
from tests.standalone_tests.test_udfs import create_table_with_one_column_and_ten_rows

celery_app = get_celery_app("localnode1")
local_node_create_table = get_celery_task_signature(celery_app, "create_table")
local_node_get_tables = get_celery_task_signature(celery_app, "get_tables")
local_node_run_udf = get_celery_task_signature(celery_app, "run_udf")
local_node_cleanup = get_celery_task_signature(celery_app, "clean_up")

request_id = "testtables" + uuid.uuid4().hex + "request"
context_id = "testtables" + uuid.uuid4().hex

time_amount = {}


def multiple_tasks():
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )

    create_table_task_on_delay = local_node_create_table.delay(
        request_id="testtables" + uuid.uuid4().hex + "request",
        context_id="testtables" + uuid.uuid4().hex,
        command_id=uuid.uuid4().hex,
        schema_json=table_schema.json(),
    )

    print("create table")

    get_tables_task_on_delay = local_node_get_tables.delay(
        request_id=request_id,
        context_id=context_id,
    )

    print("get tables")

    input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
        celery_app
    )

    kw_args_str = UDFKeyArguments(
        args={"table": NodeTableDTO(value=input_table_name)}
    ).json()

    run_udf_task_on_delay = local_node_run_udf.delay(
        command_id="1",
        request_id=request_id,
        context_id=context_id,
        func_name=make_unique_func_name(high_proccess_need_step),
        positional_args_json=UDFPosArguments(args=[]).json(),
        keyword_args_json=kw_args_str,
    )

    print("run udf")

    return create_table_task_on_delay, get_tables_task_on_delay, run_udf_task_on_delay


def parallel_node_tasks(amount):
    start_time = time.time()
    results = [multiple_tasks() for _ in range(0, amount)]
    for (
        create_table_task_on_delay,
        get_tables_task_on_delay,
        run_udf_task_on_delay,
    ) in results:
        create_table_task_on_delay.get()
        get_tables_task_on_delay.get()
        run_udf_task_on_delay.get()

    end_time = time.time() - start_time
    time_amount[amount] = end_time
    raise ValueError(time_amount)


def test_parallel_node_tasks_10():
    parallel_node_tasks(10)


#
#
# def test_parallel_node_tasks_15():
#     parallel_node_tasks(15)
#
# def test_parallel_node_tasks_20():
#     parallel_node_tasks(20)
#
#
# def test_parallel_node_tasks_50():
#     parallel_node_tasks(50)
#
#
# def test_parallel_node_tasks_100():
#     parallel_node_tasks(100)
