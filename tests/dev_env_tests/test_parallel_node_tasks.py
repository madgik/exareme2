import random
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


input_table_name, input_table_name_sum = create_table_with_one_column_and_ten_rows(
    celery_app
)

kw_args_str = UDFKeyArguments(
    args={"table": NodeTableDTO(value=input_table_name)}
).json()

myFile = open("/tmp/mipengine/localnode1.out", "w")

myFile.write("")

myFile.close()


def multiple_tasks(counter, amount):
    run_udf_task_on_delay = local_node_run_udf.delay(
        command_id=random.randint(0, 10000000).__str__(),
        request_id=str(counter) + str(amount) + "request_id",
        context_id=str(counter) + str(amount),
        func_name=make_unique_func_name(high_proccess_need_step),
        positional_args_json=UDFPosArguments(args=[]).json(),
        keyword_args_json=kw_args_str,
    )
    #
    # table_schema = TableSchema(
    #     columns=[
    #         ColumnInfo(name="col1", dtype=DType.INT),
    #         ColumnInfo(name="col2", dtype=DType.FLOAT),
    #         ColumnInfo(name="col3", dtype=DType.STR),
    #     ]
    # )
    #
    # create_table_task_on_delay = local_node_create_table.delay(
    #     request_id=str(counter) + str(amount) + "request_id",
    #     context_id=str(counter) + str(amount),
    #     command_id=random.randint(0,100000000000000).__str__(),
    #     schema_json=table_schema.json(),
    # )

    return run_udf_task_on_delay


def parallel_node_tasks(amount):
    start_time = time.time()
    tasks = [multiple_tasks(counter, amount) for counter in range(amount)]
    print(len(tasks))
    count = 0
    for run_udf_task_on_delay in tasks:
        run_udf_task_on_delay.get()
        print(count)
        count += 1
    end_time = time.time() - start_time
    f = open("evaluation12.txt", "a")
    f.write({amount: end_time}.__str__())
    f.close()


def test_parallel_node_tasks_1():
    parallel_node_tasks(1)


def test_parallel_node_tasks_4():
    parallel_node_tasks(4)


def test_parallel_node_tasks_16():
    parallel_node_tasks(16)


def test_parallel_node_tasks_64():
    parallel_node_tasks(64)
