import json
import random
import time

import pytest

from mipengine import DType
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableSchema
from tests.dev_env_tests.nodes_communication import get_celery_app
from tests.dev_env_tests.nodes_communication import get_celery_task_signature

local_node = get_celery_app("localnode1")
local_node_create_table = get_celery_task_signature(local_node, "create_table")


def pararrel_table_creation(counter):
    table_schema = TableSchema(
        columns=[
            ColumnInfo(name="col1", dtype=DType.INT),
            ColumnInfo(name="col2", dtype=DType.FLOAT),
            ColumnInfo(name="col3", dtype=DType.STR),
        ]
    )

    create_table_task_on_delay = local_node_create_table.delay(
        request_id=str(counter) + str(counter) + "request_id",
        context_id=str(counter) + str(counter),
        command_id=random.randint(0, 100000000000000).__str__(),
        schema_json=table_schema.json(),
    )

    return create_table_task_on_delay


def test_parallel_node_tasks():
    start_time = time.time()
    tasks = [pararrel_table_creation(counter) for counter in range(1000)]
    for run_udf_task_on_delay in tasks:
        run_udf_task_on_delay.get()
    end_time = time.time() - start_time


def routine():
    id = random.randint(0, 123456)
    request_id = "7222076" + id.__str__()
    context_id = "2522180" + id.__str__()
    table_name = f"normal_localnode1_{context_id}_7_0"
    opener = open("input0.txt", "r")
    x = json.loads(opener.read())
    opener.close()

    for task, kwargs in x:
        if not kwargs:
            continue

        if task == "run_udf":
            keyword_args_json = kwargs["keyword_args_json"]
            keyword_args_json = keyword_args_json.replace("2522180", context_id)
            get_celery_task_signature(local_node, task).delay(
                command_id=kwargs["command_id"],
                request_id=request_id,
                context_id=context_id,
                func_name=kwargs["func_name"],
                positional_args_json=kwargs["positional_args_json"],
                keyword_args_json=keyword_args_json,
            ).get()
        elif task == "create_data_model_view":
            columns = kwargs["columns"]
            while "row_id" in columns:
                columns.remove("row_id")
            get_celery_task_signature(local_node, task).delay(
                request_id=request_id,
                context_id=context_id,
                command_id=kwargs["command_id"],
                data_model=kwargs["data_model"],
                datasets=kwargs["datasets"],
                columns=columns,
            ).get()
        elif task == "get_table_data":
            get_celery_task_signature(local_node, task).delay(
                request_id=request_id, table_name=table_name
            ).get()
        elif task == "clean_up":
            get_celery_task_signature(local_node, task).delay(
                request_id=request_id, context_id=context_id
            ).get()
        elif task == "get_node_datasets_per_data_model":
            get_celery_task_signature(local_node, task).delay(
                request_id=request_id
            ).get()
        elif task == "get_node_info":
            get_celery_task_signature(local_node, task).delay(
                request_id=request_id
            ).get()
        elif task == "get_data_model_cdes":
            get_celery_task_signature(local_node, task).delay(
                request_id=request_id, data_model=kwargs["data_model"]
            ).get()


@pytest.mark.parametrize("counter", range(100))
def pca_routine(counter):
    routine()
