import json
import random

import pytest

from tests.dev_env_tests.nodes_communication import get_celery_app
from tests.dev_env_tests.nodes_communication import get_celery_task_signature

local_node = get_celery_app("localnode1")


def routine():
    id = random.randint(0, 123456)
    request_id = "7222076" + id.__str__()
    context_id = "2522180" + id.__str__()
    table_name = f"normal_localnode1_{context_id}_7_0"
    opener = open("input0.txt", "r")
    tasks = json.loads(opener.read())
    opener.close()

    for task, kwargs in tasks:
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
            get_celery_task_signature(local_node, task).delay(
                request_id=request_id,
                context_id=context_id,
                command_id=kwargs["command_id"],
                data_model=kwargs["data_model"],
                datasets=kwargs["datasets"],
                columns=kwargs["columns"],
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
def test_pca_routine(counter):
    routine()
