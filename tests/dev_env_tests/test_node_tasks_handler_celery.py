import pytest

import json
import toml
from os import path, listdir
from pathlib import Path
from typing import List, Final

from mipengine.controller.node_tasks_handler_celery import NodeTasksHandlerCelery
from mipengine.controller.node_tasks_handler_celery import CeleryParamsDTO

from mipengine.node_tasks_DTOs import TableSchema, ColumnInfo
from mipengine import DType

from mipengine import AttrDict
from tasks import CONTROLLER_CONFIG_DIR

import mipengine

PROJECT_ROOT = Path(mipengine.__file__).parent.parent

TASKS_CONTEXT_ID = "contextid123"

# This fixture returns a celery app object upon which tasks will be triggered in order
# to test them
@pytest.fixture
def node_task_handler():
    # The instantiation of a celery app object requires the following parameters: ip,port
    # user, password, vhost and the transport options(max_retries,interval_start,
    # interval_step,interval_max)
    # Node config: ip and port are read from the node config
    # Controller config: user, password, vhost and the transport options are read from
    # the controller config
    NODES_CONFIG_DIR = path.join(path.join(PROJECT_ROOT, "configs"), "nodes")
    a_localnode_config_file = ""
    for filename in listdir(NODES_CONFIG_DIR):
        if "local" in filename:
            a_localnode_config_file = path.join(NODES_CONFIG_DIR, filename)
            break

    if not a_localnode_config_file:
        pytest.fail(f"Config file for localnode was not found in {NODES_CONFIG_DIR}")

    with open(CONTROLLER_CONFIG_DIR / "controller.toml") as fp:
        controller_config = AttrDict(toml.load(fp))

    celery_params_dto = None
    with open(a_localnode_config_file) as fp:
        tmp = toml.load(fp)
        # TODO celery params and rabbitmq params in the config files should be one..
        node_id = tmp["identifier"]
        celery_params = tmp["celery"]
        rabbitmq_params = tmp["rabbitmq"]
        monetdb_params = tmp["monetdb"]
        celery_params_dto = CeleryParamsDTO(
            task_queue_domain=rabbitmq_params["ip"],
            task_queue_port=rabbitmq_params["port"],
            db_domain=monetdb_params["ip"],
            db_port=monetdb_params["port"],
            user=controller_config.rabbitmq.user,
            password=controller_config.rabbitmq.password,
            vhost=controller_config.rabbitmq.vhost,
            max_retries=controller_config.rabbitmq.celery_tasks_max_retries,
            interval_start=controller_config.rabbitmq.celery_tasks_interval_start,
            interval_step=controller_config.rabbitmq.celery_tasks_interval_step,
            interval_max=controller_config.rabbitmq.celery_tasks_interval_max,
            tasks_timeout=controller_config.rabbitmq.celery_tasks_timeout,
        )

    return NodeTasksHandlerCelery(node_id=node_id, celery_params=celery_params_dto)


@pytest.fixture
def a_test_table_params():
    command_id = "cmndid123"
    schema = TableSchema(
        columns=[
            ColumnInfo(name="var1", dtype=DType.INT),
            ColumnInfo(name="var2", dtype=DType.STR),
        ]
    )
    return (command_id, schema)


@pytest.fixture
def cleanup(node_task_handler):
    yield
    # teardown
    node_task_handler.clean_up(context_id=TASKS_CONTEXT_ID)


@pytest.mark.usefixtures("cleanup")
def test_create_table(node_task_handler, a_test_table_params):
    command_id = a_test_table_params[0]
    schema = a_test_table_params[1]

    table_name = node_task_handler.create_table(
        context_id=TASKS_CONTEXT_ID, command_id=command_id, schema=schema
    )

    table_name_parts = table_name.split("_")
    assert table_name_parts[0] == "normal"
    assert table_name_parts[2] == TASKS_CONTEXT_ID
    assert table_name_parts[3] == command_id


@pytest.mark.usefixtures("cleanup")
def test_get_tables(node_task_handler, a_test_table_params):
    command_id = a_test_table_params[0]
    schema = a_test_table_params[1]
    table_name = node_task_handler.create_table(
        context_id=TASKS_CONTEXT_ID, command_id=command_id, schema=schema
    )
    tables = node_task_handler.get_tables(context_id=TASKS_CONTEXT_ID)
    assert table_name in tables


@pytest.mark.usefixtures("cleanup")
def test_get_table_schema(node_task_handler, a_test_table_params):
    command_id = a_test_table_params[0]
    schema = a_test_table_params[1]
    table_name = node_task_handler.create_table(
        context_id=TASKS_CONTEXT_ID, command_id=command_id, schema=schema
    )
    schema_result = node_task_handler.get_table_schema(table_name)
    assert schema_result == schema
