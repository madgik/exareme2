import pytest

import json
import toml
import sys
from os import path, listdir
from pathlib import Path
from typing import List, Final

from mipengine.controller import config as controller_config
from mipengine.controller.node_tasks_handler_celery import (
    NodeTasksHandlerCelery,
    CeleryParamsDTO,
)

from mipengine.node_tasks_DTOs import TableSchema, ColumnInfo

# TODO  If folder structure changes this will not be the project parent folder anymore. Needs a more standardized way to refer to the project root..
PROJECT_ROOT = Path(__file__).parent.parent.parent
TASKS_CONTEXT_ID="contextid123"

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
    # print(f"{NODES_CONFIG_DIR=}")
    a_localnode_config_file = ""
    for filename in listdir(NODES_CONFIG_DIR):
        if "local" in filename:
            a_localnode_config_file = path.join(NODES_CONFIG_DIR, filename)
            break

    if not a_localnode_config_file:
        pytest.fail(f"Config file for localnode was not found in {NODES_CONFIG_DIR}")

    celery_params_dto = None
    with open(a_localnode_config_file) as fp:
        # print(f"{fp=}")
        tmp = toml.load(fp)
        # TODO celery params and rabbitmq params in the config files should be one..
        node_id=tmp["identifier"]
        celery_params = tmp["celery"]
        rabbitmq_params = tmp["rabbitmq"]
        monetdb_params=tmp["monetdb"]
        # print(f"{celery_params=}")
        # print(f"{rabbitmq_params=}")
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
        )

    node_task_handler = NodeTasksHandlerCelery(
        node_id= node_id,celery_params=celery_params_dto
    )
    # print(f"{node_task_handler=}")
    # node_task_handler.clean_up()

    return node_task_handler

@pytest.fixture
def a_test_table_params():
    command_id = "cmndid123"
    schema = TableSchema(
        [
            ColumnInfo(name="var1", data_type="INT"),
            ColumnInfo(name="var2", data_type="TEXT"),
        ] 
    )
    return (command_id,schema)

@pytest.fixture
def cleanup(node_task_handler):
    yield
    # teardown
    node_task_handler.clean_up(context_id=TASKS_CONTEXT_ID)

@pytest.mark.usefixtures("cleanup")
def test_create_table(node_task_handler,a_test_table_params):
    command_id = a_test_table_params[0]
    schema = a_test_table_params[1]
   
    table_name = node_task_handler.create_table(context_id=TASKS_CONTEXT_ID,command_id=command_id, schema=schema)
    print(f"{table_name=}")

    if table_name.startswith(f"table_{command_id}_{TASKS_CONTEXT_ID}_"):
        assert True
    else:
        assert False

@pytest.mark.usefixtures("cleanup")
def test_get_tables(node_task_handler,a_test_table_params):
    command_id = a_test_table_params[0]
    schema = a_test_table_params[1]
    table_name = node_task_handler.create_table(context_id=TASKS_CONTEXT_ID,command_id=command_id, schema=schema)
    tables = node_task_handler.get_tables(context_id=TASKS_CONTEXT_ID)
    if table_name in tables:
        assert True
    else:
        assert False

        
@pytest.mark.usefixtures("cleanup")
def test_get_table_schema(node_task_handler,a_test_table_params):
    command_id = a_test_table_params[0]
    schema = a_test_table_params[1]
    table_name = node_task_handler.create_table(context_id=TASKS_CONTEXT_ID,command_id=command_id, schema=schema)
    schema_result = node_task_handler.get_table_schema(table_name)
    # print(f"{schema_result=}")
    if schema_result==schema:
        assert True
    else:
        assert False
