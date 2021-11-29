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

from mipengine.controller.node_tasks_handler_celery import ClosedBrokerConnectionError
from celery.exceptions import TimeoutError


from mipengine import AttrDict
from tasks import CONTROLLER_CONFIG_DIR

import mipengine
import time
import subprocess

PROJECT_ROOT = Path(mipengine.__file__).parent.parent

TASKS_CONTEXT_ID = "contextid412"


@pytest.fixture
def node_task_handler_params():
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

    return {"node_id": node_id, "celery_params": celery_params_dto}


@pytest.fixture
def test_table_params():
    command_id = "cmndid123"
    schema = TableSchema(
        columns=[
            ColumnInfo(name="var1", dtype=DType.INT),
            ColumnInfo(name="var2", dtype=DType.STR),
        ]
    )
    return {"command_id": command_id, "schema": schema}


@pytest.fixture
def test_view_params():
    # define parameters to create a test view
    context_id = TASKS_CONTEXT_ID
    command_id = "0x"  # test_table_params["command_id"]
    pathology = "dementia"
    columns = [
        "lefthippocampus",
        "righthippocampus",
        "rightppplanumpolare",
        "leftamygdala",
        "rightamygdala",
    ]
    return {
        "context_id": context_id,
        "command_id": command_id,
        "pathology": pathology,
        "columns": columns,
    }


@pytest.fixture
def cleanup(node_task_handler_params):
    yield

    # teardown
    node_task_handler = NodeTasksHandlerCelery(
        node_id=node_task_handler_params["node_id"],
        celery_params=node_task_handler_params["celery_params"],
    )
    node_task_handler.clean_up(context_id=TASKS_CONTEXT_ID)


@pytest.mark.usefixtures("cleanup")
def test_create_table(node_task_handler_params, test_table_params):
    node_task_handler = NodeTasksHandlerCelery(
        node_id=node_task_handler_params["node_id"],
        celery_params=node_task_handler_params["celery_params"],
    )

    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]

    table_name = node_task_handler.create_table(
        context_id=TASKS_CONTEXT_ID, command_id=command_id, schema=schema
    )
    assert table_name.startswith(f"normal_{command_id}_{TASKS_CONTEXT_ID}_")


@pytest.mark.usefixtures("cleanup")
def test_get_tables(node_task_handler_params, test_table_params):
    node_task_handler = NodeTasksHandlerCelery(
        node_id=node_task_handler_params["node_id"],
        celery_params=node_task_handler_params["celery_params"],
    )

    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = node_task_handler.create_table(
        context_id=TASKS_CONTEXT_ID, command_id=command_id, schema=schema
    )
    tables = node_task_handler.get_tables(context_id=TASKS_CONTEXT_ID)
    assert table_name in tables


@pytest.mark.usefixtures("cleanup")
def test_get_table_schema(node_task_handler_params, test_table_params):
    node_task_handler = NodeTasksHandlerCelery(
        node_id=node_task_handler_params["node_id"],
        celery_params=node_task_handler_params["celery_params"],
    )

    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = node_task_handler.create_table(
        context_id=TASKS_CONTEXT_ID, command_id=command_id, schema=schema
    )
    schema_result = node_task_handler.get_table_schema(table_name)
    assert schema_result == schema


@pytest.mark.usefixtures("cleanup")
def test_broker_connection_closed_exception_get_table_schema(
    node_task_handler_params, test_table_params
):

    node_task_handler = NodeTasksHandlerCelery(
        node_id=node_task_handler_params["node_id"],
        celery_params=node_task_handler_params["celery_params"],
    )

    # create a test table on the node
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = node_task_handler.create_table(
        context_id=TASKS_CONTEXT_ID, command_id=command_id, schema=schema
    )
    # Stop rabbitmq container of this node
    node_id = node_task_handler.node_id
    cmd = f"docker stop rabbitmq-{node_id}"
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)

    # Queue a test task quering the schema of the created table which will raise the
    # exception
    with pytest.raises(ClosedBrokerConnectionError):
        node_task_handler.get_table_schema(table_name)

    # Restart the rabbitmq container of this node
    cmd = f"docker start rabbitmq-{node_id}"
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)

    # Wait until it is up again
    subcmd = "'{{.State.Health.Status}}'"
    cmd = f"docker inspect -f {subcmd} rabbitmq-{node_id}"
    result = subprocess.check_output(cmd, shell=True).decode("utf-8")
    while "healthy" not in result:
        result = subprocess.check_output(cmd, shell=True).decode("utf-8")
        time.sleep(1)


@pytest.mark.usefixtures("cleanup")
def test_broker_connection_closed_exception_queue_udf(
    node_task_handler_params, test_view_params
):

    node_task_handler = NodeTasksHandlerCelery(
        node_id=node_task_handler_params["node_id"],
        celery_params=node_task_handler_params["celery_params"],
    )

    # create a test view
    view_name = node_task_handler.create_pathology_view(
        context_id=TASKS_CONTEXT_ID,
        command_id=test_view_params["command_id"],
        pathology=test_view_params["pathology"],
        columns=test_view_params["columns"],
        filters=None,
    )

    # Stop rabbitmq container of this node
    node_id = node_task_handler.node_id
    cmd = f"docker stop rabbitmq-{node_id}"
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)

    # queue the udf
    positional_args = {"kind": 1, "value": view_name}
    positional_args_json = json.dumps(positional_args)
    with pytest.raises(ClosedBrokerConnectionError):
        async_result = node_task_handler.queue_run_udf(
            context_id=TASKS_CONTEXT_ID,
            command_id=1,
            func_name="relation_to_matrix_53ft",
            positional_args=[positional_args_json],
            keyword_args={},
        )

    # Restart the rabbitmq container of this node
    cmd = f"docker start rabbitmq-{node_id}"
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)

    # Wait until it is up again
    subcmd = "'{{.State.Health.Status}}'"
    cmd = f"docker inspect -f {subcmd} rabbitmq-{node_id}"
    result = subprocess.check_output(cmd, shell=True).decode("utf-8")
    while "healthy" not in result:
        result = subprocess.check_output(cmd, shell=True).decode("utf-8")
        time.sleep(1)


@pytest.mark.skip(
    reason="the AsyncResult.get() method raises a broken connection error only under some timing circumstances which are not correctly reproduced in this test "
)
@pytest.mark.usefixtures("cleanup")
def test_broker_connection_closed_exception_get_udf_result(
    node_task_handler_params, test_view_params
):

    node_task_handler = NodeTasksHandlerCelery(
        node_id=node_task_handler_params["node_id"],
        celery_params=node_task_handler_params["celery_params"],
    )

    # create a test view
    view_name = node_task_handler.create_pathology_view(
        context_id=TASKS_CONTEXT_ID,
        command_id=test_view_params["command_id"],
        pathology=test_view_params["pathology"],
        columns=test_view_params["columns"],
        filters=None,
    )

    # queue a udf
    positional_args = {"kind": 1, "value": view_name}
    positional_args_json = json.dumps(positional_args)
    async_result = node_task_handler.queue_run_udf(
        context_id=TASKS_CONTEXT_ID,
        command_id=1,
        func_name="relation_to_matrix_53ft",
        positional_args=[positional_args_json],
        keyword_args={},
    )

    # Stop rabbitmq container of this node
    node_id = node_task_handler.node_id
    cmd = f"docker stop rabbitmq-{node_id}"
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)

    with pytest.raises(ClosedBrokerConnectionError):
        result = node_task_handler.get_queued_udf_result(async_result)

    # Restart the rabbitmq container of this node
    cmd = f"docker start rabbitmq-{node_id}"
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)

    # Wait until it is up again
    subcmd = "'{{.State.Health.Status}}'"
    cmd = f"docker inspect -f {subcmd} rabbitmq-{node_id}"
    result = subprocess.check_output(cmd, shell=True).decode("utf-8")
    while "healthy" not in result:
        result = subprocess.check_output(cmd, shell=True).decode("utf-8")
        time.sleep(1)


@pytest.mark.usefixtures("cleanup")
def test_time_limit_exceeded_exception(node_task_handler_params, test_table_params):
    node_task_handler = NodeTasksHandlerCelery(
        node_id=node_task_handler_params["node_id"],
        celery_params=node_task_handler_params["celery_params"],
    )

    # create a test table
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = node_task_handler.create_table(
        context_id=TASKS_CONTEXT_ID, command_id=command_id, schema=schema
    )

    # Stop all nodes (NOT the task queue of the nodes, only the celery app)
    # (inv kill-node <node-id> is buggy, so we just kill all nodes..)
    node_id = node_task_handler.node_id
    cmd = f"killall celery"
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)

    # Queue a task which will raise the exception
    with pytest.raises(TimeoutError):
        schema_result = node_task_handler.get_table_schema(table_name)

    # Restart all nodes
    cmd = f"inv start-node --all"
    subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
