import pytest
import subprocess
import json

from .node_fixtures import (
    rabbitmq_container,
    node,
    node_tasks_handler_celery,
    remove_rabbitmq_controller,
)
from mipengine.node_tasks_DTOs import TableSchema, ColumnInfo
from mipengine import DType
from mipengine.controller.node_tasks_handler_celery import ClosedBrokerConnectionError
from celery.exceptions import TimeoutError


TASKS_CONTEXT_ID = "cntxt1"


@pytest.fixture
def test_table_params():
    command_id = "cmndid1"
    schema = TableSchema(
        columns=[
            ColumnInfo(name="var1", dtype=DType.INT),
            ColumnInfo(name="var2", dtype=DType.STR),
        ]
    )
    return {"command_id": command_id, "schema": schema}


@pytest.fixture
def test_view_params():
    context_id = TASKS_CONTEXT_ID
    command_id = "0x"
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


def test_create_table(node_tasks_handler_celery, test_table_params):
    node_tasks_handler_celery = node_tasks_handler_celery["tasks_handler"]

    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]

    table_name = node_tasks_handler_celery.create_table(
        context_id=TASKS_CONTEXT_ID, command_id=command_id, schema=schema
    )

    table_name_parts = table_name.split("_")
    assert table_name_parts[0] == "normal"
    assert table_name_parts[2] == TASKS_CONTEXT_ID
    assert table_name_parts[3] == command_id


def test_get_tables(node_tasks_handler_celery, test_table_params):
    node_tasks_handler_celery = node_tasks_handler_celery["tasks_handler"]

    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = node_tasks_handler_celery.create_table(
        context_id=TASKS_CONTEXT_ID, command_id=command_id, schema=schema
    )
    tables = node_tasks_handler_celery.get_tables(context_id=TASKS_CONTEXT_ID)
    assert table_name in tables


def test_get_table_schema(node_tasks_handler_celery, test_table_params):
    node_tasks_handler_celery = node_tasks_handler_celery["tasks_handler"]
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = node_tasks_handler_celery.create_table(
        context_id=TASKS_CONTEXT_ID, command_id=command_id, schema=schema
    )
    schema_result = node_tasks_handler_celery.get_table_schema(table_name)
    assert schema_result == schema


@pytest.mark.slow
def test_broker_connection_closed_exception_get_table_schema(
    node_tasks_handler_celery, test_table_params
):
    pids = node_tasks_handler_celery["pids"]
    node_tasks_handler_celery = node_tasks_handler_celery["tasks_handler"]

    # create a test table on the node
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = node_tasks_handler_celery.create_table(
        context_id=TASKS_CONTEXT_ID, command_id=command_id, schema=schema
    )

    # Stop rabbitmq container of this node
    remove_rabbitmq_controller(pids["rabbitmq_container_id"])

    # Queue a test task quering the schema of the created table which to raise the
    # exception
    with pytest.raises(ClosedBrokerConnectionError):
        node_tasks_handler_celery.get_table_schema(table_name)


@pytest.mark.slow
def test_broker_connection_closed_exception_queue_udf(
    node_tasks_handler_celery, test_table_params
):
    pids = node_tasks_handler_celery["pids"]
    node_tasks_handler_celery = node_tasks_handler_celery["tasks_handler"]

    # create a test table on the node
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = node_tasks_handler_celery.create_table(
        context_id=TASKS_CONTEXT_ID, command_id=command_id, schema=schema
    )
    # Stop rabbitmq container of this node
    remove_rabbitmq_controller(pids["rabbitmq_container_id"])

    # queue the udf
    func_name = "relation_to_matrix_4lfu"
    args = json.dumps({"kind": 1, "value": table_name})
    keyword_args = {"rel": args}
    with pytest.raises(ClosedBrokerConnectionError):
        async_result = node_tasks_handler_celery.queue_run_udf(
            context_id=TASKS_CONTEXT_ID,
            command_id=1,
            func_name=func_name,
            positional_args=[],
            keyword_args=keyword_args,
        )


@pytest.mark.slow
def test_time_limit_exceeded_exception(node_tasks_handler_celery, test_table_params):
    pids = node_tasks_handler_celery["pids"]
    node_tasks_handler_celery = node_tasks_handler_celery["tasks_handler"]

    # create a test table
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = node_tasks_handler_celery.create_table(
        context_id=TASKS_CONTEXT_ID, command_id=command_id, schema=schema
    )

    # Stop the nodes (NOT the task queue of the node, only the celery app)
    pids["celery_app_pid"].kill()

    # Queue a task which will raise the exception
    with pytest.raises(TimeoutError):
        schema_result = node_tasks_handler_celery.get_table_schema(table_name)
