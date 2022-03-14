import random
import time

import pytest
from celery.exceptions import TimeoutError

from mipengine import DType
from mipengine.controller.node_tasks_handler_celery import ClosedBrokerConnectionError
from mipengine.controller.node_tasks_handler_interface import UDFKeyArguments
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFPosArguments
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_NAME
from tests.standalone_tests.conftest import _remove_rabbitmq_container
from tests.standalone_tests.conftest import kill_node_service

COMMON_TASKS_REQUEST_ID = "rqst1"


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


def test_create_table(localnode1_tasks_handler_celery, test_table_params):

    context_id = get_a_random_context_id()
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]

    table_name = localnode1_tasks_handler_celery.create_table(
        request_id=COMMON_TASKS_REQUEST_ID,
        context_id=context_id,
        command_id=command_id,
        schema=schema,
    )

    table_name_parts = table_name.split("_")
    assert table_name_parts[0] == "normal"
    assert table_name_parts[2] == context_id
    assert table_name_parts[3] == command_id


def test_get_tables(localnode1_tasks_handler_celery, test_table_params):

    context_id = get_a_random_context_id()
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = localnode1_tasks_handler_celery.create_table(
        request_id=COMMON_TASKS_REQUEST_ID,
        context_id=context_id,
        command_id=command_id,
        schema=schema,
    )
    tables = localnode1_tasks_handler_celery.get_tables(
        request_id=COMMON_TASKS_REQUEST_ID, context_id=context_id
    )

    assert table_name in tables


def test_get_table_schema(localnode1_tasks_handler_celery, test_table_params):

    context_id = get_a_random_context_id()
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = localnode1_tasks_handler_celery.create_table(
        request_id=COMMON_TASKS_REQUEST_ID,
        context_id=context_id,
        command_id=command_id,
        schema=schema,
    )
    schema_result = localnode1_tasks_handler_celery.get_table_schema(
        request_id=COMMON_TASKS_REQUEST_ID, table_name=table_name
    )

    assert schema_result == schema


@pytest.mark.slow
def test_broker_connection_closed_exception_get_table_schema(
    localnodetmp_tasks_handler_celery,
    test_table_params,
):

    # create a test table on the node
    context_id = get_a_random_context_id()
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = localnodetmp_tasks_handler_celery.create_table(
        request_id=COMMON_TASKS_REQUEST_ID,
        context_id=context_id,
        command_id=command_id,
        schema=schema,
    )

    # Stop rabbitmq container of this node
    _remove_rabbitmq_container(RABBITMQ_LOCALNODETMP_NAME)

    # Queue a test task quering the schema of the created table which to raise the
    # exception
    with pytest.raises(ClosedBrokerConnectionError):
        localnodetmp_tasks_handler_celery.get_table_schema(
            request_id=COMMON_TASKS_REQUEST_ID, table_name=table_name
        )


@pytest.mark.slow
def test_broker_connection_closed_exception_queue_udf(
    localnodetmp_tasks_handler_celery,
    test_table_params,
):

    # create a test table on the node
    context_id = get_a_random_context_id()
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = localnodetmp_tasks_handler_celery.create_table(
        request_id=COMMON_TASKS_REQUEST_ID,
        context_id=context_id,
        command_id=command_id,
        schema=schema,
    )

    # Stop rabbitmq container of this node
    _remove_rabbitmq_container(RABBITMQ_LOCALNODETMP_NAME)

    # queue the udf
    func_name = "relation_to_matrix_4lfu"
    arg = NodeTableDTO(value=table_name)
    keyword_args = UDFKeyArguments(args={"rel": arg})
    with pytest.raises(ClosedBrokerConnectionError):
        localnodetmp_tasks_handler_celery.queue_run_udf(
            request_id=COMMON_TASKS_REQUEST_ID,
            context_id=context_id,
            command_id=1,
            func_name=func_name,
            positional_args=UDFPosArguments(args=[]),
            keyword_args=keyword_args,
        )


@pytest.mark.slow
def test_time_limit_exceeded_exception(
    localnodetmp_tasks_handler_celery,
    localnodetmp_node_service,
    test_table_params,
):

    # create a test table
    context_id = get_a_random_context_id()
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = localnodetmp_tasks_handler_celery.create_table(
        request_id=COMMON_TASKS_REQUEST_ID,
        context_id=context_id,
        command_id=command_id,
        schema=schema,
    )

    # Stop the nodes (NOT the task queue of the node, only the celery app)
    kill_node_service(localnodetmp_node_service)

    # Queue a task which will raise the exception
    with pytest.raises(TimeoutError):
        localnodetmp_tasks_handler_celery.get_table_schema(
            request_id=COMMON_TASKS_REQUEST_ID, table_name=table_name
        )
        time.sleep(localnodetmp_tasks_handler_celery.tasks_timeout)


def get_a_random_context_id() -> str:
    return str(random.randint(1, 99999))
