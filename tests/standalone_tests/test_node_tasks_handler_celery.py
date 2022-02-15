import pytest
from celery.exceptions import TimeoutError

from mipengine import DType
from mipengine.controller.node_tasks_handler_celery import ClosedBrokerConnectionError
from mipengine.controller.node_tasks_handler_interface import UDFKeyArguments
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import NodeTableDTO
from mipengine.node_tasks_DTOs import TableSchema
from mipengine.node_tasks_DTOs import UDFPosArguments

from .conftest import kill_node_service
from .conftest import remove_tmp_localnode_rabbitmq

TASKS_CONTEXT_ID = "cntxt1"
TASKS_REQUEST_ID = "rqst1"


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
    request_id = TASKS_REQUEST_ID
    context_id = TASKS_CONTEXT_ID
    command_id = "0x"
    data_model = "dementia:0.1"
    columns = [
        "lefthippocampus",
        "righthippocampus",
        "rightppplanumpolare",
        "leftamygdala",
        "rightamygdala",
    ]
    return {
        "request_id": request_id,
        "context_id": context_id,
        "command_id": command_id,
        "data_model": data_model,
        "columns": columns,
    }


def test_create_table(
    globalnode_tasks_handler_celery, use_globalnode_database, test_table_params
):
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]

    table_name = globalnode_tasks_handler_celery.create_table(
        request_id=TASKS_REQUEST_ID,
        context_id=TASKS_CONTEXT_ID,
        command_id=command_id,
        schema=schema,
    )

    table_name_parts = table_name.split("_")
    assert table_name_parts[0] == "normal"
    assert table_name_parts[2] == TASKS_CONTEXT_ID
    assert table_name_parts[3] == command_id


def test_get_tables(
    globalnode_tasks_handler_celery, use_globalnode_database, test_table_params
):
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = globalnode_tasks_handler_celery.create_table(
        request_id=TASKS_REQUEST_ID,
        context_id=TASKS_CONTEXT_ID,
        command_id=command_id,
        schema=schema,
    )
    tables = globalnode_tasks_handler_celery.get_tables(
        request_id=TASKS_REQUEST_ID, context_id=TASKS_CONTEXT_ID
    )
    assert table_name in tables


def test_get_table_schema(
    globalnode_tasks_handler_celery, use_globalnode_database, test_table_params
):
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = globalnode_tasks_handler_celery.create_table(
        request_id=TASKS_REQUEST_ID,
        context_id=TASKS_CONTEXT_ID,
        command_id=command_id,
        schema=schema,
    )
    schema_result = globalnode_tasks_handler_celery.get_table_schema(
        request_id=TASKS_REQUEST_ID, table_name=table_name
    )
    assert schema_result == schema


@pytest.mark.slow
def test_broker_connection_closed_exception_get_table_schema(
    tmp_localnode_tasks_handler_celery,
    test_table_params,
):
    # create a test table on the node
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = tmp_localnode_tasks_handler_celery.create_table(
        request_id=TASKS_REQUEST_ID,
        context_id=TASKS_CONTEXT_ID,
        command_id=command_id,
        schema=schema,
    )

    remove_tmp_localnode_rabbitmq()

    # Queue a test task quering the schema of the created table which to raise the
    # exception
    with pytest.raises(ClosedBrokerConnectionError):
        tmp_localnode_tasks_handler_celery.get_table_schema(
            request_id=TASKS_REQUEST_ID, table_name=table_name
        )


@pytest.mark.slow
def test_broker_connection_closed_exception_queue_udf(
    tmp_localnode_tasks_handler_celery,
    test_table_params,
):
    # create a test table on the node
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = tmp_localnode_tasks_handler_celery.create_table(
        request_id=TASKS_REQUEST_ID,
        context_id=TASKS_CONTEXT_ID,
        command_id=command_id,
        schema=schema,
    )

    remove_tmp_localnode_rabbitmq()

    # queue the udf
    func_name = "relation_to_matrix_4lfu"
    arg = NodeTableDTO(value=table_name)
    keyword_args = UDFKeyArguments(args={"rel": arg})
    with pytest.raises(ClosedBrokerConnectionError):
        _ = tmp_localnode_tasks_handler_celery.queue_run_udf(
            request_id=TASKS_REQUEST_ID,
            context_id=TASKS_CONTEXT_ID,
            command_id=1,
            func_name=func_name,
            positional_args=UDFPosArguments(args=[]),
            keyword_args=keyword_args,
        )


@pytest.mark.slow
def test_time_limit_exceeded_exception(
    tmp_localnode_tasks_handler_celery,
    tmp_localnode_node_service,
    test_table_params,
):
    # create a test table
    command_id = test_table_params["command_id"]
    schema = test_table_params["schema"]
    table_name = tmp_localnode_tasks_handler_celery.create_table(
        request_id=TASKS_REQUEST_ID,
        context_id=TASKS_CONTEXT_ID,
        command_id=command_id,
        schema=schema,
    )

    kill_node_service(tmp_localnode_node_service)

    # Queue a task which will raise the exception
    with pytest.raises(TimeoutError):
        tmp_localnode_tasks_handler_celery.get_table_schema(
            request_id=TASKS_REQUEST_ID, table_name=table_name
        )
