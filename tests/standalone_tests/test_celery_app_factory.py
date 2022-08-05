import random
from unittest.mock import patch

import celery
import pytest

from mipengine import AttrDict
from mipengine import DType
from mipengine.controller.celery_app import CeleryAppFactory
from mipengine.controller.celery_app import CeleryConnectionError
from mipengine.controller.celery_app import CeleryTaskTimeoutException
from mipengine.controller.celery_app import CeleryWrapper
from mipengine.node_tasks_DTOs import ColumnInfo
from mipengine.node_tasks_DTOs import TableSchema
from tests.standalone_tests.conftest import COMMON_IP
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODE1_PORT
from tests.standalone_tests.conftest import RABBITMQ_LOCALNODETMP_PORT
from tests.standalone_tests.conftest import kill_service
from tests.standalone_tests.conftest import remove_localnodetmp_rabbitmq
from tests.standalone_tests.std_output_logger import StdOutputLogger

REQUEST_ID = "testrequestid"


@pytest.fixture(scope="session")
def controller_config_dict_mock():
    controller_config = {
        "log_level": "DEBUG",
        "framework_log_level": "INFO",
        "rabbitmq": {
            "user": "user",
            "password": "password",
            "vhost": "user_vhost",
        },
    }

    return controller_config


@pytest.fixture(autouse=True, scope="session")
def patch_celery_app(controller_config_dict_mock):
    with patch(
        "mipengine.controller.celery_app.controller_config",
        AttrDict(controller_config_dict_mock),
    ):
        yield


@pytest.fixture(scope="session")
def task_signatures():
    return {
        "create_table": "mipengine.node.tasks.tables.create_table",
    }


# TODO: create slow udf to provoke TimeoutException


@pytest.mark.slow
def test_queue_task(localnode1_node_service, task_signatures):
    socket_addr = f"{COMMON_IP}:{RABBITMQ_LOCALNODE1_PORT}"
    celery_app = CeleryWrapper(socket_addr=socket_addr)

    test_table_schema = TableSchema(
        columns=[
            ColumnInfo(name="var1", dtype=DType.INT),
            ColumnInfo(name="var2", dtype=DType.STR),
        ]
    )
    async_result = celery_app.queue_task(
        task_signature=task_signatures["create_table"],
        logger=StdOutputLogger(),
        request_id=REQUEST_ID,
        context_id=get_a_random_context_id(),
        command_id="testcmndid",
        schema_json=test_table_schema.json(),
    )
    assert isinstance(async_result, celery.result.AsyncResult)


@pytest.mark.slow
def test_get_result(localnode1_node_service, task_signatures):
    socket_addr = f"{COMMON_IP}:{RABBITMQ_LOCALNODE1_PORT}"
    celery_app = CeleryWrapper(socket_addr=socket_addr)

    test_table_schema = TableSchema(
        columns=[
            ColumnInfo(name="var1", dtype=DType.INT),
            ColumnInfo(name="var2", dtype=DType.STR),
        ]
    )
    async_result = celery_app.queue_task(
        task_signature=task_signatures["create_table"],
        logger=StdOutputLogger(),
        request_id=REQUEST_ID,
        context_id=get_a_random_context_id(),
        command_id="testcmndid",
        schema_json=test_table_schema.json(),
    )

    result = celery_app.get_result(
        async_result=async_result,
        logger=StdOutputLogger(),
        timeout=30,
    )

    assert isinstance(result, str)


@pytest.mark.slow
def test_queue_task_node_down(localnodetmp_node_service, task_signatures):
    socket_addr = f"{COMMON_IP}:{RABBITMQ_LOCALNODETMP_PORT}"
    celery_app = CeleryWrapper(socket_addr=socket_addr)

    remove_localnodetmp_rabbitmq()

    test_table_schema = TableSchema(
        columns=[
            ColumnInfo(name="var1", dtype=DType.INT),
            ColumnInfo(name="var2", dtype=DType.STR),
        ]
    )

    with pytest.raises(CeleryConnectionError):
        celery_app.queue_task(
            task_signature=task_signatures["create_table"],
            logger=StdOutputLogger(),
            request_id=REQUEST_ID,
            context_id=get_a_random_context_id(),
            command_id="testcmndid",
            schema_json=test_table_schema.json(),
        )


@pytest.mark.slow
def test_get_result_node_down(localnodetmp_node_service, task_signatures):
    socket_addr = f"{COMMON_IP}:{RABBITMQ_LOCALNODETMP_PORT}"
    celery_app = CeleryWrapper(socket_addr=socket_addr)

    test_table_schema = TableSchema(
        columns=[
            ColumnInfo(name="var1", dtype=DType.INT),
            ColumnInfo(name="var2", dtype=DType.STR),
        ]
    )

    async_result = celery_app.queue_task(
        task_signature=task_signatures["create_table"],
        logger=StdOutputLogger(),
        request_id=REQUEST_ID,
        context_id=get_a_random_context_id(),
        command_id="testcmndid",
        schema_json=test_table_schema.json(),
    )

    kill_service(localnodetmp_node_service)
    remove_localnodetmp_rabbitmq()

    with pytest.raises(CeleryConnectionError):
        result = celery_app.get_result(
            async_result=async_result, logger=StdOutputLogger(), timeout=5
        )


def get_a_random_context_id() -> str:
    return str(random.randint(1, 99999))
